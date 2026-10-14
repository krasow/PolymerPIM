# Broadcasting.  The Broadcasted tree stays lazy and lowers to a single RPN
# program, so these also pin the one-pass-per-expression property.

@testset "single operator broadcasts" begin
    a_data = Int32.(collect(1:N))
    b_data = fill(Int32(3), N)
    a = DPUVector(a_data)
    b = DPUVector(b_data)

    @test Array(a .+ b)   == a_data .+ b_data
    @test Array(a .- b)   == a_data .- b_data
    @test Array(a .* b)   == a_data .* b_data
    @test Array(a .* 2)   == a_data .* Int32(2)
    @test Array(2 .* a)   == a_data .* Int32(2)
    @test Array(.-a)      == -a_data
    @test Array(abs.(.-a)) == a_data
    @test Array(a .>> 1)  == a_data .>> Int32(1)
    @test Array(a .< b)   == Int32.(a_data .< b_data)
end

@testset "broadcast lowers to one program" begin
    n = 512
    av = Int32.(collect(1:n)); bv = Int32.(collect(n:-1:1))
    cv = fill(Int32(3), n)
    a = DPUVector(av); b = DPUVector(bv); c = DPUVector(cv)

    @test Array(a .+ b .* c) == av .+ bv .* cv
    @test Array(abs.(a .- b) .+ 1) == abs.(av .- bv) .+ 1
    @test Array((a .+ 1) .* (b .- 2)) == (av .+ 1) .* (bv .- 2)
    @test Array(a .* a) == av .* av
    @test Array(.-a) == .-av
    @test Array(a .>> 2) == av .>> 2
    @test Array(2 .* a .+ 1) == 2 .* av .+ 1

    # comparisons inside a broadcast
    @test Array(a .> b) == Int32.(av .> bv)
    @test Array(a .<= b) == Int32.(av .<= bv)
    @test Array(a .== b) == Int32.(av .== bv)

    # ifelse is the broadcast spelling of select
    @test Array(ifelse.(a .> b, a, b)) == max.(av, bv)
end

@testset "launch-time scalar leaves" begin
    av = Int32.(collect(1:256))
    a = DPUVector(av)

    @test Array(abs2.(a .- 7)) == abs2.(av .- Int32(7))

    # A captured subtree keeps one launch slot when reused. Separately captured
    # equal-valued occurrences remain structurally distinct.
    leaf = PolymerPIM._capture_scalars(Base.broadcasted(-, a, 7))
    shared = Base.broadcasted(+, leaf, leaf)
    _, primary, operands, scalars = PolymerPIM._lower_tree(
        shared; consume = false)
    @test primary === a
    @test isempty(operands)
    @test scalars == Int32[7]

    separate = Base.broadcasted(+, Base.broadcasted(-, a, 7),
                                 Base.broadcasted(-, a, 7))
    _, _, _, scalars = PolymerPIM._lower_tree(separate; consume = false)
    @test scalars == Int32[7, 7]

    # Runtime values never enter the opcode stream or its JIT cache key.
    same = @code_jitted (a .- 7) .+ (a .- 7)
    different = @code_jitted (a .- 7) .+ (a .- 8)
    @test same.ops == different.ops
    @test same.hash == different.hash
end

@testset "a whole expression is one kernel pass" begin
    n = 512
    vs = [DPUVector(Int32.(collect(1:n) .+ k)) for k in 1:8]
    want = sum(Int32.(collect(1:n) .+ k) for k in 1:8)

    PolymerPIM.sync()
    before = PolymerPIM.stat_compute_launches()
    fused = PolymerPIM.stat_vertical_fusions()
    got = Array(vs[1] .+ vs[2] .+ vs[3] .+ vs[4] .+
                vs[5] .+ vs[6] .+ vs[7] .+ vs[8])
    passes = PolymerPIM.stat_compute_launches() - before

    @test got == want
    # One program, so one pass -- and no reliance on the fusion pass to get
    # there.  The eager operator spelling of the same thing costs 7.
    @test passes == 1
    @test PolymerPIM.stat_vertical_fusions() - fused == 0
end

# The runtime only declines to merge, so the lowering splits the tree itself.
@testset "an expression wider than the operand table splits" begin
    n = 512
    k = MAX_VFUSE_INPUTS + 5
    data = [Int32.(collect(1:n) .+ d) for d in 1:k]
    vs = [DPUVector(copy(d)) for d in data]
    q = Int32.(collect(1:k))
    want = sum(data[d] .* q[d] for d in 1:k)

    dot(vecs) = foldl((acc, d) -> acc .+ vecs[d] .* q[d], 2:k;
                      init = vecs[1] .* q[1])

    PolymerPIM.sync()
    before = PolymerPIM.stat_compute_launches()
    got = Array(dot(vs))
    passes = PolymerPIM.stat_compute_launches() - before

    @test got == want
    # One cut: the table's worth in the first program, the rest in the second.
    @test passes == 2

    # The same through a reduction terminal, which is vector_search's shape.
    PolymerPIM.sync()
    before = PolymerPIM.stat_compute_launches()
    value, index = findmax(dot(vs))
    PolymerPIM.sync()

    @test value == maximum(want)
    @test index == argmax(want)
    @test PolymerPIM.stat_compute_launches() - before == 2
end

# Wide enough to cut twice, which is where a nested lowering could release an
# intermediate the enclosing program still reads.
@testset "an expression wide enough to cut twice" begin
    n = 512
    k = 2 * (MAX_VFUSE_INPUTS + 1) + 2
    data = [Int32.(collect(1:n) .+ d) for d in 1:k]
    vs = [DPUVector(copy(d)) for d in data]
    q = Int32.(collect(1:k))
    want = sum(data[d] .* q[d] for d in 1:k)

    PolymerPIM.sync()
    before = PolymerPIM.stat_compute_launches()
    got = Array(foldl((acc, d) -> acc .+ vs[d] .* q[d], 2:k;
                      init = vs[1] .* q[1]))
    passes = PolymerPIM.stat_compute_launches() - before

    @test got == want
    @test passes == 3
end

# The GC cannot see MRAM, so an unreleased cut result accumulates until the
# DPUs are full -- vector_search died at query 48 of 60 that way.
@testset "split intermediates are released" begin
    n = 512
    k = MAX_VFUSE_INPUTS + 5
    vs = [DPUVector(Int32.(collect(1:n) .+ d)) for d in 1:k]
    q = Int32.(collect(1:k))
    dot(vecs) = foldl((acc, d) -> acc .+ vecs[d] .* q[d], 2:k;
                      init = vecs[1] .* q[1])

    for _ in 1:8
        findmax(dot(vs))
        @test length(PolymerPIM._CUT_TEMPS) <= 1   # one expression's worth
    end
    PolymerPIM.sync()
    @test isempty(PolymerPIM._CUT_TEMPS)
end

@testset "in-place broadcast writes through" begin
    n = 512
    av = Int32.(collect(1:n)); bv = Int32.(collect(n:-1:1))
    a = DPUVector(av); b = DPUVector(bv)

    d = DPUVector(n)
    PolymerPIM.sync()
    before = PolymerPIM.stat_compute_launches()
    d .= a .+ b .* 2
    @test Array(d) == av .+ bv .* 2
    @test PolymerPIM.stat_compute_launches() - before == 1

    # DPUVector is a handle type, so `.=` must update the buffer rather than
    # rebind -- an alias has to observe the write.
    alias = d
    d .= a .* 3
    @test Array(alias) == av .* 3

    # the destination may appear in its own expression
    c = DPUVector(copy(av))
    c .= c .+ 100
    @test Array(c) == av .+ 100

    e = DPUVector(copy(av))
    e .= e .* e
    @test Array(e) == av .* av

    @test_throws DimensionMismatch (DPUVector(8) .= a .+ b)
end

@testset "broadcast rejects what it cannot lower" begin
    a = DPUVector(Int32.(collect(1:64)))
    @test_throws ArgumentError Array(sqrt.(a))
    # Lazily built, so it raises at first use -- and only once: a failed
    # expression must not be retried by a later, unrelated sync().
    bad = sqrt.(a)
    @test_throws ArgumentError Array(bad)
    @test sync() === nothing
    @test_throws ArgumentError Array(sin.(a))
end

# abs2 lowers to `sqr`, which is OP_DUP + OP_MUL, so its argument is computed
# once.  Spelled `x .* x` the lowering has no CSE and computes it twice -- which
# is why knn uses abs2.
@testset "abs2 loads its argument once" begin
    a = DPUVector(Int32.(1:N)); b = DPUVector(Int32.(N:-1:1))
    O = PolymerPIM.Internal.Opcodes

    sq = @code_jitted abs2.(a .- b)
    @test O.OP_DUP in sq.ops
    @test count(==(O.OP_SUB), sq.ops) == 1

    naive = @code_jitted (a .- b) .* (a .- b)
    @test count(==(O.OP_SUB), naive.ops) == 2
    @test !(O.OP_DUP in naive.ops)
    @test length(sq.ops) < length(naive.ops)

    # No abs2 opcode exists: it is those two.
    @test !isdefined(O, :OP_ABS2)

    @test Array(abs2.(a .- b)) == (Array(a) .- Array(b)) .^ 2
end

@testset "lazy zeros fold out of an accumulator" begin
    n = 512
    av = Int32.(collect(1:n)); bv = Int32.(collect(n:-1:1))
    a = DPUVector(av); b = DPUVector(bv)
    PolymerPIM.sync()

    @test length(PolymerPIM.zeros(Int32, n)) == n
    @test Array(DPUVector(PolymerPIM.zeros(Int32, 8))) == zeros(Int32, 8)
    @test_throws ArgumentError PolymerPIM.zeros(Int32, -1)

    # Seeding with the zeros emits the same program as seeding from the first
    # term, so starting at 1 costs nothing.
    seeded = abs2.(a .- Int32(3))
    seeded = seeded .+ abs2.(b .- Int32(5))
    folded = PolymerPIM.zeros(Int32, n)
    for (vec, scalar) in ((a, Int32(3)), (b, Int32(5)))
        folded = folded .+ abs2.(vec .- scalar)
    end
    @test code_jitted(folded).ops == code_jitted(seeded).ops

    expected = abs2.(av .- 3) .+ abs2.(bv .- 5)
    @test Array(folded) == expected
    @test Int64(sum(folded)[]) == sum(Int64.(expected))
end

@testset "zeros materialise outside the additive fold" begin
    n = 64
    xv = Int32.(collect(1:n))
    x = DPUVector(xv)
    PolymerPIM.sync()

    @test Array(PolymerPIM.zeros(Int32, n) .* x) == zeros(Int32, n)
    @test Array(x .* PolymerPIM.zeros(Int32, n)) == zeros(Int32, n)
    @test Array(PolymerPIM.zeros(Int32, n) .- x) == -xv
    @test Int64(minimum(PolymerPIM.zeros(Int32, n))[]) == 0
    @test Int64(sum(PolymerPIM.zeros(Int32, n))[]) == 0
end
