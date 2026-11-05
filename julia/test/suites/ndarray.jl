# DPUArray beyond one dimension: a shape over the same flat column-major
# storage, so everything elementwise works unchanged.

const DIMS2 = (64, N ÷ 64)
const DIMS3 = (16, 8, N ÷ 128)

@testset "construction and conversion" begin
    for dims in (DIMS2, DIMS3)
        host = reshape(Int32.(1:N), dims)
        a = DPUArray(host)
        @test a isa DPUArray{length(dims)}
        @test size(a) == dims
        @test length(a) == N
        @test ndims(a) == length(dims)
        @test axes(a) == axes(host)
        @test Array(a) == host
        @test Array(a) isa Array{Int32,length(dims)}
        @test a[2] == host[2]
        @test a[ntuple(i -> 2, length(dims))...] == host[ntuple(i -> 2, length(dims))...]
    end
    @test DPUMatrix(Int64[1 2; 3 4]) isa DPUMatrix
    @test Matrix(DPUArray(Int32[1 2; 3 4])) == Int32[1 2; 3 4]
    @test size(DPUArray(3, 4, 5)) == (3, 4, 5)
    @test size(DPUArray((3, 4))) == (3, 4)
    @test size(similar(DPUArray(reshape(Int32.(1:12), 3, 4)))) == (3, 4)
    @test_throws ArgumentError DPUArray(zeros(Int32, 2, 2, 2, 2))
    @test_throws ArgumentError DPUArray(2, 2, 2, 2)
end

@testset "reshape shares storage" begin
    host = Int32.(1:N)
    v = DPUVector(host)
    m = reshape(v, DIMS2)
    @test m isa DPUMatrix
    @test Array(m) == reshape(host, DIMS2)
    @test size(reshape(v, 16, :, 8)) == (16, N ÷ 128, 8)
    @test vec(m) isa DPUVector
    @test Array(vec(m)) == host
    add!(m, 1)                     # written through the matrix
    @test Array(v) == host .+ 1    # seen through the vector
    @test_throws DimensionMismatch reshape(v, 3, :)
    @test_throws DimensionMismatch reshape(v, 2, N)
    @test_throws ArgumentError reshape(v, 2, 2, 2, :)
end

@testset "elementwise" begin
    ah = reshape(Int32.(1:N), DIMS3)
    bh = reshape(Int32.(N:-1:1), DIMS3)
    a, b = DPUArray(ah), DPUArray(bh)

    @test Array(a + b) == ah + bh
    @test size(a + b) == DIMS3
    @test Array(a * 3) == ah * 3
    @test Array(-a) == -ah
    @test Array(a .* 2 .+ b) == ah .* 2 .+ bh
    @test Array(abs.(a .- b)) == abs.(ah .- bh)
    @test Array(ifelse.(a .< b, a, b)) == min.(ah, bh)

    lazy = a .+ b
    @test lazy isa DpuLazy
    @test size(lazy) == DIMS3
    @test DPUArray(lazy) isa DPUArray{3}
    @test_throws ArgumentError DPUVector(a .+ 1)

    dest = similar(a)
    dest .= a .- b
    @test Array(dest) == ah .- bh

    c = DPUArray(ah)
    add!(c, b)
    @test Array(c) == ah .+ bh
end

@testset "shapes must match" begin
    m = DPUArray(reshape(Int32.(1:N), DIMS2))
    t = DPUArray(reshape(Int32.(1:N), DIMS3))
    flipped = DPUArray(reshape(Int32.(1:N), 2 * DIMS2[1], :))
    row = DPUArray(ones(Int32, 1, DIMS2[2]))
    @test_throws DimensionMismatch m .+ t
    @test_throws DimensionMismatch m .+ flipped       # same length, other shape
    @test_throws DimensionMismatch m .+ row           # no singleton expansion
    @test_throws DimensionMismatch m + flipped
    @test_throws DimensionMismatch add!(m, flipped)
    @test_throws DimensionMismatch (similar(flipped) .= m .+ 1)
    @test_throws DimensionMismatch PolymerPIM.zeros(Int32, DIMS3) .+ m
end

@testset "reductions" begin
    host = reshape(Int32.(1:N) .% Int32(97), DIMS2)
    m = DPUArray(host)
    @test sum(m)[] == sum(Int64.(host))
    @test maximum(m)[] == maximum(host)
    @test minimum(m .+ 5)[] == minimum(host) + 5
    @test sum(abs, m .- 50)[] == sum(abs.(Int64.(host) .- 50))
    @test findmax(m) == (maximum(host), argmax(host))
    @test argmin(m .- 1) == argmin(host)
    @test argmax(m) isa CartesianIndex{2}
end

@testset "zeros and fill" begin
    z = PolymerPIM.zeros(Int32, 3, 4)
    @test size(z) == (3, 4)
    @test Array(DPUArray(z)) == zeros(Int32, 3, 4)
    @test Array(PolymerPIM.fill(Int32, (2, 3, 4), 7)) == fill(Int32(7), 2, 3, 4)
    m = DPUArray(reshape(Int32.(1:12), 3, 4))
    @test (PolymerPIM.zeros(Int32, 3, 4) .+ m) === m
end

@testset "display" begin
    m = DPUArray(Int32[1 2; 3 4])
    @test sprint(show, m) == "DPUMatrix{Int32}(2, 2)"
    out = sprint(show, MIME"text/plain"(), m)
    @test startswith(out, "2×2 DPUMatrix{Int32}:")
    @test occursin("1  2", out)
    t = DPUArray(reshape(Int32.(1:8), 2, 2, 2))
    @test startswith(sprint(show, MIME"text/plain"(), t), "2×2×2 DPUArray{Int32, 3}:")
end
