# Construction, host transfers, indexing, display, and the plumbing
# Julia expects from a container type.

@testset "construction and conversion" begin
    data = Int32.(collect(1:N))
    v = DPUVector(data)
    @test length(v) == N
    @test size(v) == (N,)
    @test eltype(v) == Int32

    back = Array(v)
    @test back == data
    @test Vector(v) == data
    @test collect(v) == data

    data2 = Int32.(collect(1:2N))
    v2 = DPUVector(data2)
    @test length(v2) == 2N
    @test Array(v2) == data2
end

@testset "runtime shape" begin
    # Vectors are sharded across the DPUs, so the count has to be positive and
    # divide the sizes the rest of the suite uses.
    @test ndpus() > 0
    @test ntasklets() > 0
    @test N % ndpus() == 0
    requested = parse(Int, get(ENV, "NR_DPUS", "8"))
    if get(configuration(), "DEVICE_RUNTIME", "native") == "legion"
        @test ndpus() >= requested  # Legion allocates whole ranks
    else
        @test ndpus() == requested
    end
    @test ntasklets() == parse(Int, configuration()["NR_TASKLETS"])

    out = sprint(PolymerPIM.versioninfo)
    @test occursin("PolymerPIM v", out)
    @test occursin(configuration()["BACKEND"], out)
    @test occursin(string(ndpus()), out)
end

@testset "scalar indexing" begin
    data = Int32.(collect(10:10:10N))
    v = DPUVector(data)
    @test v[1] == Int32(10)
    @test v[N] == Int32(10N)
    @test_throws BoundsError v[0]
    @test_throws BoundsError v[N+1]
end

@testset "ragged lengths" begin
    # Any length is safe now.  These used to come back corrupt: the host
    # readback pushed align8(shard) into unpadded slots, so a length that
    # was not a multiple of 2*num_dpus silently lost lanes.
    for n in (1, 2, 7, 15, 17, 33, 100, 1000, 4099, 9973)
        data = Int32.(collect(1:n))
        v = DPUVector(data)
        @test length(v) == n
        back = Array(v)
        @test length(back) == n
        @test back == data

        w = DPUVector(fill(Int32(3), n))
        @test Array(v + w) == data .+ Int32(3)
        @test sum(v)[] == sum(Int64.(data))
    end
end

@testset "display" begin
    v = DPUVector(Int32.(collect(1:N)))
    buf = IOBuffer()
    show(buf, v)
    @test String(take!(buf)) == "DPUVector{Int32}($N)"

    show(buf, MIME("text/plain"), v)
    s = String(take!(buf))
    @test occursin("$N-element DPUVector{Int32}:", s)
    @test occursin("1", s)
end

@testset "similar and empty-runtime sync" begin
    a = DPUVector(Int32.(collect(1:N)))
    @test length(similar(a)) == N
    @test length(DPUVector(16)) == 16
    @test_throws ArgumentError DPUVector(-1)
    # sync() must be safe even with nothing submitted
    @test PolymerPIM.sync() === nothing
end

@testset "device-side fill" begin
    # Ragged lengths exercise the kernel's block loop and its 8-byte DMA
    # rounding; the values round-trip through a uint32 argument slot.
    for n in (1, 7, 16, 17, ndpus() - 1, ndpus() + 1, N, N + 3)
        for v in (Int32(0), Int32(-3), Int32(7), typemax(Int32), typemin(Int32))
            got = Array(PolymerPIM.fill(Int32, n, v))
            @test length(got) == n
            @test all(==(v), got)
        end
    end
    @test_throws ArgumentError PolymerPIM.fill(Int32, -1, Int32(0))

    # zeros() materialises through the same kernel rather than a host buffer.
    @test Array(DPUVector(PolymerPIM.zeros(Int32, N))) == zeros(Int32, N)

    @test @inferred(PolymerPIM.fill(Int32, 16, Int32(1))) isa DPUVector
    @test @inferred(DPUVector(PolymerPIM.zeros(Int32, 16))) isa DPUVector
end
