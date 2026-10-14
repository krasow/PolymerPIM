# Vector search: for each query, return the best (score, row index) pair over
# the dataset. Mirrors benchmarks/main-benchmarks/polymerpim/vector_search/run.cc.
#
# `iterations` is the number of queries in the timed batch, not a repeat count.

using PolymerPIM
using Printf
include(joinpath(@__DIR__, "../bench.jl"))
using .Bench

const _gen = joinpath(@__DIR__, "Param.generated.jl")
include(isfile(_gen) ? _gen : joinpath(@__DIR__, "Param.jl"))

const LABEL = "julia"

# Bound as constants, not locals in main(): a local holding Param.T is only a
# DataType to inference, which makes every host array abstractly typed and
# turns each element store into a dynamic dispatch.
const T = Param.T
const N = Param.N
const DIM = Param.DIM
const seed = Param.seed

# Ports of vector_search_common.h.  The dataset is generated from a counter so
# it is reproducible without a stored reference copy.
function mix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

function dataset_value(seed::Integer, row::Integer, dim::Integer, dims::Integer)
    counter = UInt64(row) * UInt64(dims) + UInt64(dim)
    return (mix64((UInt64(seed) << 32) ⊻ counter) & 1) == 1 ? Int32(1) : Int32(-1)
end

function query_value(seed::Integer, query_id::Integer, dim::Integer)
    counter = 0xd1b54a32d192ed03 ⊻
              (UInt64(query_id) * 0x9e3779b97f4a7c15) ⊻ UInt64(dim)
    return (mix64((UInt64(seed) << 32) ⊻ counter) & 1) == 1 ? Int32(1) : Int32(-1)
end

function main()

    stages = BenchStages()
    warm_stages = BenchStages()

    @printf("[VERIFY_TAG] Starting julia_vector_search with N=%d DIM=%d\n", N, DIM)

    stage_begin!(stages, :init)
    PolymerPIM.sync()
    stage_end!(stages)

    # DIM dataset columns, all SoA.
    cols = Vector{DPUVector}()
    host_cols = Vector{Vector{T}}()
    for d in 0:(DIM - 1)
        stage_begin!(stages, :alloc)
        col = Vector{T}(undef, N)
        stage_end!(stages)
        stage_begin!(stages, :load)
        if Param.load_ref != 0
            load_bin!(joinpath(Param.ref_path, "SoA", "col_$(d).bin"), col)
        else
            @inbounds for i in 0:(N - 1)
                col[i + 1] = dataset_value(seed, i, d, DIM)
            end
        end
        stage_end!(stages)
        stage_begin!(stages, :write)
        push!(cols, DPUVector(col))
        fence(cols[end])
        stage_end!(stages)
        Param.check_correctness != 0 && push!(host_cols, col)
    end

    query_id = Ref{Int}(0)
    last_query = Vector{T}(undef, DIM)

    # score = sum_d (col_d * query_d); findmax returns the score and the first
    # global row index as two 32-bit fields.
    function run_queries(count, st)
        result = (typemin(Int32), typemax(UInt32))
        for _ in 1:count
            stage_begin!(st, :write)
            for d in 0:(DIM - 1)
                last_query[d + 1] = query_value(seed, query_id[], d)
            end
            query_id[] += 1
            stage_end!(st)

            stage_begin!(st, :kernel)
            e = PolymerPIM.zeros(T, N)
            for d in 1:DIM
                e = e .+ cols[d] .* last_query[d]
            end
            value, index = findmax(e)
            result = (Int32(value), UInt32(index - 1))
            stage_end!(st)
        end
        return result
    end

    warm = BenchStats()
    if Param.warmup_iterations > 0
        t0 = time_ns()
        run_queries(Param.warmup_iterations, warm_stages)
        stats_update!(warm, elapsed_us(t0) / Param.warmup_iterations)
        stats_print("$(LABEL)_warmup", warm)
    end

    stats = BenchStats()
    result = (typemin(Int32), typemax(UInt32))
    if Param.iterations > 0
        t0 = time_ns()
        result = run_queries(Param.iterations, stages)
        stats_update!(stats, elapsed_us(t0) / Param.iterations)
    end
    stats_print(LABEL, stats)
    stages_report(LABEL, stages)
    stages_report("$(LABEL)_cold", warm_stages)

    if Param.check_correctness != 0 && Param.iterations > 0
        best = (typemin(Int32), typemax(UInt32))
        for i in 0:(N - 1)
            score = 0
            for d in 1:DIM
                score += Int64(host_cols[d][i + 1]) * Int64(last_query[d])
            end
            candidate = (Int32(score), UInt32(i))
            if candidate[1] > best[1] ||
                    (candidate[1] == best[1] && candidate[2] < best[2])
                best = candidate
            end
        end
        if best != result
            @printf("Mismatch: CPU = (%d, %u), DPU = (%d, %u)\n",
                    best[1], best[2], result[1], result[2])
            exit(1)
        end
        println("the result is correct")
    end
end

main()
