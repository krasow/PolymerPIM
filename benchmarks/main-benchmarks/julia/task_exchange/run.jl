# Task exchange: many independent tasks that trade data every round.
#
# Each task owns N/tasks elements. A round runs one local kernel per task, then
# has every task mix in a peer's result, the peer shifting each round, so all
# of every task's data crosses to another task once per round. Nothing says
# where tasks run: placement, and so how much of that traffic leaves a rank,
# is the mapper's.

using PolymerPIM
using Printf
include(joinpath(@__DIR__, "../bench.jl"))
using .Bench

const _gen = joinpath(@__DIR__, "Param.generated.jl")
include(isfile(_gen) ? _gen : joinpath(@__DIR__, "Param.jl"))

const LABEL = "julia"

const T = Param.T
const N = Param.N
const TASKS = Param.tasks
const ROUNDS = Param.rounds
const STEPS = Param.local_steps
const BULK = Param.bulk_sync != 0

# Round r pairs each task with the one `shift` after it, cycling through every
# other task.
function peer(t, r)
    shift = 1 + (r - 1) % max(TASKS - 1, 1)
    return mod(t - 1 + shift, TASKS) + 1
end

# The same arithmetic on DPU expressions and host arrays; Int32 throughout so
# the host wraps as the DPUs do.
function local_step(x, r)
    for j in 1:STEPS
        x = abs.((x .* T(3) .+ T(r + j)) .- (x .>> 2)) .* T(5) .+ x
    end
    return x
end
mix(a, b) = (a .+ (b .>> 1)) .- (a .>> 3)

function rounds!(s)
    for r in 1:ROUNDS
        u = [DPUArray(local_step(s[t], r)) for t in 1:TASKS]
        BULK && PolymerPIM.sync()   # control: no exchange until every task computed
        next = [DPUArray(mix(u[t], u[peer(t, r)])) for t in 1:TASKS]
        foreach(release!, u)
        foreach(release!, s)
        # Without it the queue fuses each round into the next, re-deriving
        # both inputs of every mix; it only dispatches, so overlap survives.
        fence(next[1])
        s = next
    end
    return s
end

function reference(states)
    s = copy.(states)
    for r in 1:ROUNDS
        u = [local_step(x, r) for x in s]
        s = [mix(u[t], u[peer(t, r)]) for t in 1:TASKS]
    end
    return s
end

function main()
    stages = BenchStages()
    warm_stages = BenchStages()
    n = N ÷ TASKS
    @printf("[VERIFY_TAG] Starting julia_task_exchange with N=%d tasks=%d rounds=%d\n",
            N, TASKS, ROUNDS)

    stage_begin!(stages, :init)
    PolymerPIM.sync()
    stage_end!(stages)

    stage_begin!(stages, :load)
    rng_state = UInt32(Param.seed)
    states = [Vector{T}(undef, n) for _ in 1:TASKS]
    for x in states, i in eachindex(x)
        rng_state = rng_state * UInt32(1664525) + UInt32(1013904223)
        x[i] = T(rng_state >> 24)
    end
    stage_end!(stages)

    result = Ref{Vector{Vector{T}}}()

    function round_trip(st)
        s = Bench.@stage st :write begin
            v = [DPUVector(x) for x in states]
            PolymerPIM.sync()
            v
        end
        s = Bench.@stage st :kernel begin
            v = rounds!(s)
            PolymerPIM.sync()
            v
        end
        result[] = Bench.@stage st :read [Array(v) for v in s]
        foreach(release!, s)
    end

    warm = BenchStats()
    for _ in 1:Param.warmup_iterations
        t0 = time_ns()
        round_trip(warm_stages)
        stats_update!(warm, elapsed_us(t0))
    end
    Param.warmup_iterations > 0 && stats_print("$(LABEL)_warmup", warm)

    stats = BenchStats()
    for _ in 1:Param.iterations
        t0 = time_ns()
        round_trip(stages)
        stats_update!(stats, elapsed_us(t0))
    end
    stats_print(LABEL, stats)
    stages_report(LABEL, stages)
    stages_report("$(LABEL)_cold", warm_stages)

    if Param.check_correctness != 0
        if reference(states) != result[]
            println("Mismatch: DPU task states differ from the host reference")
            exit(1)
        end
        println("the result is correct")
    end
end

main()
