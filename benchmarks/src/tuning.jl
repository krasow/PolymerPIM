const DEFAULT_FUSION_BUILD = Dict(
    "FUSION_LOOKAHEAD" => 128,
    "MAX_HFUSE_CHAINS" => 9,
    "JIT_BATCH_SIZE" => 16,
    "MAX_VFUSE_OPS" => 128,
    "MAX_VFUSE_INPUTS" => 11,
    "BLOCK_SIZE_LOG2" => 4,
)

const DEFAULT_FUSION_SEARCH = Dict(
    "FUSION_LOOKAHEAD" => [0, 1, 2, 4, 8, 16, 32, 64, 128],
    # 9 is the ceiling: a tenth chain overflows WRAM at stack depth 3.
    "MAX_HFUSE_CHAINS" => [1, 2, 4, 6, 8, 9],
    "JIT_BATCH_SIZE" => [0, 1, 2, 4, 8, 16, 32],
    "MAX_VFUSE_OPS" => [1, 8, 16, 32, 64, 96, 128, 192],
)

const CORE_FUSION_KNOBS = (
    "FUSION_LOOKAHEAD",
    "MAX_HFUSE_CHAINS",
    "JIT_BATCH_SIZE",
    "MAX_VFUSE_OPS",
)

Base.@kwdef mutable struct TuneOptions
    benchmarks::Vector{String} = String[]
    dpus::Vector{Int} = [256]
    elements_per_dpu::Union{Nothing,Vector{Int}} = nothing
    warmup::Union{Nothing,Int} = nothing
    iterations::Union{Nothing,Int} = nothing
    passes::Int = 2
    timeout::Int = DEFAULT_RUN_TIMEOUT
    build_timeout::Int = 120
    check::Bool = false
    verbose::Bool = false
    resume::Bool = true
    reset::Bool = false
    config::String = DEFAULT_CONFIG
    profiles::String = joinpath(BENCHMARK_DIR, "results", "fusion", "profiles")
    checkpoints::String = joinpath(BENCHMARK_DIR, "results", "fusion")
    search::Dict{String,Vector{Int}} = deepcopy(DEFAULT_FUSION_SEARCH)
    workspace_profiles::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[]
    invocation::Int = 0
end

struct TuneResult
    status::String
    objective::Float64
    times::Vector{Float64}
    cases::Vector{Tuple{Int,Int}}
end

mutable struct TuneCheckpoint
    path::String
    benchmark::String
    signature::Dict{String,Any}
    trials::Vector{Dict{String,Any}}
    cache::Dict{Tuple,TuneResult}
    recorded::Set{Tuple}
    complete::Bool
end

geomean(values) = isempty(values) || any(x -> x <= 0, values) ? Inf :
    exp(sum(log, values) / length(values))

config_key(build) = Tuple(build[knob] for knob in FUSION_BUILD_KNOBS)

function coordinate_descent(seed::Dict{String,Int}, search, passes::Int,
                            evaluate; observe = (_, _, _, _, _) -> nothing)
    best = copy(seed)
    best_result = evaluate(best)
    for pass in 1:passes
        changed = false
        for knob in CORE_FUSION_KNOBS
            local_build, local_result = best, best_result
            for candidate in unique([best[knob]; search[knob]])
                trial = copy(best)
                trial[knob] = candidate
                result = evaluate(trial)
                observe(pass, knob, candidate, trial, result)
                if result.objective < local_result.objective
                    local_build, local_result = trial, result
                end
            end
            if config_key(local_build) != config_key(best)
                best, best_result = local_build, local_result
                changed = true
            end
        end
        changed || break
    end
    return best, best_result
end

function tune_usage(io::IO = stdout)
    println(io, """
    Usage: julia tune.jl [options] [benchmark ...]

      --dpus N[,N...]             Representative DPU counts
      --elements-per-dpu N[,N...] Override benchmark.toml sizes
      --warmup N                  Override warmup iterations
      --iterations N              Override measured iterations
      --passes N                  Maximum coordinate-descent passes
      --lookahead N[,N...]        FUSION_LOOKAHEAD candidates
      --hfuse-chains N[,N...]     MAX_HFUSE_CHAINS candidates
      --jit-batch N[,N...]        JIT_BATCH_SIZE candidates
      --vfuse-ops N[,N...]        MAX_VFUSE_OPS candidates
      --workspace I:B[,I:B...]    MAX_VFUSE_INPUTS:BLOCK_SIZE_LOG2 profiles
      --check                     Verify each winner
      --verbose                   Print benchmark and build subprocess output
      --reset                     Discard saved tuning for selected benchmarks
      --profiles PATH             Output profile directory
      --checkpoints PATH          Tuning checkpoint directory
      --timeout N                 Run timeout in seconds (default: 1800)
      --build-timeout N           Build timeout in seconds
      --config PATH               Benchmark TOML path
      -h, --help                  Show this help
    """)
end

function nonnegative_list(value::AbstractString)
    values = parse.(Int, filter(!isempty, split(value, ',')))
    isempty(values) && error("expected a comma-separated integer list")
    all(>=(0), values) || error("list values must be non-negative")
    return values
end

function workspace_list(value::AbstractString)
    profiles = Tuple{Int,Int}[]
    for item in split(value, ',')
        fields = split(item, ':')
        length(fields) == 2 || error("workspace must use INPUTS:BLOCK_LOG2")
        profile = (parse(Int, fields[1]), parse(Int, fields[2]))
        all(>(0), profile) || error("workspace values must be positive")
        push!(profiles, profile)
    end
    return unique(profiles)
end

function parse_tune_args(args)
    options = TuneOptions()
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("-h", "--help")
            tune_usage()
            return nothing
        elseif arg in ("--check", "--resume", "--verbose")
            setproperty!(options, Symbol(arg[3:end]), true)
        elseif arg == "--reset"
            options.reset = true
            options.resume = false
        elseif arg in ("--dpus", "--elements-per-dpu", "--warmup", "--iterations",
                       "--passes", "--lookahead", "--hfuse-chains", "--jit-batch",
                       "--vfuse-ops", "--workspace", "--profiles", "--checkpoints",
                       "--timeout", "--build-timeout", "--config")
            value = option_value(args, index, arg)
            index += 1
            if arg == "--dpus"
                options.dpus = int_list(value)
            elseif arg == "--elements-per-dpu"
                options.elements_per_dpu = int_list(value)
            elseif arg in ("--warmup", "--iterations")
                parsed = parse(Int, value)
                parsed >= 0 || error("$arg must be non-negative")
                setproperty!(options, Symbol(arg[3:end]), parsed)
            elseif arg == "--passes"
                options.passes = parse(Int, value)
                options.passes > 0 || error("--passes must be positive")
            elseif arg == "--lookahead"
                options.search["FUSION_LOOKAHEAD"] = nonnegative_list(value)
            elseif arg == "--hfuse-chains"
                options.search["MAX_HFUSE_CHAINS"] = int_list(value)
            elseif arg == "--jit-batch"
                options.search["JIT_BATCH_SIZE"] = nonnegative_list(value)
            elseif arg == "--vfuse-ops"
                options.search["MAX_VFUSE_OPS"] = int_list(value)
            elseif arg == "--workspace"
                options.workspace_profiles = workspace_list(value)
            elseif arg in ("--profiles", "--checkpoints", "--config")
                setproperty!(options, Symbol(arg[3:end]), abspath(value))
            elseif arg == "--timeout"
                options.timeout = parse(Int, value)
            else
                options.build_timeout = parse(Int, value)
            end
        elseif startswith(arg, "-")
            error("unknown option $arg")
        else
            push!(options.benchmarks, arg)
        end
        index += 1
    end
    min(options.timeout, options.build_timeout) > 0 || error("timeouts must be positive")
    return options
end

function rebuild_for_tuning(config::RunnerConfig, benchmark::String, build,
                            options::TuneOptions; julia::Bool = false)
    config.setup === nothing && error("benchmark.toml has no setup commands")
    profile = FusionProfile(benchmark, build, Dict{String,Any}(), "")
    context = setup_context(config, profile)
    for raw in config.setup.commands
        command = render_template(raw, context)
        julia || (command *= " JULIA=__skip_julia_wrapper__")
        println("[fusion] build ", fusion_flags(profile))
        result = execute_command(config, command, config.paths.benchmarks, 1;
                                 timeout = options.build_timeout,
                                 echo = options.verbose)
        successful(result) || return result
    end
    return nothing
end

function run_variant_capture(config::RunnerConfig, variant::VariantSpec,
                             case::RunCase, options::TuneOptions)
    directory, context = prepare_variant(config, variant, case)
    return execute_variant(
        config, variant, case, directory, context;
        timeout = options.timeout, build_timeout = options.build_timeout,
        echo = options.verbose)
end

tuning_sizes(spec::BenchmarkSpec, options::TuneOptions) = something(
    options.elements_per_dpu, [minimum(spec.elements_per_dpu)])

function tuning_cases(spec::BenchmarkSpec, defaults::RunnerDefaults,
                      options::TuneOptions; check::Bool = false)
    sizes = tuning_sizes(spec, options)
    warmup = something(options.warmup, spec.warmup, defaults.warmup)
    iterations = something(options.iterations, spec.iterations, defaults.iterations)
    # Tuning never materializes the CPU reference set, so its cases synthesize
    # inputs in-process; every knob candidate is timed the same way.
    return [RunCase(spec.name, dpus, size, warmup, iterations, check, false,
                    something(spec.seed, defaults.seed), spec.parameters, spec.operation)
            for dpus in options.dpus for size in sizes]
end

function evaluate_build(config::RunnerConfig, spec::BenchmarkSpec, build,
                        options::TuneOptions, built::Base.RefValue;
                        check::Bool = false)
    key = config_key(build)
    if built[] != key
        failure = rebuild_for_tuning(config, spec.name, build, options)
        failure === nothing || return TuneResult(
            string(command_failure(:build, failure)), Inf,
            Float64[], Tuple{Int,Int}[])
        built[] = key
    end

    times, case_ids = Float64[], Tuple{Int,Int}[]
    for case in tuning_cases(spec, config.defaults, options; check)
        println("[fusion] run $(spec.name): $(case.dpus) DPUs × ",
                "$(case.elements_per_dpu) elements/DPU")
        if check && is_implemented(config, config.variants["cpu"], case)
            cpu = run_variant_capture(config, config.variants["cpu"], case, options)
            record_timing(joinpath(options.checkpoints, "runs.csv"), case, "cpu", cpu;
                          invocation = options.invocation)
            successful(cpu) || return TuneResult(
                "cpu_$(cpu.status)", Inf, times, case_ids)
        end
        result = run_variant_capture(config, config.variants["polymerpim"],
                                     case, options)
        record_timing(joinpath(options.checkpoints, "runs.csv"), case,
                      "polymerpim", result;
                      invocation = options.invocation, build)
        successful(result) || return TuneResult(
            string(result.status), Inf, times, case_ids)
        elapsed = result.timing["time"]
        wall = result.timing["real_s"]
        suffix = wall isa Number ? " (wall $(round(wall; digits = 2)) s)" : ""
        println("[fusion] result $(round(elapsed; digits = 3)) ms$suffix")
        push!(times, elapsed)
        push!(case_ids, (case.dpus, case.elements_per_dpu))
    end
    return TuneResult("ok", geomean(times), times, case_ids)
end

function tuning_source_fingerprint(config::RunnerConfig, spec::BenchmarkSpec)
    roots = [
        joinpath(config.paths.repo, "Makefile"),
        config.paths.config,
        joinpath(config.paths.repo, "common"),
        joinpath(config.paths.repo, "dpu"),
        joinpath(config.paths.repo, "host"),
        joinpath(config.paths.repo, "tools"),
        joinpath(config.paths.benchmarks, "src"),
        joinpath(config.paths.backends, "polymerpim.toml"),
        joinpath(config.paths.benchmarks, config.defaults.benchmark_root,
                 "polymerpim", spec.name),
    ]
    extensions = Set((".c", ".cc", ".cpp", ".h", ".inl", ".jl", ".py", ".toml"))
    files = String[]
    for root in roots
        if isfile(root)
            push!(files, root)
        elseif isdir(root)
            for (directory, subdirs, names) in walkdir(root)
                filter!(name -> name ∉ ("build", "bin", ".git"), subdirs)
                for name in names
                    occursin(".generated.", name) && continue
                    (name == "Makefile" || splitext(name)[2] in extensions) || continue
                    push!(files, joinpath(directory, name))
                end
            end
        end
    end
    buffer = IOBuffer()
    for path in sort!(unique(files))
        write(buffer, relpath(path, config.paths.repo), '\0', read(path), '\0')
    end
    return bytes2hex(sha256(take!(buffer)))
end

function checkpoint_signature(config::RunnerConfig, spec::BenchmarkSpec,
                              options::TuneOptions)
    return Dict{String,Any}(
        "dpus" => options.dpus,
        "elements_per_dpu" => tuning_sizes(spec, options),
        "warmup" => something(options.warmup, spec.warmup, config.defaults.warmup),
        "iterations" => something(options.iterations, spec.iterations,
                                   config.defaults.iterations),
        "passes" => options.passes,
        "check" => options.check,
        "search" => options.search,
        "workspace_profiles" => ["$a:$b" for (a, b) in options.workspace_profiles],
        "source_fingerprint" => tuning_source_fingerprint(config, spec),
    )
end

function trial_result(trial)
    cases = Tuple{Int,Int}[]
    for item in get(trial, "cases", String[])
        dpu, size = split(item, 'x'; limit = 2)
        push!(cases, (parse(Int, dpu), parse(Int, size)))
    end
    return TuneResult(string(trial["status"]),
                      Float64(get(trial, "objective_ms", Inf)),
                      Float64.(get(trial, "times_ms", Float64[])), cases)
end

function load_checkpoint(config::RunnerConfig, spec::BenchmarkSpec,
                         options::TuneOptions)
    path = joinpath(options.checkpoints, spec.name * ".toml")
    signature = checkpoint_signature(config, spec, options)
    trials, cache = Dict{String,Any}[], Dict{Tuple,TuneResult}()
    complete = false
    loaded = options.resume && isfile(path)
    if loaded
        raw = TOML.parsefile(path)
        get(raw, "version", 0) == 1 || error("unsupported tuning checkpoint $path")
        raw["benchmark"] == spec.name || error("wrong benchmark in $path")
        saved_trials = get(raw, "trials", Dict{String,Any}[])
        complete = Bool(get(raw, "complete", false))
        if raw["signature"] != signature
            profile = joinpath(options.profiles, spec.name * ".toml")
            reusable = complete && isfile(profile)
            if reusable
                println("[fusion] tuning inputs changed; reusing completed $(spec.name) profile")
            elseif complete || !isempty(saved_trials)
                error("tuning options do not match $path; reset tuning to retune")
            else
                loaded = false
                println("[fusion] refreshing empty checkpoint $(spec.name)")
            end
        end
        if loaded
            append!(trials, saved_trials)
            for trial in trials
                build = Dict(knob => Int(trial["build"][knob])
                             for knob in FUSION_BUILD_KNOBS)
                result = trial_result(trial)
                result.status == "ok" && (cache[config_key(build)] = result)
            end
        end
    end
    checkpoint = TuneCheckpoint(path, spec.name, signature, trials, cache,
                                Set(keys(cache)), complete)
    loaded || save_checkpoint(checkpoint)
    return checkpoint
end

function reset_tuning(names, options::TuneOptions)
    for name in names
        for path in (joinpath(options.profiles, name * ".toml"),
                     joinpath(options.checkpoints, name * ".toml"),
                     joinpath(options.checkpoints, name * ".csv"))
            rm(path; force = true)
        end
        println("[fusion] reset $name")
    end
end

function save_checkpoint_csv(checkpoint::TuneCheckpoint)
    path = splitext(checkpoint.path)[1] * ".csv"
    fields = ["benchmark", "phase", "knob", "candidate", "status", "objective_ms",
              collect(FUSION_BUILD_KNOBS)..., "cases"]
    temporary = path * ".tmp"
    open(temporary, "w") do io
        println(io, join(fields, ','))
        for trial in checkpoint.trials
            build = trial["build"]
            cases = join(("$case:$time" for (case, time) in
                          zip(trial["cases"], trial["times_ms"])), ';')
            values = Any[checkpoint.benchmark, trial["phase"], trial["knob"],
                         trial["candidate"], trial["status"],
                         get(trial, "objective_ms", ""),
                         (build[k] for k in FUSION_BUILD_KNOBS)..., cases]
            println(io, join(csv_cell.(values), ','))
        end
    end
    mv(temporary, path; force = true)
end

function save_checkpoint(checkpoint::TuneCheckpoint)
    mkpath(dirname(checkpoint.path))
    temporary = checkpoint.path * ".tmp"
    open(temporary, "w") do io
        write_toml(io, Dict("version" => 1,
                            "benchmark" => checkpoint.benchmark,
                            "complete" => checkpoint.complete,
                            "signature" => checkpoint.signature,
                            "trials" => checkpoint.trials))
    end
    mv(temporary, checkpoint.path; force = true)
    save_checkpoint_csv(checkpoint)
end

function record!(checkpoint::TuneCheckpoint, build, result::TuneResult,
                 phase, knob, candidate)
    key = config_key(build)
    key in checkpoint.recorded && return false
    checkpoint.complete = false
    trial = Dict{String,Any}(
        "phase" => string(phase), "knob" => string(knob),
        "candidate" => string(candidate), "status" => result.status,
        "build" => Dict(k => build[k] for k in FUSION_BUILD_KNOBS),
        "times_ms" => result.times,
        "cases" => ["$(d)x$(n)" for (d, n) in result.cases],
    )
    isfinite(result.objective) && (trial["objective_ms"] = result.objective)
    push!(checkpoint.trials, trial)
    checkpoint.cache[key] = result
    push!(checkpoint.recorded, key)
    save_checkpoint(checkpoint)
    return true
end

function write_profile(path, config, spec, build, initial, best, options, verified)
    case_times = Dict("$(case[1])x$(case[2])" => round(time; digits = 6)
                      for (case, time) in zip(best.cases, best.times))
    measurement = Dict{String,Any}(
        "status" => best.status, "verified" => verified,
        "objective_ms" => round(best.objective; digits = 6),
        "initial_objective_ms" => round(initial.objective; digits = 6),
        "ratio_vs_initial" => round(best.objective / initial.objective; digits = 6),
        "case_times_ms" => case_times,
        "dpus" => options.dpus,
        "elements_per_dpu" => tuning_sizes(spec, options),
        "warmup" => something(options.warmup, spec.warmup,
                              config.defaults.warmup),
        "iterations" => something(options.iterations, spec.iterations,
                                  config.defaults.iterations),
        "passes" => options.passes,
    )
    document = Dict("version" => 1, "benchmark" => spec.name,
                    "tuned_at" => Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"),
                    "measurement_variant" => "polymerpim",
                    "build" => Dict(k => build[k] for k in FUSION_BUILD_KNOBS),
                    "measurement" => measurement)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        write_toml(io, document)
    end
    mv(temporary, path; force = true)
end

function tune_target(config::RunnerConfig, spec::BenchmarkSpec, options::TuneOptions)
    checkpoint = load_checkpoint(config, spec, options)
    profile_path = joinpath(options.profiles, spec.name * ".toml")
    if options.resume && checkpoint.complete && isfile(profile_path)
        println("[fusion] $(spec.name) already complete -> $profile_path")
        return
    end
    isempty(checkpoint.cache) ||
        println("[fusion] resumed $(length(checkpoint.cache)) configurations for $(spec.name)")
    built = Ref{Any}(nothing)
    evaluate = function (build)
        key = config_key(build)
        haskey(checkpoint.cache, key) && return checkpoint.cache[key]
        result = evaluate_build(config, spec, build, options, built)
        checkpoint.cache[key] = result
        return result
    end

    seed = copy(DEFAULT_FUSION_BUILD)
    # The seed must be buildable: vector_search at dim=16 needs 16 inputs.
    if !isempty(options.workspace_profiles)
        seed["MAX_VFUSE_INPUTS"], seed["BLOCK_SIZE_LOG2"] =
            options.workspace_profiles[1]
    end
    for (knob, candidates) in options.search
        haskey(seed, knob) && !(seed[knob] in candidates) &&
            (seed[knob] = maximum(candidates))
    end
    initial = evaluate(seed)
    record!(checkpoint, seed, initial, "initial", "", "")
    isfinite(initial.objective) || error("initial configuration failed for $(spec.name)")
    println("[fusion] $(spec.name) initial $(round(initial.objective; digits = 3)) ms")
    observe = function (pass, knob, candidate, build, result)
        cached = !record!(checkpoint, build, result, pass, knob, candidate)
        value = isfinite(result.objective) ? round(result.objective; digits = 3) : result.status
        println("[fusion] pass $pass $knob=$candidate -> $value",
                cached ? " (cached)" : "")
    end
    best, result = coordinate_descent(seed, options.search, options.passes,
                                      evaluate; observe)

    for (inputs, block) in options.workspace_profiles
        trial = copy(best)
        trial["MAX_VFUSE_INPUTS"], trial["BLOCK_SIZE_LOG2"] = inputs, block
        candidate = evaluate(trial)
        observe(0, "workspace", "$inputs:$block", trial, candidate)
        candidate.objective < result.objective && ((best, result) = (trial, candidate))
    end

    apply_failure = rebuild_for_tuning(
        config, spec.name, best, options; julia = true)
    apply_failure === nothing || error(
        "could not apply winning configuration for $(spec.name): " *
        string(apply_failure.status))
    built[] = config_key(best)
    verified = false
    if options.check
        verification = evaluate_build(config, spec, best, options, built; check = true)
        verification.status == "ok" || error("winning configuration failed verification")
        verified = true
    end

    write_profile(profile_path, config, spec, best, initial, result, options,
                  verified)
    checkpoint.complete = true
    save_checkpoint(checkpoint)
    println("[fusion] best $(spec.name): $(round(result.objective; digits = 3)) ms -> $profile_path")
end

function tuning_manifest_entry(config::RunnerConfig, names,
                               options::TuneOptions, args)
    sweeps = Dict{String,Any}[]
    for name in names
        spec = only(config.benchmarks[name])
        cases = tuning_cases(spec, config.defaults, options;
                             check = options.check)
        sweep = manifest_dimensions(cases, ["polymerpim"])
        merge!(sweep, Dict{String,Any}(
            "seed_build" => DEFAULT_FUSION_BUILD,
            "search" => options.search,
            "workspace_profiles" => ["$a:$b" for (a, b) in
                                      options.workspace_profiles],
            "passes" => options.passes,
            "profile" => joinpath(options.profiles, name * ".toml"),
            "checkpoint" => joinpath(options.checkpoints, name * ".toml"),
        ))
        push!(sweeps, sweep)
    end
    return Dict{String,Any}(
        "arguments" => string.(args),
        "config" => config.paths.config,
        "profiles" => options.profiles,
        "checkpoints" => options.checkpoints,
        "csv" => joinpath(options.checkpoints, "runs.csv"),
        "check" => options.check,
        "resume" => options.resume,
        "reset" => options.reset,
        "verbose" => options.verbose,
        "timeout_seconds" => options.timeout,
        "build_timeout_seconds" => options.build_timeout,
        "sweeps" => sweeps,
    )
end

function run_tuner(options::TuneOptions, args = String[])
    config = load_config(options.config)
    names = isempty(options.benchmarks) ?
            filter(name -> "polymerpim" in config.benchmarks[name][1].variants,
                   config.benchmark_names) : unique(options.benchmarks)
    unknown = setdiff(names, config.benchmark_names)
    isempty(unknown) || error("unknown benchmark(s): $(join(unknown, ", "))")
    for name in names
        specs = config.benchmarks[name]
        length(specs) == 1 || error("tuning requires one [[$name]] table")
        "polymerpim" in specs[1].variants || error("$name has no PolymerPIM variant")
    end

    manifest = begin_manifest(
        joinpath(options.checkpoints, "Manifest.toml"), "fusion_tuning",
        tuning_manifest_entry(config, names, options, args))
    options.invocation = manifest.index
    try
        options.reset && reset_tuning(names, options)
        for name in names
            tune_target(config, only(config.benchmarks[name]), options)
        end
        finish_manifest(manifest, "complete")
    catch exception
        try
            finish_manifest(manifest, "failed"; failure = sprint(showerror, exception))
        catch manifest_error
            @error "could not update tuning manifest" path = manifest.path message =
                sprint(showerror, manifest_error)
        end
        rethrow()
    end
end

function tune_main(args = ARGS)
    try
        options = parse_tune_args(args)
        options === nothing || run_tuner(options, args)
        return 0
    catch exception
        exception isa InterruptException && rethrow()
        println(stderr, "error: ", sprint(showerror, exception))
        return 1
    end
end
