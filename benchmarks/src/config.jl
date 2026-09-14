const DEFAULT_RUN_TIMEOUT = 1800

struct Paths
    benchmarks::String
    repo::String
    backends::String
    environment::String
    config::String
end

Paths(config::AbstractString = DEFAULT_CONFIG) = Paths(
    BENCHMARK_DIR, REPO_ROOT, BACKEND_DIR, DEFAULT_ENV, abspath(config))

Base.@kwdef mutable struct Options
    benchmarks::Vector{String} = String[]
    variants::Vector{String} = String[]
    dpus::Union{Nothing,Vector{Int}} = nothing
    elements_per_dpu::Union{Nothing,Vector{Int}} = nothing
    warmup::Union{Nothing,Int} = nothing
    iterations::Union{Nothing,Int} = nothing
    ntrials::Union{Nothing,Int} = nothing
    timeout::Int = DEFAULT_RUN_TIMEOUT
    build_timeout::Int = 120
    check::Bool = false
    load_ref::Bool = true
    keep_ref_data::Bool = false
    skip_setup::Bool = false
    generate_only::Bool = false
    dry_run::Bool = false
    verbose::Bool = false
    action::Symbol = :run
    config::String = DEFAULT_CONFIG
    profiles::String = joinpath(BENCHMARK_DIR, "results", "fusion", "profiles")
    use_profiles::Bool = true
    resume::Bool = false
    reset::Bool = false
    state::String = joinpath(BENCHMARK_DIR, "results", "runner-state.toml")
    csv::Union{Nothing,String} = nothing
end

struct VariantSpec
    name::String
    directory::String
    parameter_file::Union{Nothing,String}
    prepare::Vector{String}
    build::Vector{String}
    run::String
    setup::Vector{String}
    timing_label::String
    use_profile::Bool
end

VariantSpec(name, directory, parameter_file, prepare, build, run) =
    VariantSpec(name, directory, parameter_file, prepare, build, run,
                String[], name, name in ("polymerpim", "julia"))

struct RunnerDefaults
    dpus::Vector{Int}
    warmup::Int
    iterations::Int
    ntrials::Int
    seed::Int
    variants::Vector{String}
    group_by_variant::Bool
    benchmark_root::String
end

struct SetupSpec
    variants::Vector{String}
    commands::Vector{String}
end

struct BenchmarkSpec
    name::String
    elements_per_dpu::Vector{Int}
    dpus::Union{Nothing,Vector{Int}}
    warmup::Union{Nothing,Int}
    iterations::Union{Nothing,Int}
    seed::Union{Nothing,Int}
    variants::Vector{String}
    parameters::Dict{String,Any}
    operation::Union{Nothing,String}
end

struct RunnerConfig
    paths::Paths
    defaults::RunnerDefaults
    setup::Union{Nothing,SetupSpec}
    variants::Dict{String,VariantSpec}
    benchmark_names::Vector{String}
    benchmarks::Dict{String,Vector{BenchmarkSpec}}
end

struct RunCase
    benchmark::String
    dpus::Int
    elements_per_dpu::Int
    warmup::Int
    iterations::Int
    check::Bool
    load_ref::Bool
    seed::Int
    parameters::Dict{String,Any}
    operation::Union{Nothing,String}
end

struct FusionProfile
    benchmark::String
    build::Dict{String,Int}
    metadata::Dict{String,Any}
    path::String
end

const FUSION_BUILD_KNOBS = (
    "QUEUE_ABSORB_LIMIT",
    "MAX_HFUSE_CHAINS",
    "JIT_BATCH_SIZE",
    "MAX_VFUSE_OPS",
    "MAX_VFUSE_INPUTS",
    "BLOCK_SIZE_LOG2",
)

function load_fusion_profile(directory::AbstractString, benchmark::AbstractString)
    path = joinpath(directory, benchmark * ".toml")
    isfile(path) || return nothing
    raw = TOML.parsefile(path)
    recorded = string(required(raw, "benchmark", path))
    recorded == benchmark || error("$path records benchmark $recorded")
    table = required(raw, "build", path)
    unknown = setdiff(collect(keys(table)), collect(FUSION_BUILD_KNOBS))
    isempty(unknown) || error("$path has unknown build knob(s): $(join(unknown, ", "))")
    build = Dict{String,Int}()
    for knob in FUSION_BUILD_KNOBS
        haskey(table, knob) || error("$path is missing build.$knob")
        value = Int(table[knob])
        value >= 0 || error("$path has a negative build.$knob")
        build[knob] = value
    end
    metadata = Dict{String,Any}(string(k) => v for (k, v) in raw if k != "build")
    return FusionProfile(recorded, build, metadata, path)
end

fusion_flags(profile::Nothing) = ""
fusion_flags(profile::FusionProfile) = join(
    ("$knob=$(profile.build[knob])" for knob in FUSION_BUILD_KNOBS), " ")

total_elements(case::RunCase) = case.dpus * case.elements_per_dpu

function string_list(table, key::AbstractString; default = String[])
    value = get(table, key, default)
    value isa Vector || error("$key must be a list")
    return string.(value)
end

function integer_list(table, key::AbstractString; default = nothing)
    value = get(table, key, default)
    value === nothing && return nothing
    value isa Vector || error("$key must be a list")
    result = Int.(value)
    isempty(result) && error("$key must not be empty")
    all(>(0), result) || error("$key values must be positive")
    return result
end

function nonnegative(table, key::AbstractString; default = nothing)
    value = get(table, key, default)
    value === nothing && return nothing
    result = Int(value)
    result >= 0 || error("$key must be non-negative")
    return result
end

function required(table, key::AbstractString, context::AbstractString)
    haskey(table, key) || error("$context has no $key")
    return table[key]
end

function load_variants(paths::Paths)
    isdir(paths.backends) || error("missing backends directory $(paths.backends)")
    definitions = Dict{String,VariantSpec}()
    files = sort(filter(path -> isfile(path) && endswith(path, ".toml"),
                        readdir(paths.backends; join = true)))
    isempty(files) && error("no backend TOML files found in $(paths.backends)")
    for file in files
        table = TOML.parsefile(file)
        name = string(required(table, "name", file))
        haskey(definitions, name) && error("duplicate variant $name")
        directory = string(required(table, "directory", file))
        run = string(required(table, "run", file))
        parameter_file = get(table, "parameter_file", nothing)
        definitions[name] = VariantSpec(
            name, directory,
            parameter_file === nothing ? nothing : string(parameter_file),
            string_list(table, "prepare"), string_list(table, "build"), run,
            string_list(table, "setup"),
            string(get(table, "timing_label", name)),
            Bool(get(table, "use_profile",
                     name in ("polymerpim", "julia"))))
    end
    return definitions
end

function load_config(path::AbstractString = DEFAULT_CONFIG)
    paths = Paths(path)
    raw = TOML.parsefile(paths.config)
    variants = load_variants(paths)

    runner = get(raw, "runner", Dict{String,Any}())
    default_dpus = integer_list(runner, "dpus")
    default_dpus === nothing && error("runner.dpus is required")
    default_warmup = nonnegative(runner, "warmup")
    default_warmup === nothing && error("runner.warmup is required")
    default_iterations = nonnegative(runner, "iterations")
    default_iterations === nothing && error("runner.iterations is required")
    default_ntrials = Int(get(runner, "ntrials", 1))
    default_ntrials > 0 || error("runner.ntrials must be positive")
    defaults = RunnerDefaults(
        default_dpus, default_warmup, default_iterations, default_ntrials,
        Int(get(runner, "seed", 1)),
        string_list(runner, "variants"; default = sort(collect(keys(variants)))),
        Bool(get(runner, "group_by_variant", false)),
        string(get(runner, "benchmark_root", "main-benchmarks")))

    benchmark_root = abspath(paths.benchmarks, defaults.benchmark_root)
    startswith(benchmark_root, paths.benchmarks * "/") ||
        error("runner.benchmark_root leaves the benchmark tree")
    isdir(benchmark_root) || error("missing benchmark root $benchmark_root")

    unknown_defaults = setdiff(defaults.variants, collect(keys(variants)))
    isempty(unknown_defaults) || error(
        "unknown default variant(s): $(join(unknown_defaults, ", "))")

    setup = if haskey(raw, "setup")
        table = raw["setup"]
        SetupSpec(string_list(table, "variants"), string_list(table, "commands"))
    else
        nothing
    end

    reserved = Set(("runner", "setup"))
    names = sort([name for (name, value) in raw
                  if !(name in reserved) && value isa Vector])
    isempty(names) && error("no benchmark tables found in $(paths.config)")
    benchmarks = Dict{String,Vector{BenchmarkSpec}}()
    for name in names
        specs = BenchmarkSpec[]
        for table in raw[name]
            table isa AbstractDict || error("[[$name]] must be a table")
            allowed = string_list(table, "variants"; default = defaults.variants)
            unknown = setdiff(allowed, collect(keys(variants)))
            isempty(unknown) || error(
                "$name has unknown variant(s): $(join(unknown, ", "))")
            parameters = Dict{String,Any}(
                string(key) => value for (key, value) in
                get(table, "parameters", Dict{String,Any}()))
            operation = get(table, "operation", nothing)
            sizes = integer_list(table, "elements_per_dpu")
            sizes === nothing && error("$name.elements_per_dpu is required")
            push!(specs, BenchmarkSpec(
                name, sizes,
                integer_list(table, "dpus"),
                nonnegative(table, "warmup"),
                nonnegative(table, "iterations"),
                haskey(table, "seed") ? Int(table["seed"]) : nothing,
                allowed, parameters,
                operation === nothing ? nothing : string(operation)))
        end
        benchmarks[name] = specs
    end
    return RunnerConfig(paths, defaults, setup, variants, names, benchmarks)
end
