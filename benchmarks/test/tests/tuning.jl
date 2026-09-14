@testset "fusion search" begin
    # 9 is the ceiling: a tenth chain overflows WRAM at stack depth 3.
    @test BenchmarkRunner.DEFAULT_FUSION_SEARCH["MAX_HFUSE_CHAINS"] ==
          [1, 2, 4, 6, 8, 9]
    seed = copy(BenchmarkRunner.DEFAULT_FUSION_BUILD)
    search = Dict(knob => [value] for (knob, value) in seed)
    search["FUSION_LOOKAHEAD"] = [1, 2, 3]
    evaluations = Ref(0)
    evaluate = build -> begin
        evaluations[] += 1
        objective = Float64((build["FUSION_LOOKAHEAD"] - 2)^2 + 1)
        BenchmarkRunner.TuneResult("ok", objective, [objective], [(1, 1)])
    end

    best, result = BenchmarkRunner.coordinate_descent(seed, search, 3, evaluate)

    @test best["FUSION_LOOKAHEAD"] == 2
    @test result.objective == 1.0
    @test evaluations[] < 3 * sum(length, values(search))
end

@testset "fusion profiles" begin
    mktempdir() do directory
        profile = Dict(
            "version" => 1,
            "benchmark" => "elementwise",
            "build" => Dict(knob => index for
                            (index, knob) in enumerate(
                                BenchmarkRunner.FUSION_BUILD_KNOBS)),
        )
        open(joinpath(directory, "elementwise.toml"), "w") do io
            TOML.print(io, profile)
        end
        loaded = BenchmarkRunner.load_fusion_profile(directory, "elementwise")
        @test loaded.build["FUSION_LOOKAHEAD"] == 1

        text = captured_output() do
            run_cli([
                "elementwise", "--variant", "polymerpim", "--dpus", "2",
                "--elements-per-dpu", "64", "--warmup", "0",
                "--iterations", "1", "--profiles", directory, "--dry-run",
            ])
        end
        @test occursin("Using fusion profile", text)
        @test occursin("FUSION_LOOKAHEAD=1", text)
    end
end

@testset "tuning checkpoints" begin
    config = load_config()
    spec = only(config.benchmarks["elementwise"])
    reset = BenchmarkRunner.parse_tune_args(["elementwise", "--reset"])
    @test reset.reset && !reset.resume
    @test BenchmarkRunner.tuning_sizes(spec, BenchmarkRunner.TuneOptions()) ==
          [minimum(spec.elements_per_dpu)]
    @test BenchmarkRunner.tuning_sizes(
        spec, BenchmarkRunner.TuneOptions(elements_per_dpu = [64, 128])) ==
          [64, 128]

    mktempdir() do directory
        options = BenchmarkRunner.TuneOptions(
            dpus = [2], elements_per_dpu = [64], warmup = 0, iterations = 1,
            checkpoints = directory)
        build = copy(BenchmarkRunner.DEFAULT_FUSION_BUILD)
        result = BenchmarkRunner.TuneResult("ok", 4.5, [4.5], [(2, 64)])
        checkpoint = BenchmarkRunner.load_checkpoint(config, spec, options)
        @test BenchmarkRunner.record!(
            checkpoint, build, result, 1, "FUSION_LOOKAHEAD", 128)

        resumed = BenchmarkRunner.load_checkpoint(
            config, spec, BenchmarkRunner.TuneOptions(
                dpus = [2], elements_per_dpu = [64], warmup = 0,
                iterations = 1, checkpoints = directory, resume = true))
        @test resumed.cache[BenchmarkRunner.config_key(build)].objective == 4.5
        @test isfile(joinpath(directory, "elementwise.csv"))
        raw = TOML.parsefile(checkpoint.path)
        @test raw["signature"]["search"] == options.search

        mismatched = BenchmarkRunner.TuneOptions(
            dpus = [4], elements_per_dpu = [64], warmup = 0, iterations = 1,
            checkpoints = directory, resume = true)
        @test_throws ErrorException BenchmarkRunner.load_checkpoint(
            config, spec, mismatched)

        checkpoint.complete = true
        BenchmarkRunner.save_checkpoint(checkpoint)
        profiles = joinpath(directory, "profiles")
        mkpath(profiles)
        write(joinpath(profiles, "elementwise.toml"), "saved")
        completed = BenchmarkRunner.load_checkpoint(
            config, spec, BenchmarkRunner.TuneOptions(
                dpus = [4], elements_per_dpu = [64], warmup = 0,
                iterations = 1, checkpoints = directory,
                profiles = profiles, resume = true))
        @test completed.complete
    end

    mktempdir() do directory
        options = BenchmarkRunner.TuneOptions(
            dpus = [2], elements_per_dpu = [64], warmup = 0, iterations = 1,
            checkpoints = directory)
        checkpoint = BenchmarkRunner.load_checkpoint(config, spec, options)
        failed = BenchmarkRunner.TuneResult(
            "runtime_failed", Inf, Float64[], Tuple{Int,Int}[])
        BenchmarkRunner.record!(checkpoint, copy(BenchmarkRunner.DEFAULT_FUSION_BUILD),
                                failed, 1, "FUSION_LOOKAHEAD", 128)
        resumed = BenchmarkRunner.load_checkpoint(config, spec, options)
        @test isempty(resumed.cache)
        @test length(resumed.trials) == 1
    end

    mktempdir() do directory
        path = joinpath(directory, "elementwise.toml")
        open(path, "w") do io
            TOML.print(io, Dict(
                "version" => 1, "benchmark" => "elementwise",
                "complete" => false, "trials" => [],
                "signature" => Dict("stale" => true)))
        end
        options = BenchmarkRunner.TuneOptions(
            dpus = [2], elements_per_dpu = [64], warmup = 0, iterations = 1,
            checkpoints = directory)
        refreshed = quietly() do
            BenchmarkRunner.load_checkpoint(config, spec, options)
        end
        @test isempty(refreshed.trials)
        @test refreshed.signature ==
              BenchmarkRunner.checkpoint_signature(config, spec, options)
    end
end

@testset "tuning reset" begin
    mktempdir() do directory
        profiles = joinpath(directory, "profiles")
        checkpoints = joinpath(directory, "checkpoints")
        mkpath(profiles)
        mkpath(checkpoints)
        targets = [joinpath(profiles, "elementwise.toml"),
                   joinpath(checkpoints, "elementwise.toml"),
                   joinpath(checkpoints, "elementwise.csv")]
        foreach(path -> write(path, "saved"), targets)
        runs = joinpath(checkpoints, "runs.csv")
        write(runs, "saved")

        quietly() do
            BenchmarkRunner.reset_tuning(
                ["elementwise"], BenchmarkRunner.TuneOptions(
                    profiles = profiles, checkpoints = checkpoints,
                    reset = true))
        end

        @test all(!isfile, targets)
        @test isfile(runs)
    end
end
