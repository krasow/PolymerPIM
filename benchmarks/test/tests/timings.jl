const SAMPLE_TIMINGS = """
polymerpim_warmup (ms): mean=12.500 stddev=0.200 min=12.300 max=12.700 n=2
polymerpim (ms): mean=3.250 stddev=0.125 min=3.100 max=3.400 n=4
polymerpim_stage_alloc (ms): 1.500
polymerpim_stage_kernel (ms): 13.000
polymerpim_cold_stage_kernel (ms): 9.750
polymerpim_cold_stage_merge (ms): 2.250
"""
const SAMPLE_APP_TIME =
    "trace output\nAPP_TIME real=1.23 user=0.80 sys=0.10 maxrss=4567\n"

@testset "timing parsing and CSV" begin
    timing = BenchmarkRunner.parse_timings(
        "polymerpim", SAMPLE_TIMINGS, SAMPLE_APP_TIME)
    expected = Dict(
        "time" => 3.25, "stddev" => 0.125, "min" => 3.1, "max" => 3.4,
        "warmup_ms" => 12.5, "real_s" => 1.23, "max_rss_kb" => 4567.0,
        "alloc_ms" => 1.5, "kernel_cold_ms" => 9.75,
        "merge_cold_ms" => 2.25, "read_ms" => "",
    )
    @test all(timing[key] == value for (key, value) in expected)
    @test BenchmarkRunner.trial_summary(timing, 4) ==
          "iterations=   4  mean=     3.250 ms  stddev=     0.125 ms  " *
          "min=     3.100 ms  max=     3.400 ms  wall=    1.23 s"

    mktempdir() do directory
        case = test_case(warmup = 2, iterations = 4, operation = "a + b")
        command = BenchmarkRunner.CommandResult(
            :success, 0, SAMPLE_TIMINGS, SAMPLE_APP_TIME, 1.3, "")
        result = BenchmarkRunner.assess_run("polymerpim", case, command)
        path = joinpath(directory, "runs.csv")
        BenchmarkRunner.record_timing(
            path, case, "polymerpim", result; invocation = 3, trial = 3,
            build = BenchmarkRunner.DEFAULT_FUSION_BUILD)

        run = only(csv_records(path))
        @test run["variant"] == "polymerpim"
        @test run["status"] == "complete"
        @test run["trial"] == "3"
        @test run["parameters"] == ""
        @test run["time"] == "3.25"
        @test run["stddev"] == "0.125"
        @test run["min"] == "3.1"
        @test run["max"] == "3.4"
        @test [run[knob] for knob in BenchmarkRunner.FUSION_BUILD_KNOBS] ==
              ["128", "9", "16", "128", "11", "4"]

        sections = csv_records(BenchmarkRunner.sections_csv(path))
        @test length(sections) == 4
        @test all(row["trial"] == "3" for row in sections)
        @test only(filter(row -> row["section"] == "alloc", sections))["time_ms"] ==
              "1.5"
        @test only(filter(row -> row["section"] == "kernel" &&
                               row["kind"] == "cold", sections))["time_ms"] ==
              "9.75"
    end
end

@testset "run assessment" begin
    empty = BenchmarkRunner.CommandResult(:success, 0, "", "", 0.1, "")
    mismatch = BenchmarkRunner.CommandResult(
        :success, 0, "baseline (ms): 1.0\nMismatch at index 0\n", "", 0.1, "")
    @test BenchmarkRunner.assess_run(
        "baseline", test_case(), empty).status == :timing_missing
    @test BenchmarkRunner.assess_run(
        "baseline", test_case(check = true), mismatch).status ==
          :verification_failed
    timed_out = BenchmarkRunner.CommandResult(:timed_out, 124, "", "", 2.0, "")
    @test BenchmarkRunner.command_failure(:build, timed_out) == :build_timed_out
end
