# Benchmarks

Benchmark suites are TOML files. Workloads live in `main-benchmarks/` and
`dynamic-benchmarks/`; backend definitions live in `backends/`. Dependencies
are installed into `../opt/` on first use.

## Run

`run.sh` tunes fusion parameters, then runs the selected benchmarks:

```bash
./run.sh
./run.sh elementwise --resume
./run.sh elementwise --reset --default-params
./run.sh --config main-benchmarks/polymerpim-modes.toml --default-params
./run.sh --config dynamic-benchmarks/benchmark.toml --default-params
```

- `--resume` continues tuning and retries unfinished trials.
- `--reset` discards saved runs; `--reset-tune` discards the tuning profile.
- `--default-params` skips tuning and ignores profiles.
- `--no-load-ref` synthesizes inputs in-process instead of reading them.
- `--verbose` prints subprocess output.

`./run.sh --help` lists runner options, `--help-all` the advanced ones.
Arguments after `--tune` or `--runner` apply only to that phase. Benchmarks run
in the order named. `ntrials` launches independent processes; `iterations`
controls each process's workload loop.

## Inputs

Every variant reads the same reference files, so their load times are
comparable. The runner generates one case's files, runs all variants against
them, then deletes them — a single case reaches ~180 GB at the largest DPU
counts. `--check` additionally generates expected outputs and verifies results
against them.

`--no-load-ref` has each variant synthesize its own inputs instead: fast and
needs no disk, but the generators are not equally optimized, so load times are
no longer comparable across variants. The mode is recorded per row in the
`load_ref` column of `results/runs.csv`.

## Suites

- `main-benchmarks/benchmark.toml`: standard benchmark suite.
- `main-benchmarks/polymerpim-modes.toml`: JIT, Pipeline, and Eager comparison.
- `dynamic-benchmarks/benchmark.toml`: cold-start `adaptive_image` and
  `dynamic_query` workloads without warmup.

Dynamic workloads keep independent results under `results/dynamic/`:

```bash
./dynamic-benchmarks/adaptive_image.sh --reset
./dynamic-benchmarks/query_sweep.sh --reset
```

## Direct commands

```bash
julia --project=. tune.jl elementwise --dpus 256 --passes 2 --check
julia --project=. runner.jl elementwise --variant julia --check
julia --project=. test/runtests.jl
```

## Results

- `results/Manifest.toml`: resolved invocation and status.
- `results/runs.csv`: run status and total timings.
- `results/runs.sections.csv`: parsed timed sections.
- `results/fusion/`: tuning profiles, measurements, and checkpoints.

Runs use `/usr/bin/time`. Failures are recorded and do not stop the suite.
Generated results, parameters, binaries, and reference data are untracked.

```bash
./run_paper_results.sh                 # run every benchmark the paper needs
./plot_paper_figures.sh                # plot every paper figure
python3 plot_weak_scaling.py           # weak-scaling grids and averaged CSV
python3 plot_runtime_decomposition.py  # one runtime breakdown per DPU count
```

Decomposition bars average the completed trials: solid segments are setup and
measured-iteration work, hatched segments are warm-up work, and the unmeasured
segment is the residual between the instrumented stages and wall time. Error
bars show the wall-time standard deviation.

See `loc-analysis/` for the lines-of-code comparison across programming models.

The UPMEM SDK is loaded from `$UPMEM_ENV` or `/usr/upmem_env.sh`. Set `JULIA` to
select the Julia executable.
