"""Shared configuration and trial loading for benchmark plots."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

import os

BENCHMARKS = Path(__file__).resolve().parent
RESULTS = BENCHMARKS / "results"

# Which benchmark suite to plot:  PLOT_SUITE=modes
# Each suite has its own config, its own runs.csv, and its own output folder.
# The runner derives the same folder from the config name (see cli.jl), so a
# suite's figures sit beside the runs.csv they came from.
SUITES = {
    "main": ("benchmark.toml",
             ("polymerpim", "julia", "baseline", "simplepim",
              "simplepim-patched")),
    "modes": ("polymerpim-modes.toml",
              ("polymerpim-jit", "polymerpim-pipeline", "polymerpim-eager")),
}
SUITE = os.environ.get("PLOT_SUITE", "main").strip() or "main"
if SUITE not in SUITES:
    raise SystemExit(f"unknown PLOT_SUITE {SUITE!r}; "
                     f"expected one of {', '.join(sorted(SUITES))}")
_config_name, VARIANT_ORDER = SUITES[SUITE]
CONFIG = BENCHMARKS / "main-benchmarks" / _config_name
SUITE_RESULTS = (RESULTS if _config_name == "benchmark.toml"
                 else RESULTS / Path(_config_name).stem)
# PLOT_RUNS_CSV plots another runs CSV; its stem tags the figures so two
# inputs cannot overwrite each other.
_runs_override = os.environ.get("PLOT_RUNS_CSV", "").strip()
RUNS_CSV = Path(_runs_override) if _runs_override else SUITE_RESULTS / "runs.csv"
if _runs_override and not RUNS_CSV.is_file():
    raise SystemExit(f"PLOT_RUNS_CSV not found: {RUNS_CSV}")
RUNS_TAG = "" if RUNS_CSV.stem == "runs" else f"-{RUNS_CSV.stem}"

# Restrict the expected grid, or a partial sweep reads as unfinished and every
# benchmark is dropped.  PLOT_ONLY_DPUS=256,2048  PLOT_ONLY_VARIANTS=polymerpim
ONLY_DPUS = frozenset(int(part) for part
                      in os.environ.get("PLOT_ONLY_DPUS", "").split(",")
                      if part.strip())
ONLY_VARIANTS = frozenset(part.strip() for part
                          in os.environ.get("PLOT_ONLY_VARIANTS", "").split(",")
                          if part.strip())

BENCHMARK_ORDER = (
    "elementwise",
    "hist",
    "red",
    "kmeans",
    "knn",
    "linreg",
    "multitask_classifier",
    "vector_search",
)
@dataclass(frozen=True)
class FigureView:
    """The slice of the data one figure covers, and where it is written.

    Chosen by the environment, so a filtered figure never overwrites the
    canonical set:

        PLOT_ONLY_BENCHMARKS=elementwise   one panel, in its own folder
        PLOT_EXCLUDE_VARIANTS=simplepim    rescale axes one variant dominates
        PLOT_LOG_Y=1                       runtimes span two orders of magnitude
    """

    benchmarks: frozenset = frozenset()   # empty means every benchmark
    without: frozenset = frozenset()
    log_y: bool = False

    @staticmethod
    def from_env():
        return FigureView(_env_names("PLOT_ONLY_BENCHMARKS"),
                          _env_names("PLOT_EXCLUDE_VARIANTS"),
                          os.environ.get("PLOT_LOG_Y", "").strip()
                          not in ("", "0", "false", "no"))

    def covers(self, benchmark):
        return not self.benchmarks or benchmark in self.benchmarks

    def keeps(self, variant):
        return variant not in self.without

    def path(self, stem, extension):
        directory = (SUITE_RESULTS / "-".join(sorted(self.benchmarks))
                     if self.benchmarks else SUITE_RESULTS)
        directory.mkdir(parents=True, exist_ok=True)
        dropped = "-no-" + "-".join(sorted(self.without)) if self.without else ""
        scale = "-log" if self.log_y else ""
        return directory / f"{stem}{RUNS_TAG}{dropped}{scale}{extension}"


def _env_names(key):
    return frozenset(part.strip()
                     for part in os.environ.get(key, "").split(",") if part.strip())


VIEW = FigureView.from_env()

EXCLUDED_BENCHMARKS = {"multitask_classifier"}

BENCHMARK_LABELS = {
    "elementwise": "Elementwise",
    "hist": "Histogram",
    "kmeans": "K-Means",
    "knn": "KNN",
    "linreg": "Linear Regression",
    "multitask_classifier": "Multitask Classifier",
    "red": "Reduction",
    "vector_search": "Vector Search",
}

VARIANT_STYLES = {
    "polymerpim": ("PolymerPIM", "#3264a8", "o", "-"),
    "julia": ("Julia", "#dd7f27", "D", "-"),
    "baseline": ("Hand-tuned baseline", "#3b8f5a", "s", "-"),
    "simplepim": ("SimplePIM", "#b94a48", "^", "-"),
    "simplepim-patched": ("SimplePIM (direct gather)", "#8c564b", "v", "--"),
    "polymerpim-jit": ("PolymerPIM (JIT)", "#3264a8", "o", "-"),
    "polymerpim-pipeline": ("PolymerPIM (interpreter)", "#7b5ea7", "s", "-"),
    "polymerpim-eager": ("PolymerPIM (eager)", "#c26a2a", "^", "--"),
}


@dataclass(frozen=True)
class BenchmarkSelection:
    name: str
    elements_per_dpu: int
    dpus: tuple
    variants: tuple
    warmup: int
    iterations: int
    ntrials: int


def load_selections():
    with CONFIG.open("rb") as file:
        config = tomllib.load(file)

    defaults = config["runner"]
    selections = []
    for name in BENCHMARK_ORDER:
        if name in EXCLUDED_BENCHMARKS:
            continue
        if not VIEW.covers(name):
            continue
        specs = config.get(name, [])
        if not specs:
            continue
        target_size = min(
            int(size) for spec in specs for size in spec["elements_per_dpu"]
        )
        spec = next(
            spec for spec in specs if target_size in spec["elements_per_dpu"]
        )
        selections.append(BenchmarkSelection(
            name=name,
            elements_per_dpu=target_size,
            dpus=tuple(value for value
                       in (int(v) for v in spec.get("dpus", defaults["dpus"]))
                       if not ONLY_DPUS or value in ONLY_DPUS),
            variants=tuple(v for v in spec.get("variants", defaults["variants"])
                           if VIEW.keeps(v)
                           and (not ONLY_VARIANTS or v in ONLY_VARIANTS)),
            warmup=int(spec.get("warmup", defaults["warmup"])),
            iterations=int(spec.get("iterations", defaults["iterations"])),
            ntrials=int(defaults["ntrials"]),
        ))
    return selections


def load_successful_rows():
    rows = []
    malformed = 0
    with RUNS_CSV.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                malformed += 1
                continue
            if row.get("status") != "complete" or row.get("command_status") != "success":
                continue
            if row.get("check", "").lower() != "false":
                continue
            try:
                for field in ("dpus", "elements_per_dpu", "warmup", "iterations", "trial"):
                    row[field] = int(row[field])
                row["time"] = float(row["time"])
                row["real_s"] = float(row["real_s"])
            except (KeyError, TypeError, ValueError):
                malformed += 1
                continue
            rows.append(row)
    if malformed:
        print(f"Ignored {malformed} malformed/incomplete CSV row(s)")
    return rows


def select_latest_trials(rows, selections):
    """Return the newest configured row for each benchmark/variant/DPU/trial."""
    selected = {}
    by_name = {selection.name: selection for selection in selections}
    for row in rows:
        selection = by_name.get(row["benchmark"])
        if selection is None:
            continue
        if (row["elements_per_dpu"] != selection.elements_per_dpu
                or row["dpus"] not in selection.dpus
                or row["variant"] not in selection.variants
                or row["warmup"] != selection.warmup
                or row["iterations"] != selection.iterations
                or not 1 <= row["trial"] <= selection.ntrials):
            continue
        key = (row["benchmark"], row["variant"], row["dpus"], row["trial"])
        previous = selected.get(key)
        if previous is None or row["timestamp"] > previous["timestamp"]:
            selected[key] = row
    return selected


def load_trials():
    if not RUNS_CSV.is_file():
        raise SystemExit(f"missing benchmark results: {RUNS_CSV}")
    selections = load_selections()
    return selections, select_latest_trials(load_successful_rows(), selections)


def complete_trial_rows(latest, selection, variant, dpus):
    rows = [
        latest.get((selection.name, variant, dpus, trial))
        for trial in range(1, selection.ntrials + 1)
    ]
    return [] if any(row is None for row in rows) else rows


def average(values):
    return sum(values) / len(values)


def sample_stddev(values):
    if len(values) < 2:
        return 0.0
    mean = average(values)
    return math.sqrt(
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def format_elements(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}K"
    return str(value)


def benchmark_title(name, elements_per_dpu):
    return (f"{BENCHMARK_LABELS[name]}\n"
            f"{format_elements(elements_per_dpu)} elements/DPU")


def grid_shape(count, max_columns=2):
    columns = min(max_columns, count)
    return math.ceil(count / columns), columns
