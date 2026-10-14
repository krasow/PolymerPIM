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
    "kmeans": "k-Means",
    "knn": "k-NN",
    "linreg": "Lin Reg",
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


# --- Figure style -----------------------------------------------------------
# Figures are included at the width they are authored at and never rescaled, so
# a point size here is the point size on the page.  Anything that has to match
# across figures lives here; geometry specific to one figure stays in its own
# script, under its own parameter section.

# Canvas, in inches.
TEXT_WIDTH_IN = 7.0        # \textwidth: the width of a full-width figure
PANEL_WIDTH_IN = 3.5       # one column of a multi-panel grid
TEXT_HEIGHT_IN = 9.0       # taller than this and LaTeX scales the figure down

# Type scale, in points.
FIGURE_TITLE_PT = 13
PANEL_TITLE_PT = 12.5
AXIS_LABEL_PT = 11.5
TICK_PT = 10.5
LEGEND_PT = 10.5
ANNOTATION_PT = 10

# Grid and rules.
GRID_COLOR = "#d8d8d8"
GRID_MINOR_COLOR = "#eeeeee"
GRID_LW = 0.8
GRID_MINOR_LW = 0.6
RULE_COLOR = "#444444"     # zero line, or whatever the data is read against

# Layout.  pad is what closes the gap under a suptitle; h_pad/w_pad do not.
LAYOUT_PAD = 0.3
SAVE_PAD_IN = 0.02
TICK_LENGTH = 3
TICK_PAD = 2


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
    # One line: two lines cost more height than the title and legend together.
    return (f"{BENCHMARK_LABELS[name]} "
            f"({format_elements(elements_per_dpu)}/DPU)")


HEADER_GAP_IN = 0.06       # breathing room under the title and legend


def artist_bottom_in(figure, artist):
    """Inches from the top of the figure to the bottom of `artist`."""
    box = artist.get_window_extent(figure.canvas.get_renderer())
    return figure.get_figheight() - box.y0 / figure.dpi


def header_rect_top(figure, gap_in=HEADER_GAP_IN):
    """The free fraction below the suptitle and any figure legend.

    Measured after a draw, not reserved as a fraction of the height: a title is
    a fixed number of inches tall whatever the figure, so a guessed fraction
    squeezes the shortest one first and the legend lands on the panel titles.
    """
    figure.canvas.draw()
    used = max((artist_bottom_in(figure, a)
                for a in (figure._suptitle, *figure.legends) if a is not None),
               default=0.0)
    return 1.0 - (used + gap_in) / figure.get_figheight()


def header_bottom_in(figure):
    """Inches from the top of the figure to the bottom of title and legend."""
    return max((artist_bottom_in(figure, a)
                for a in (figure._suptitle, *figure.legends) if a is not None),
               default=0.0)


def fit_below_header(figure, gap_in=HEADER_GAP_IN):
    """Raise the panels until their topmost ink clears the header by `gap_in`.

    tight_layout leaves slack above the panels that does not depend on the rect
    it was given, so the gap can only be closed once it has run.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    height = figure.get_figheight()
    tops = [height - axis.get_tightbbox(renderer).y1 / figure.dpi
            for axis in figure.axes if axis.get_visible()]
    if not tops:
        return
    slack = min(tops) - (header_bottom_in(figure) + gap_in)
    if slack > 0:
        figure.subplots_adjust(top=figure.subplotpars.top + slack / height)


def save_figure(figure, path, pad_inches=SAVE_PAD_IN):
    """Write a figure cropped to its ink."""
    figure.savefig(path, bbox_inches="tight", pad_inches=pad_inches)


def trim_spines(axis, sides=("top", "right")):
    for side in sides:
        axis.spines[side].set_visible(False)


def grid_shape(count, max_columns=2):
    columns = min(max_columns, count)
    return math.ceil(count / columns), columns
