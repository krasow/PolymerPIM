#!/usr/bin/env python3
"""Tuned-vs-default fusion parameters, as gain over default per benchmark.

Whiskers span the per-size gains.  A bootstrap CI resampling both sides
independently is much wider, but that width is an artefact: the 2048-DPU
launch noise is common-mode and cancels in the ratio.
"""
import collections
import csv
import statistics

from matplotlib import ticker

from _plot_common import (
    AXIS_LABEL_PT,
    BENCHMARK_LABELS,
    EXCLUDED_BENCHMARKS,
    FIGURE_TITLE_PT,
    GRID_COLOR,
    GRID_LW,
    GRID_MINOR_COLOR,
    GRID_MINOR_LW,
    LAYOUT_PAD,
    LEGEND_PT,
    PANEL_TITLE_PT,
    RESULTS,
    RULE_COLOR,
    TEXT_WIDTH_IN,
    TICK_LENGTH,
    TICK_PAD,
    TICK_PT,
    save_figure,
    trim_spines,
)

MAIN_RESULTS = RESULTS / "main"
FIGURE_STEM = "param-gain"
VARIANT = "polymerpim"
DPU_COUNTS = (256, 2048)

# Figure parameters.
FIGURE_SIZE = (TEXT_WIDTH_IN, 2.7)
BAR_WIDTH = 0.38            # of a slot; the DPU counts sit side by side
DPU_COLORS = {256: "#3264a8", 2048: "#c1662f"}
WHISKER_COLOR = "#333333"
LABEL_MIN_PERCENT = 1.0     # smaller than this reads as zero; leave it bare
LABEL_OFFSET_PT = 3
GAIN_TICK_PERCENT = 4
BENCHMARK_LABEL_ROTATION = 18


def load(path):
    cells = collections.defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "complete" or not row["time"]:
                continue
            if (row["variant"] != VARIANT
                    or row["benchmark"] in EXCLUDED_BENCHMARKS):
                continue
            key = (row["benchmark"], int(row["dpus"]),
                   int(row["elements_per_dpu"]))
            cells[key].append(float(row["time"]))
    return cells


def gain_and_spread(default, tuned, keys):
    """Median gain in percent, plus the spread of the per-size gains."""
    per_size = [100 * (statistics.median(default[k]) - statistics.median(tuned[k]))
                / statistics.median(default[k]) for k in keys]
    middle = statistics.median(per_size)
    return middle, middle - min(per_size), max(per_size) - middle


def main():
    tuned = load(MAIN_RESULTS / "tuned.csv")
    default = load(MAIN_RESULTS / "default.csv")
    shared = sorted(set(tuned) & set(default))
    if not shared:
        raise SystemExit("no cells present in both runs")

    sizes = collections.defaultdict(list)
    for benchmark, dpus, elements in shared:
        sizes[(benchmark, dpus)].append((benchmark, dpus, elements))
    gains = {pair: gain_and_spread(default, tuned, keys)
             for pair, keys in sizes.items()}

    # Largest gain first; alphabetical order buries the one that moves.
    def rank(name):
        return max(gain for (benchmark, _), (gain, _, _) in gains.items()
                   if benchmark == name)

    benchmarks = sorted({benchmark for benchmark, _ in gains}, key=rank,
                        reverse=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for offset, dpus in enumerate(DPU_COUNTS):
        placed = [(index + (offset - 0.5) * BAR_WIDTH, gains[(benchmark, dpus)])
                  for index, benchmark in enumerate(benchmarks)
                  if (benchmark, dpus) in gains]
        if not placed:
            continue
        columns = [column for column, _ in placed]
        middles = [middle for _, (middle, _, _) in placed]
        lows = [low for _, (_, low, _) in placed]
        highs = [high for _, (_, _, high) in placed]
        axis.bar(columns, middles, width=BAR_WIDTH, yerr=[lows, highs],
                 color=DPU_COLORS[dpus], label=f"{dpus} DPUs",
                 error_kw=dict(elinewidth=1.2, capsize=3, capthick=1.2,
                               ecolor=WHISKER_COLOR))
        for column, middle, high in zip(columns, middles, highs):
            if abs(middle) < LABEL_MIN_PERCENT:
                continue
            axis.annotate(f"{middle:+.1f}%", (column, middle + high),
                          textcoords="offset points",
                          xytext=(0, LABEL_OFFSET_PT), ha="center",
                          va="bottom", fontsize=TICK_PT,
                          color=DPU_COLORS[dpus])

    axis.axhline(0, color=RULE_COLOR, linewidth=1.3)
    axis.set_xticks(range(len(benchmarks)))
    axis.set_xticklabels([BENCHMARK_LABELS.get(b, b) for b in benchmarks],
                         fontsize=PANEL_TITLE_PT,
                         rotation=BENCHMARK_LABEL_ROTATION, ha="right")
    axis.set_ylabel("Runtime reduction (%)", fontsize=AXIS_LABEL_PT)
    axis.yaxis.set_major_locator(ticker.MultipleLocator(GAIN_TICK_PERCENT))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(GAIN_TICK_PERCENT / 2))
    axis.tick_params(labelsize=TICK_PT, length=TICK_LENGTH, pad=TICK_PAD)
    axis.tick_params(axis="y", which="minor", length=1)
    axis.grid(True, axis="y", which="major", color=GRID_COLOR,
              linewidth=GRID_LW)
    axis.grid(True, axis="y", which="minor", color=GRID_MINOR_COLOR,
              linewidth=GRID_MINOR_LW)
    axis.set_axisbelow(True)
    trim_spines(axis)
    axis.legend(fontsize=LEGEND_PT, frameon=False, ncol=len(DPU_COUNTS),
                loc="upper right")
    axis.set_title("Tuned Fusion Parameters vs. Defaults",
                   fontsize=FIGURE_TITLE_PT, fontweight="bold", pad=4)
    figure.tight_layout(pad=LAYOUT_PAD)
    path = MAIN_RESULTS / f"{FIGURE_STEM}.pdf"
    save_figure(figure, path)
    plt.close(figure)

    print(f"Wrote {path}  ({len(benchmarks)} benchmarks, {len(shared)} cells)")
    for benchmark in benchmarks:
        parts = []
        for dpus in DPU_COUNTS:
            if (benchmark, dpus) in gains:
                middle, low, high = gains[(benchmark, dpus)]
                parts.append(f"{dpus}: {middle:+5.1f}% (-{low:.1f}/+{high:.1f})")
        print(f"  {benchmark:<16} " + "   ".join(parts))


if __name__ == "__main__":
    main()
