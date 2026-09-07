#!/usr/bin/env python3
"""Tuned-vs-default fusion parameters, as gain over default per cell."""
import collections
import csv
import random
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "main"
FIGURE_STEM = "param-gain"
VARIANT = "polymerpim"
VARIANT_COLOR = "#3264a8"
# One figure per DPU count.  Profiles are tuned at 256; at 2048 some tuned
# builds go bimodal run to run, which the whiskers make visible.
DPU_COUNTS = (256, 2048)


def _gain_ci_width(default, tuned, keys, draws=4000):
    """Width of the bootstrap 95% CI on the plotted bar, in points."""
    rng = random.Random(1)
    bars = []
    for _ in range(draws):
        per_size = []
        for k in keys:
            a = [rng.choice(default[k]) for _ in default[k]]
            b = [rng.choice(tuned[k]) for _ in tuned[k]]
            med_a = statistics.median(a)
            per_size.append(100 * (med_a - statistics.median(b)) / med_a)
        bars.append(statistics.median(per_size))
    bars.sort()
    return bars[int(0.975 * draws)] - bars[int(0.025 * draws)]


def load(path):
    cells = collections.defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "complete" or not row["time"]:
                continue
            key = (row["benchmark"], row["variant"], int(row["dpus"]),
                   int(row["elements_per_dpu"]))
            cells[key].append(float(row["time"]))
    return cells


def main():
    tuned, default = load(RESULTS / "tuned.csv"), load(RESULTS / "default.csv")
    shared = sorted(set(tuned) & set(default))
    if not shared:
        raise SystemExit("no cells present in both runs")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dpus in DPU_COUNTS:
        benchmarks = sorted({key[0] for key in shared if key[2] == dpus})
        if not benchmarks:
            print(f"no cells at {dpus} DPUs")
            continue

        figure, axis = plt.subplots(1, 1, figsize=(7.2, 4.4))
        xs, gains, lows, highs, flags = [], [], [], [], []
        for index, benchmark in enumerate(benchmarks):
            keys = [k for k in shared
                    if k[0] == benchmark and k[1] == VARIANT and k[2] == dpus]
            if not keys:
                continue
            # Median of 5 trials on both sides: one stalled run (a random I/O
            # or allocation hiccup) cannot move a median, but would dominate a
            # min/max over raw trials.  Whiskers span the per-size gains.
            per_size = [100 * (statistics.median(default[k])
                               - statistics.median(tuned[k]))
                        / statistics.median(default[k]) for k in keys]
            middle = statistics.median(per_size)
            # Flag uncertainty in the gain, not in the raw times: at 2048 DPUs a
            # cell can be noisy on both sides and still pin the gain down, since
            # common-mode noise cancels in the ratio.
            unstable = _gain_ci_width(default, tuned, keys) > 10.0
            xs.append(index)
            gains.append(middle)
            lows.append(middle - min(per_size))
            highs.append(max(per_size) - middle)
            flags.append(unstable)
        bars = axis.bar(xs, gains, width=0.55, yerr=[lows, highs], capsize=4,
                        color=VARIANT_COLOR)
        for bar, unstable in zip(bars, flags):
            if not unstable:
                continue
            bar.set_hatch("//")
            bar.set_edgecolor("white")
            axis.annotate("unstable", xy=(bar.get_x() + bar.get_width() / 2,
                                          0), xytext=(0, -14),
                          textcoords="offset points", ha="center",
                          fontsize=7, color="#a0342c")
        axis.axhline(0, color="#444444", linewidth=1)
        axis.set_xticks(range(len(benchmarks)))
        axis.set_xticklabels(benchmarks, rotation=30, ha="right")
        axis.grid(True, axis="y", color="#dddddd", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_ylabel("Gain from tuned parameters (%)")
        figure.suptitle(f"Tuned fusion parameters vs defaults ({dpus} DPUs)",
                        fontsize=14, fontweight="bold")
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        path = RESULTS / f"{FIGURE_STEM}-{dpus}.pdf"
        figure.savefig(path)
        plt.close(figure)
        cells = sum(1 for k in shared if k[1] == VARIANT and k[2] == dpus)
        print(f"Wrote {path}  ({cells} cells, {len(benchmarks)} benchmarks)")
    return


if __name__ == "__main__":
    main()
