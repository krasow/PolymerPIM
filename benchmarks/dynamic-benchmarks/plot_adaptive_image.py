#!/usr/bin/env python3

import csv
import math
from collections import defaultdict
from statistics import median
from dataclasses import dataclass
from pathlib import Path

from plot_common import (configure_axis, draw_series, legend_handles,
                         load_pyplot, write_summary)


BENCHMARKS = Path(__file__).resolve().parent.parent
RESULTS = BENCHMARKS / "results" / "dynamic"
RUNS_CSV = RESULTS / "adaptive-image.csv"
SUMMARY_CSV = RESULTS / "adaptive-image-summary.csv"
FIGURE = RESULTS / "adaptive-image.pdf"
# The regime where compile latency dominates the tail; the larger size inverts
# the ordering because compile cost is fixed while execution scales.
PLOT_SIZE = 262144

MODEL_ORDER = (
    "polymerpim-jit",
    "polymerpim-hybrid",
    "polymerpim-pipeline",
    "polymerpim-eager",
)
@dataclass(frozen=True)
class Summary:
    model: str
    elements_per_dpu: int
    dpus: int
    total_elements: int
    iterations: int
    ntrials: int
    mean_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    max_mean_ms: float
    max_min_ms: float
    max_max_ms: float
    wall_mean_s: float
    wall_stddev_s: float
    wall_min_s: float
    wall_max_s: float


def sample_stddev(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values)
                     / (len(values) - 1))


def pooled_stats(rows):
    count = sum(int(row["iterations"]) for row in rows)
    mean = sum(float(row["time"]) * int(row["iterations"]) for row in rows) / count
    variance = sum(
        (int(row["iterations"]) - 1) * float(row["stddev"]) ** 2
        + int(row["iterations"]) * (float(row["time"]) - mean) ** 2
        for row in rows
    ) / (count - 1)
    return mean, math.sqrt(variance)


def row_identity(row):
    return tuple(row[name] for name in (
        "variant", "dpus", "elements_per_dpu", "warmup", "iterations",
        "trial", "check", "seed", "parameters",
    ))


def load_rows():
    completed = {}
    with RUNS_CSV.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["benchmark"] == "adaptive_image" and row["status"] == "complete":
                completed[row_identity(row)] = row

    rows = list(completed.values())
    signatures = {(row["iterations"], row["check"], row["parameters"])
                  for row in rows}
    if len(signatures) > 1:
        raise SystemExit("results contain mixed adaptive_image configurations; "
                         "rerun with --reset")
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], int(row["elements_per_dpu"]),
                 int(row["dpus"]))].append(row)

    summaries = []
    for (model, elements_per_dpu, dpus), trials in sorted(grouped.items()):
        mean_ms, stddev_ms = pooled_stats(trials)
        trial_maxima = [float(row["max"]) for row in trials]
        wall = [float(row["real_s"]) for row in trials if row["real_s"]]
        summaries.append(Summary(
            model=model,
            elements_per_dpu=elements_per_dpu,
            dpus=dpus,
            total_elements=elements_per_dpu * dpus,
            iterations=int(trials[0]["iterations"]),
            ntrials=len(trials),
            mean_ms=mean_ms,
            stddev_ms=stddev_ms,
            min_ms=min(float(row["min"]) for row in trials),
            max_ms=max(float(row["max"]) for row in trials),
            max_mean_ms=sum(trial_maxima) / len(trial_maxima),
            max_min_ms=min(trial_maxima),
            max_max_ms=max(trial_maxima),
            wall_mean_s=sum(wall) / len(wall),
            wall_stddev_s=sample_stddev(wall),
            wall_min_s=min(wall),
            wall_max_s=max(wall),
        ))
    return summaries


def draw(axis, rows, model, value, low=None, high=None):
    points = sorted((row for row in rows if row.model == model),
                    key=lambda row: row.dpus)
    if not points:
        return
    values = [getattr(row, value) for row in points]
    yerr = None
    if low and high:
        yerr = (
            [center - getattr(row, low) for center, row in zip(values, points)],
            [getattr(row, high) - center for center, row in zip(values, points)],
        )
    draw_series(axis, model, list(zip((row.dpus for row in points), values)),
                yerr=yerr, linewidth=2.1)


def plot(rows):
    plt = load_pyplot(FIGURE)

    selected = [row for row in rows if row.elements_per_dpu == PLOT_SIZE]
    if not selected:
        raise SystemExit(f"no adaptive_image rows at {PLOT_SIZE} elements/DPU")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    mean_axis, worst_axis = axes
    for model in MODEL_ORDER:
        draw(mean_axis, selected, model, "mean_ms")
        # Max over trials: a compile stall is a rare event, so averaging the
        # per-trial maxima averages away exactly what this panel measures.
        draw(worst_axis, selected, model, "max_max_ms")

    mean_axis.set_title("Mean iteration", fontweight="bold")
    worst_axis.set_title("Worst iteration (max over trials)",
                         fontweight="bold")
    mean_axis.set_ylabel("Time (ms)")
    worst_axis.set_ylabel("Time (ms)")
    dpus = sorted({row.dpus for row in selected})
    for axis in (mean_axis, worst_axis):
        configure_axis(axis, dpus, dpus, "DPUs")

    figure.suptitle(
        f"Adaptive image: dynamic execution trade-offs "
        f"({PLOT_SIZE:,} elements/DPU)", fontsize=14, fontweight="bold")
    figure.legend(handles=legend_handles(MODEL_ORDER, linewidth=2.1),
                  loc="upper center", ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, 0.90))
    figure.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.6)
    figure.savefig(FIGURE)


def main():
    if not RUNS_CSV.is_file():
        raise SystemExit("run adaptive_image.sh before plotting")
    rows = summarize(load_rows())
    if not rows:
        raise SystemExit("no complete adaptive_image measurements found")
    write_summary(SUMMARY_CSV, rows)
    plot(rows)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
