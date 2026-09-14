#!/usr/bin/env python3

import csv
import math
from dataclasses import dataclass, replace

from _plot_common import (
    AXIS_LABEL_PT,
    BENCHMARK_ORDER,
    FIGURE_TITLE_PT,
    GRID_COLOR,
    GRID_LW,
    PANEL_TITLE_PT,
    PANEL_WIDTH_IN,
    TEXT_WIDTH_IN,
    TICK_PT,
    VARIANT_ORDER,
    VARIANT_STYLES,
    VIEW,
    average,
    benchmark_title,
    complete_trial_rows,
    grid_shape,
    load_trials,
    sample_stddev,
    save_figure,
)

# Figure parameters.  Sized to the width these are included at; a wider canvas
# is only scaled down by LaTeX, which shrinks the type with it.
SINGLE_PANEL_SIZE = (TEXT_WIDTH_IN, 4.6)
SINGLE_ROW_HEIGHT_IN = 3.2
ROW_HEIGHT_IN = 2.15        # keeps a full grid inside the ~9in text block
# Header pieces in inches; a fraction-based band grows with the figure.
TITLE_IN = 0.20
LEGEND_ROW_IN = 0.21
HEADER_GAP_IN = 0.02
LEGEND_COLUMNS = 3          # what the narrowest figure fits on one row
LEGEND_PT = 12              # panel-sized, not the shared figure-legend size
INSET_LEGEND_ANCHOR = (-0.14, 1.226)   # level with the panel titles
SERIES_LW = 1.8
MARKER_PT = 5.5
CAPSIZE = 2.5
BASELINE_COLOR = "#3b8f5a"

SUMMARY_CSV = VIEW.path("weak-scaling-summary", ".csv")
ITERATION_FIGURE = VIEW.path("weak-scaling-mean-iteration", ".pdf")
RUNTIME_FIGURE = VIEW.path("weak-scaling-total-runtime", ".pdf")
NORMALIZED_FIGURE = VIEW.path("weak-scaling-mean-iteration-normalized", ".pdf")


@dataclass(frozen=True)
class Point:
    benchmark: str
    variant: str
    elements_per_dpu: int
    dpus: int
    ntrials: int
    mean_iteration_ms: float
    iteration_stddev_ms: float
    mean_runtime_s: float
    runtime_stddev_s: float


def normalize_to_baseline(points, value, error):
    """Points rescaled so the hand-tuned baseline is 1.0 at every DPU count."""
    reference = {
        (point.benchmark, point.dpus): getattr(point, value)
        for point in points if point.variant == "baseline"
    }
    scaled = []
    for point in points:
        if point.variant == "baseline":
            continue
        divisor = reference.get((point.benchmark, point.dpus))
        if not divisor:
            continue
        scaled.append(replace(point, **{
            value: getattr(point, value) / divisor,
            error: getattr(point, error) / divisor,
        }))
    return scaled


def aggregate(latest, selections):
    points = []
    complete_benchmarks = []
    for selection in selections:
        benchmark_points = []
        complete = True
        for variant in selection.variants:
            for dpus in selection.dpus:
                trials = complete_trial_rows(latest, selection, variant, dpus)
                if not trials:
                    complete = False
                    continue
                iteration_values = [row["time"] for row in trials]
                runtime_values = [row["real_s"] for row in trials]
                benchmark_points.append(Point(
                    benchmark=selection.name,
                    variant=variant,
                    elements_per_dpu=selection.elements_per_dpu,
                    dpus=dpus,
                    ntrials=len(trials),
                    mean_iteration_ms=average(iteration_values),
                    iteration_stddev_ms=sample_stddev(iteration_values),
                    mean_runtime_s=average(runtime_values),
                    runtime_stddev_s=sample_stddev(runtime_values),
                ))
        if complete:
            complete_benchmarks.append(selection.name)
            points.extend(benchmark_points)
        else:
            print(f"Omitting incomplete benchmark: {selection.name}")
    return points, complete_benchmarks


def geomean(values):
    if not values or any(value <= 0 for value in values):
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def aggregate_speedup(points, candidate, reference, value):
    indexed = {
        (point.benchmark, point.variant, point.dpus): getattr(point, value)
        for point in points
    }
    benchmark_speedups = []
    for benchmark in BENCHMARK_ORDER:
        candidate_points = sorted(
            (point for point in points
             if point.benchmark == benchmark and point.variant == candidate),
            key=lambda point: point.dpus,
        )
        ratios = []
        for point in candidate_points:
            candidate_time = getattr(point, value)
            baseline_time = indexed.get((benchmark, "baseline", point.dpus))
            simplepim_time = indexed.get((benchmark, "simplepim", point.dpus))
            if reference == "baseline":
                reference_time = baseline_time
            elif reference == "simplepim":
                reference_time = simplepim_time
            else:
                available = [time for time in (baseline_time, simplepim_time)
                             if time is not None]
                reference_time = min(available) if available else None
            if (reference_time is not None and reference_time > 0
                    and candidate_time > 0):
                ratios.append(reference_time / candidate_time)
        if ratios:
            benchmark_speedups.append(geomean(ratios))
    return geomean(benchmark_speedups), len(benchmark_speedups)


def print_speedups(points, value, heading):
    # Only meaningful for a suite that has a reference to compare against; the
    # build-mode suite is all one implementation.
    candidates = [(name, label) for name, label in (("polymerpim", "PolymerPIM"),
                                                    ("julia", "Julia"))
                  if name in VARIANT_ORDER]
    if not candidates or not any(r in VARIANT_ORDER
                                 for r in ("baseline", "simplepim")):
        return
    print(f"{heading} (geomean across benchmarks):")
    for candidate, label in candidates:
        comparisons = []
        benchmark_count = 0
        references = [(r, l) for r, l in (
            ("baseline", "baseline"),
            ("simplepim", "SimplePIM"),
            ("best", "best of baseline and SimplePIM"))
            if VIEW.keeps(r)]
        for reference, reference_label in references:
            speedup, count = aggregate_speedup(
                points, candidate, reference, value)
            benchmark_count = max(benchmark_count, count)
            comparisons.append(f"{speedup:.3f}x vs {reference_label}")
        print(f"  {label} ({benchmark_count} benchmarks): "
              + ", ".join(comparisons))


def configure_dpu_axis(axis, dpus):
    try:
        axis.set_xscale("log", base=2)
    except (TypeError, ValueError):
        axis.set_xscale("log", basex=2)
    axis.set_xticks(dpus)
    axis.set_xticklabels([str(value) for value in dpus])
    axis.minorticks_off()
    axis.grid(True, color=GRID_COLOR, linewidth=GRID_LW, alpha=0.8)
    axis.tick_params(labelsize=TICK_PT)


def draw_header(figure, handles, title, with_legend=True):
    """Title above a legend; returns the top of the axes area as a fraction."""
    height = figure.get_figheight()
    figure.suptitle(title, fontsize=FIGURE_TITLE_PT, fontweight="bold",
                    y=0.995, va="top")
    if not with_legend:
        return 1.0 - (TITLE_IN + HEADER_GAP_IN) / height
    columns = min(len(handles), LEGEND_COLUMNS)
    rows = -(-len(handles) // columns)
    figure.legend(handles=handles, loc="upper center", ncol=columns,
                  frameon=False, fontsize=LEGEND_PT,
                  bbox_to_anchor=(0.5, 1.0 - TITLE_IN / height))
    band = TITLE_IN + rows * LEGEND_ROW_IN + HEADER_GAP_IN
    return 1.0 - band / height


def plot_grid(points, benchmarks, value, error, ylabel, title, output,
              baseline_at_one=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Only claim a slot for the legend when the grid leaves one empty anyway;
    # otherwise it goes above the panels.
    rows, columns = grid_shape(len(benchmarks))
    legend_index = 1 if rows * columns > len(benchmarks) else None
    if len(benchmarks) == 1:
        size = SINGLE_PANEL_SIZE
    elif rows == 1:
        size = (min(PANEL_WIDTH_IN * columns, TEXT_WIDTH_IN),
                SINGLE_ROW_HEIGHT_IN)
    else:
        size = (PANEL_WIDTH_IN * columns, rows * ROW_HEIGHT_IN)
    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    flat_axes = axes.ravel()

    slots = [j for j in range(rows * columns) if j != legend_index]
    for slot, benchmark in zip(slots, benchmarks):
        axis = flat_axes[slot]
        selected = [point for point in points if point.benchmark == benchmark]
        dpus = sorted({point.dpus for point in selected})
        elements_per_dpu = selected[0].elements_per_dpu
        for variant in VARIANT_ORDER:
            series = sorted((point for point in selected
                             if point.variant == variant),
                            key=lambda point: point.dpus)
            if not series:
                continue
            label, color, marker, linestyle = VARIANT_STYLES[variant]
            axis.errorbar(
                [point.dpus for point in series],
                [getattr(point, value) for point in series],
                yerr=[getattr(point, error) for point in series],
                color=color, marker=marker, linestyle=linestyle,
                linewidth=SERIES_LW, markersize=MARKER_PT, capsize=CAPSIZE,
                label=label,
            )

        axis.set_title(benchmark_title(benchmark, elements_per_dpu),
                       fontsize=PANEL_TITLE_PT, fontweight="bold")
        configure_dpu_axis(axis, dpus)
        if VIEW.log_y:
            axis.set_yscale("log")
        axis.set_xlabel("DPUs", fontsize=AXIS_LABEL_PT)
        if baseline_at_one:
            axis.axhline(1.0, color=BASELINE_COLOR, linestyle="--",
                         linewidth=1.2, zorder=1)
        if slot % columns == 0:
            axis.set_ylabel(ylabel, fontsize=AXIS_LABEL_PT)
        axis.margins(y=0.12)

    for slot in slots[len(benchmarks):]:
        flat_axes[slot].set_visible(False)

    drawn = {point.variant for point in points}
    handles = [
        Line2D([0], [0], color=color, marker=marker, linestyle=linestyle,
               linewidth=SERIES_LW, markersize=MARKER_PT, label=label)
        for variant in VARIANT_ORDER if variant in drawn
        for label, color, marker, linestyle in [VARIANT_STYLES[variant]]
    ]
    if baseline_at_one:
        handles.append(Line2D([0], [0], color=BASELINE_COLOR, linestyle="--",
                              linewidth=1.2, label="Hand-tuned baseline"))
    inset = flat_axes[legend_index] if legend_index is not None else None
    if inset is not None:
        inset.axis("off")
        # Anchored above the axes box so it sits level with the panel titles
        # rather than the plot areas.
        inset.legend(handles=handles, loc="upper left", frameon=False, ncol=1,
                     fontsize=LEGEND_PT,
                     bbox_to_anchor=INSET_LEGEND_ANCHOR)
    # Tick label widths differ per row, so the y-labels land at different x
    # unless they are aligned explicitly.
    figure.align_ylabels()
    top = draw_header(figure, handles, title, with_legend=inset is None)
    figure.tight_layout(rect=(0, 0, 1, top), pad=0.1, h_pad=0.6, w_pad=1.0)
    save_figure(figure, output)
    plt.close(figure)


def write_summary(points):
    with SUMMARY_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "benchmark", "variant", "elements_per_dpu", "dpus", "ntrials",
            "mean_iteration_ms", "iteration_stddev_ms",
            "mean_runtime_s", "runtime_stddev_s",
        ])
        writer.writeheader()
        for point in points:
            writer.writerow({
                "benchmark": point.benchmark,
                "variant": point.variant,
                "elements_per_dpu": point.elements_per_dpu,
                "dpus": point.dpus,
                "ntrials": point.ntrials,
                "mean_iteration_ms": f"{point.mean_iteration_ms:.6f}",
                "iteration_stddev_ms": f"{point.iteration_stddev_ms:.6f}",
                "mean_runtime_s": f"{point.mean_runtime_s:.6f}",
                "runtime_stddev_s": f"{point.runtime_stddev_s:.6f}",
            })


def main():
    selections, latest = load_trials()
    points, benchmarks = aggregate(latest, selections)
    if not benchmarks:
        raise SystemExit("no complete benchmark grids found")

    write_summary(points)
    plot_grid(
        points, benchmarks, "mean_iteration_ms", "iteration_stddev_ms",
        "Mean iteration time (ms)",
        "Weak Scaling: Mean Iteration Time", ITERATION_FIGURE,
    )
    plot_grid(
        points, benchmarks, "mean_runtime_s", "runtime_stddev_s",
        "Mean process runtime (s)",
        "Weak Scaling: End-to-End Runtime", RUNTIME_FIGURE,
    )
    normalized = normalize_to_baseline(
        points, "mean_iteration_ms", "iteration_stddev_ms")
    if normalized:
        plot_grid(
            normalized, benchmarks, "mean_iteration_ms",
            "iteration_stddev_ms",
            "Relative to baseline",
            "Weak Scaling: Mean Iteration Time, Normalized",
            NORMALIZED_FIGURE, baseline_at_one=True,
        )
    print(f"Averaged {points[0].ntrials} trials per data point")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print_speedups(points, "mean_iteration_ms", "Mean iteration speedup")
    print_speedups(points, "mean_runtime_s", "End-to-end runtime speedup")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {ITERATION_FIGURE}")
    if normalized:
        print(f"Wrote {NORMALIZED_FIGURE}")
    print(f"Wrote {RUNTIME_FIGURE}")


if __name__ == "__main__":
    main()
