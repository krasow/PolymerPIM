#!/usr/bin/env python3

"""Plot whole-process runtime breakdowns from completed benchmark trials.

One PDF is emitted per DPU count.  Each benchmark tile contains one stacked
bar per implementation.  Solid segments are one-time setup plus measured-loop
work, hatched segments are warm-up-loop work, and the final residual closes the
stack to the wall time reported by /usr/bin/time.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass

from _plot_common import (
    BENCHMARK_ORDER,
    GRID_COLOR,
    RESULTS,
    RUNS_CSV,
    VARIANT_ORDER,
    VARIANT_STYLES,
    average,
    benchmark_title,
    complete_trial_rows,
    grid_shape,
    load_trials,
    sample_stddev,
    save_figure,
    trim_spines,
)

SECTIONS_CSV = RUNS_CSV.with_name(f"{RUNS_CSV.stem}.sections.csv")
# Stages are recorded in ms; whole-process times read better in seconds.
MS_PER_SECOND = 1000.0

# Figure parameters.  Authored 4in wide against the other figures' 7in, with
# the type scaled to match, so include this one at 0.57\textwidth.
TILE_WIDTH_IN = 2.0
TILE_HEIGHT_IN = 1.44
TITLE_BAND_IN = 0.40
# Fractions of a fixed-width window, so tiles with three and four variants
# draw bars the same width.
BAR_WIDTH = 0.16
BAR_SPACING = 0.26
BAR_EDGE_LW = 0.3
TILE_TITLE_PT = 8
AXIS_LABEL_PT = 7.5
TICK_PT = 7
VARIANT_LABEL_PT = 6.5
VARIANT_ROTATION = 22
LEGEND_PT = 7
GRID_LW = 0.6

SHORT_VARIANT = {
    "polymerpim": "PolymerPIM",
    "baseline": "Baseline",
    "simplepim": "SimplePIM",
    "simplepim-patched": "SimplePIM-DG",
    "julia": "Julia",
}
OUTPUT_DIR = RESULTS / "runtime-decomposition"

STAGES = (
    "alloc",
    "load",
    "init",
    "write",
    "kernel",
    "read",
    "merge",
)

# Deliberately follows ../../benchmarks/reporting/charts/app_runtime.py so the
# same stage has the same visual identity in both repositories.
STAGE_STYLES = {
    "alloc": ("Allocation", "#aec7e8"),
    "load": ("Input load", "#17becf"),
    "init": ("DPU initialization", "#7f7f7f"),
    "write": ("Host → DPU", "#1f77b4"),
    "kernel": ("DPU kernel", "#2ca02c"),
    "read": ("DPU → host", "#ff7f0e"),
    "merge": ("Host merge", "#9467bd"),
    "unmeasured": ("Unmeasured / teardown", "#8c564b"),
}


@dataclass(frozen=True)
class Breakdown:
    benchmark: str
    variant: str
    elements_per_dpu: int
    dpus: int
    ntrials: int
    measured_ms: dict
    cold_ms: dict
    unmeasured_ms: float
    total_ms: float
    total_stddev_ms: float


def run_key(row):
    return (
        row["timestamp"],
        str(row["invocation"]),
        row["benchmark"],
        row["variant"],
        int(row["trial"]),
    )


def load_sections():
    sections = defaultdict(dict)
    malformed = 0
    with SECTIONS_CSV.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                malformed += 1
                continue
            try:
                key = run_key(row)
                stage = row["section"]
                kind = row["kind"]
                value = float(row["time_ms"])
            except (KeyError, TypeError, ValueError):
                malformed += 1
                continue
            if stage in STAGES and kind in ("measured", "cold"):
                sections[key][(stage, kind)] = value
    if malformed:
        print(f"Ignored {malformed} malformed/incomplete section row(s)")
    return sections


def aggregate(latest, sections, selections):
    points = []
    overaccounted = []
    required_sections = {
        (stage, kind) for stage in STAGES for kind in ("measured", "cold")
    }

    for selection in selections:
        for dpus in selection.dpus:
            pending = []
            complete = True
            for variant in selection.variants:
                trials = []
                rows = complete_trial_rows(latest, selection, variant, dpus)
                for row in rows:
                    values = sections.get(run_key(row), {})
                    if not required_sections.issubset(values):
                        complete = False
                        break

                    measured = {
                        stage: values[(stage, "measured")]
                        for stage in STAGES
                    }
                    cold = {
                        stage: values[(stage, "cold")]
                        for stage in STAGES
                    }
                    total = row["real_s"] * 1000.0
                    accounted = sum(measured.values()) + sum(cold.values())
                    residual = max(total - accounted, 0.0)
                    if accounted > total + max(1.0, total * 0.01):
                        overaccounted.append(
                            (selection.name, variant, dpus, row["trial"],
                             accounted - total)
                        )
                    trials.append((measured, cold, residual, total))

                if len(trials) != selection.ntrials:
                    complete = False
                    break

                pending.append(Breakdown(
                    benchmark=selection.name,
                    variant=variant,
                    elements_per_dpu=selection.elements_per_dpu,
                    dpus=dpus,
                    ntrials=len(trials),
                    measured_ms={
                        stage: average([values[0][stage] for values in trials])
                        for stage in STAGES
                    },
                    cold_ms={
                        stage: average([values[1][stage] for values in trials])
                        for stage in STAGES
                    },
                    unmeasured_ms=average([values[2] for values in trials]),
                    total_ms=average([values[3] for values in trials]),
                    total_stddev_ms=sample_stddev(
                        [values[3] for values in trials]),
                ))

            # A tile is only comparable when every configured implementation
            # has all n trials at this benchmark/DPU point.
            if complete:
                points.extend(pending)

    if overaccounted:
        print("Warning: instrumented stages exceeded rounded wall time by >1%:")
        for benchmark, variant, dpus, trial, excess in overaccounted:
            print(f"  {benchmark}/{variant}, {dpus} DPUs, trial {trial}: "
                  f"{excess:.1f} ms")
    return points


def ordered_benchmarks(points, dpus):
    present = {point.benchmark for point in points if point.dpus == dpus}
    return [benchmark for benchmark in BENCHMARK_ORDER if benchmark in present]


def plot_dpu_grid(points, dpus, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    benchmarks = ordered_benchmarks(points, dpus)
    if not benchmarks:
        return False

    # Only claim a slot for the legend when the grid leaves one empty anyway;
    # otherwise it goes above the panels.
    rows, columns = grid_shape(len(benchmarks))
    legend_index = 1 if rows * columns > len(benchmarks) else None
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(TILE_WIDTH_IN * columns,
                 TILE_HEIGHT_IN * rows + TITLE_BAND_IN),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    slots = [j for j in range(rows * columns) if j != legend_index]
    for slot, benchmark in zip(slots, benchmarks):
        axis = flat_axes[slot]
        selected = {
            point.variant: point for point in points
            if point.dpus == dpus and point.benchmark == benchmark
        }
        variants = [variant for variant in VARIANT_ORDER if variant in selected]
        x = np.arange(len(variants)) * BAR_SPACING
        bottom = np.zeros(len(variants))

        def seconds(pick):
            return np.array([pick(selected[variant])
                             for variant in variants]) / MS_PER_SECOND

        # As in the reference, each stage's warm-up contribution immediately
        # caps its solid contribution in the same color with a hatch.
        for stage in STAGES:
            color = STAGE_STYLES[stage][1]
            measured = seconds(lambda point: point.measured_ms[stage])
            axis.bar(
                x, measured, width=BAR_WIDTH, bottom=bottom, color=color,
                edgecolor="black", linewidth=BAR_EDGE_LW,
            )
            bottom += measured

            cold = seconds(lambda point: point.cold_ms[stage])
            axis.bar(
                x, cold, width=BAR_WIDTH, bottom=bottom, color=color,
                hatch="////", edgecolor="black", linewidth=BAR_EDGE_LW,
            )
            bottom += cold

        residual = seconds(lambda point: point.unmeasured_ms)
        axis.bar(
            x, residual, width=BAR_WIDTH, bottom=bottom,
            color=STAGE_STYLES["unmeasured"][1],
            edgecolor="black", linewidth=BAR_EDGE_LW,
        )
        bottom += residual

        totals = seconds(lambda point: point.total_ms)
        errors = seconds(lambda point: point.total_stddev_ms)
        axis.errorbar(
            x, totals, yerr=[np.minimum(errors, totals), errors], fmt="none",
            ecolor="black", elinewidth=0.9, capsize=2.5, capthick=0.9,
            zorder=10,
        )

        size = selected[variants[0]].elements_per_dpu
        axis.set_title(benchmark_title(benchmark, size),
                       fontsize=TILE_TITLE_PT, fontweight="bold", pad=3)
        # Centre the cluster in a window of constant width.
        middle = (len(variants) - 1) * BAR_SPACING / 2
        axis.set_xlim(middle - 0.5, middle + 0.5)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [SHORT_VARIANT.get(variant, VARIANT_STYLES[variant][0])
             for variant in variants],
            rotation=VARIANT_ROTATION, ha="right", fontsize=VARIANT_LABEL_PT,
        )
        if slot % columns == 0:
            axis.set_ylabel("End-to-end time (s)", fontsize=AXIS_LABEL_PT)
        axis.set_ylim(bottom=0)
        axis.margins(y=0.08)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=GRID_LW, alpha=0.75)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=TICK_PT, length=2, pad=1.5)
        trim_spines(axis)

    for slot in slots[len(benchmarks):]:
        flat_axes[slot].set_visible(False)

    stage_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=BAR_EDGE_LW,
              label=label)
        for stage in STAGES
        for label, color in [STAGE_STYLES[stage]]
    ]
    stage_handles.append(Patch(
        facecolor=STAGE_STYLES["unmeasured"][1], edgecolor="black",
        linewidth=BAR_EDGE_LW, label=STAGE_STYLES["unmeasured"][0],
    ))
    stage_handles.append(Patch(
        facecolor="white", edgecolor="black", hatch="////",
        linewidth=BAR_EDGE_LW, label="Warm-up portion",
    ))
    stage_handles.append(Line2D(
        [0], [0], color="black", marker="_", linestyle="none",
        markersize=8, label="Total ± SD",
    ))

    figure.suptitle(f"End-to-End Runtime Decomposition — {dpus} DPUs",
                    fontsize=TILE_TITLE_PT, fontweight="bold", y=0.998,
                    va="top")
    if legend_index is not None:
        legend_axis = flat_axes[legend_index]
        legend_axis.axis("off")
        legend_axis.legend(handles=stage_handles, loc="upper left", ncol=1,
                           frameon=False, fontsize=LEGEND_PT, handlelength=1.7,
                           handletextpad=0.6, labelspacing=0.42,
                           borderpad=0.0, bbox_to_anchor=(-0.02, 1.149))
    else:
        figure.legend(
            handles=stage_handles, loc="upper center", ncol=4, frameon=False,
            bbox_to_anchor=(0.5, 0.94), fontsize=LEGEND_PT,
            columnspacing=1.15, handlelength=1.7,
        )
    figure.align_ylabels()
    top = 0.995 if legend_index is not None else 0.87
    # pad, not h_pad/w_pad, is the gap under the suptitle.
    figure.tight_layout(rect=(0.005, 0.005, 0.995, top), pad=0.15,
                        h_pad=0.6, w_pad=0.5)
    save_figure(figure, output)
    plt.close(figure)
    return True


def main():
    if not SECTIONS_CSV.is_file():
        raise SystemExit(f"missing benchmark sections: {SECTIONS_CSV}")

    selections, latest = load_trials()
    points = aggregate(latest, load_sections(), selections)
    if not points:
        raise SystemExit("no complete benchmark/DPU breakdowns found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for dpus in sorted({point.dpus for point in points}):
        output = OUTPUT_DIR / f"runtime-decomposition-{dpus}-dpus.pdf"
        if plot_dpu_grid(points, dpus, output):
            written.append(output)

    print(f"Averaged {points[0].ntrials} trials per stacked bar")
    for output in written:
        benchmarks = ordered_benchmarks(points, int(output.stem.split("-")[-2]))
        print(f"Wrote {output} ({', '.join(benchmarks)})")


if __name__ == "__main__":
    main()
