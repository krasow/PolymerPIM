#!/usr/bin/env python3
"""Coordinate-descent traces: how each fusion knob behaved, per benchmark.

Objectives are relative to the search's initial configuration (phase="initial"
in the tuning CSV), so 1.0 means "no better than where the search started".

QUEUE_ABSORB_LIMIT is left out: it moves nothing beyond noise, since an
expression is already fused into one event before it reaches the queue.
"""
import collections
import csv
import statistics

from _plot_common import (
    ANNOTATION_PT,
    AXIS_LABEL_PT,
    BENCHMARK_LABELS,
    FIGURE_TITLE_PT,
    GRID_COLOR,
    GRID_LW,
    RESULTS,
    TEXT_WIDTH_IN,
    TICK_LENGTH,
    TICK_PAD,
    TICK_PT,
    save_figure,
    trim_spines,
)

FUSION = RESULTS / "fusion"
MAIN = RESULTS / "main"
OUTPUT = RESULTS / "fusion-traces"
KNOBS = ("MAX_VFUSE_OPS", "MAX_HFUSE_CHAINS", "JIT_BATCH_SIZE")
KNOB_LABELS = {"MAX_VFUSE_OPS": "Vertical Fusion Ops",
               "MAX_HFUSE_CHAINS": "Horizontal Fusion Chains",
               "JIT_BATCH_SIZE": "JIT Batch Size"}

# Figure parameters.  One panel per knob, one row.
PANEL_BLOCK_IN = 2.34       # the panels alone, stacked under a shared legend
HEADER_IN = 0.42            # title and legend, on the first figure of a stack
PASS_COLORS = ("#3264a8", "#c1662f", "#4a8b4a", "#8b4a8b")
CHOSEN_COLOR = "#a0342c"    # the value the search settled on
INITIAL_COLOR = "#555555"
TRACE_LW = 1.6
TRACE_MARKER_PT = 4.0
INITIAL_MARKER_PT = 6.0
CALLOUT_MARKER_PT = 6.5
CANDIDATE_ROTATION = 30     # 8 candidates in a ~2in panel would touch flat
XLABEL_Y = -0.185           # pinned: rotated ticks hang deeper for long values
TOP_Y = 0.995               # where the topmost text sits, as a figure fraction
HEADER_GAP = 0.012          # between title, legend and the per-figure label
BARE_TOP = 0.938


def chosen_build(stem):
    """What the search settled on: knob values plus the final objective.

    The objective comes from the main run, not the profile: the profile keeps
    the winning single launch, which biases it low by several points.
    """
    path = FUSION / "profiles" / f"{stem}.toml"
    values, meta = {}, {}
    if not path.is_file():
        return values, None, None
    for line in path.read_text().splitlines():
        key, _, value = line.partition(" = ")
        key = key.strip()
        if key in KNOBS:
            values[key] = int(value)
        elif key in ("objective_ms", "dpus", "elements_per_dpu"):
            meta[key] = value.strip()
    claimed = float(meta["objective_ms"]) if "objective_ms" in meta else None
    return values, measured_objective(stem, meta), claimed


def measured_objective(stem, meta):
    """Median time for the tuned build at the cell the search tuned at."""
    path = MAIN / "tuned.csv"
    if not path.is_file() or "dpus" not in meta:
        return None
    dpus = meta["dpus"].strip("[]")
    elements = meta.get("elements_per_dpu", "").strip("[]")
    times = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["benchmark"] == stem and row["variant"] == "polymerpim"
                    and row["status"] == "complete" and row["time"]
                    and row["dpus"] == dpus
                    and row["elements_per_dpu"] == elements):
                times.append(float(row["time"]))
    return statistics.median(times) if times else None


def load(path):
    """-> trace, initial objective, failures, each knob's starting value."""
    trace = collections.defaultdict(lambda: collections.defaultdict(dict))
    initial = None
    start = {}
    failures = collections.defaultdict(set)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] == "initial":
                if row["status"] == "ok" and row["objective_ms"]:
                    initial = float(row["objective_ms"])
                    start = {k: int(row[k]) for k in KNOBS if row.get(k)}
                continue
            knob = row["knob"]
            if knob not in KNOBS:
                continue
            candidate = int(row["candidate"])
            if row["status"] != "ok" or not row["objective_ms"]:
                failures[knob].add(candidate)
                continue
            trace[knob][row["phase"]][candidate] = float(row["objective_ms"])
    return trace, initial, failures, start


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in FUSION.glob("*.csv")
                     if p.stem not in ("runs", "runs.sections"))
    if not sources:
        raise SystemExit(f"no tuning CSVs in {FUSION}")

    for source, with_legend in ((s, legend) for s in sources
                                for legend in (True, False)):
        trace, initial, failures, start = load(source)
        if not trace or not initial:
            print(f"skipped {source.stem}: no swept knobs or no initial run")
            continue
        picked, final, claimed = chosen_build(source.stem)

        # Two variants: with the header, and without it for stacking beneath
        # one that already carries it.  The panels stay the same size in both.
        height = PANEL_BLOCK_IN + (HEADER_IN if with_legend else 0.0)
        figure, axes = plt.subplots(1, len(KNOBS),
                                    figsize=(TEXT_WIDTH_IN, height),
                                    sharey=True)
        for axis, knob in zip(axes, KNOBS):
            passes = sorted(trace.get(knob, {}))
            # Every candidate any pass tried gets a slot, so passes align.
            candidates = {c for p in passes for c in trace[knob][p]}
            if knob in start:
                candidates.add(start[knob])
            if knob in picked:
                candidates.add(picked[knob])
            candidates = sorted(candidates)
            slot = {c: i for i, c in enumerate(candidates)}
            chosen = slot.get(picked.get(knob))
            for index, phase in enumerate(passes):
                points = sorted(trace[knob][phase].items())
                axis.plot([slot[c] for c, _ in points],
                          [obj / initial for _, obj in points],
                          marker="o", markersize=TRACE_MARKER_PT,
                          linewidth=TRACE_LW, label=f"Pass {phase}",
                          color=PASS_COLORS[index % len(PASS_COLORS)])
            if knob in start and start[knob] in slot:
                # The initial config sits at 1.0 by construction.
                axis.plot(slot[start[knob]], 1.0, marker="o",
                          markersize=INITIAL_MARKER_PT,
                          markerfacecolor="white",
                          markeredgecolor=INITIAL_COLOR,
                          markeredgewidth=1.3, linestyle="none",
                          zorder=4, label="Initial Value")
            if final:
                axis.axhline(final / initial, color=CHOSEN_COLOR,
                             linewidth=1.2, linestyle=":")
            if chosen is not None:
                axis.axvline(chosen, color=CHOSEN_COLOR, linewidth=1.2,
                             alpha=0.35, zorder=1)
                # Search estimate vs main-run measurement for the same build;
                # the gap is how far the search can be trusted.
                if claimed:
                    axis.plot(chosen, claimed / initial, marker="D",
                              markersize=CALLOUT_MARKER_PT,
                              markerfacecolor="white",
                              markeredgecolor=CHOSEN_COLOR, markeredgewidth=1.3,
                              zorder=6, linestyle="none",
                              label="Search Estimate")
                if final:
                    axis.plot(chosen, final / initial, marker="D",
                              markersize=CALLOUT_MARKER_PT, color=CHOSEN_COLOR,
                              zorder=6, linestyle="none", label="Measured")
            axis.axhline(1.0, color="#888888", linewidth=1.2, linestyle="--")
            axis.set_xticks(range(len(candidates)))
            axis.set_xticklabels([str(c) for c in candidates], fontsize=TICK_PT,
                                 rotation=CANDIDATE_ROTATION, ha="right",
                                 rotation_mode="anchor")
            # The selected value is called out on the axis, not in the plot.
            if chosen is not None:
                tick = axis.get_xticklabels()[chosen]
                tick.set_color(CHOSEN_COLOR)
                tick.set_fontweight("bold")
            label = KNOB_LABELS[knob]
            if failures.get(knob):
                label += f" ({len(failures[knob])} unbuildable)"
            axis.set_xlabel(label, fontsize=AXIS_LABEL_PT)
            axis.xaxis.set_label_coords(0.5, XLABEL_Y)
            axis.grid(True, axis="y", color=GRID_COLOR, linewidth=GRID_LW)
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=TICK_PT, length=TICK_LENGTH,
                             pad=TICK_PAD)
            trim_spines(axis)

        axes[0].set_ylabel("Time / Initial", fontsize=AXIS_LABEL_PT)
        # First-encounter order interleaves passes and markers; sort instead.
        pairs = {}
        for axis in axes:
            for handle, label in zip(*axis.get_legend_handles_labels()):
                pairs.setdefault(label, handle)
        order = ("Initial Value", "Search Estimate", "Measured")

        def legend_rank(label):
            if label.startswith("Pass "):
                return (0, int(label.split()[1]))
            return (1, order.index(label) if label in order else len(order))

        labels = sorted(pairs, key=legend_rank)
        handles = [pairs[label] for label in labels]
        # No suptitle: these stack in LaTeX under one caption.
        label = BENCHMARK_LABELS.get(source.stem, source.stem)
        local = f"{label}  (Initial {initial:.0f} ms)"
        if with_legend:
            # Shared title, legend, then this figure's own label; a bare
            # figure stacked underneath needs only the last.
            figure.text(0.5, TOP_Y, "Fusion Parameter Search",
                        fontsize=FIGURE_TITLE_PT, ha="center", va="top",
                        fontweight="bold")
            legend = figure.legend(handles, labels, loc="upper center",
                                   fontsize=TICK_PT, ncol=len(labels),
                                   frameon=False,
                                   bbox_to_anchor=(0.5, 0.947),
                                   handletextpad=0.4, columnspacing=1.2)
            # Place the local title under the legend's measured bottom edge.
            figure.canvas.draw()
            box = legend.get_window_extent().transformed(
                figure.transFigure.inverted())
            label_top = box.y0 - HEADER_GAP
            figure.text(0.5, label_top, local, fontsize=ANNOTATION_PT,
                        ha="center", va="top", fontweight="bold")
            top = label_top - ANNOTATION_PT / 72 / height - HEADER_GAP
        else:
            figure.text(0.5, TOP_Y, local, fontsize=ANNOTATION_PT,
                        ha="center", va="top", fontweight="bold")
            top = BARE_TOP
        figure.tight_layout(rect=(0, 0, 1, top), pad=0.4)
        suffix = "" if with_legend else "-no-legend"
        path = OUTPUT / f"{source.stem}{suffix}.pdf"
        save_figure(figure, path)
        plt.close(figure)
        trials = sum(len(v) for phases in trace.values() for v in phases.values())
        starred = sum(1 for k in KNOBS if k in picked and final)
        print(f"Wrote {path}  ({trials} trials, {starred}/{len(KNOBS)} starred, "
              f"final {final / initial:.3f}x initial)")


if __name__ == "__main__":
    main()
