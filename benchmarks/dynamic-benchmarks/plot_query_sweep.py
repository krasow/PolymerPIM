#!/usr/bin/env python3

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from plot_common import (configure_axis, draw_series, legend_handles,
                         load_pyplot, write_summary)


BENCHMARK = "dynamic_query"
COMPILED_MODELS = ("polymerpim-jit", "simplepim")
MODEL_ORDER = (
    "polymerpim-jit",
    "polymerpim-hybrid",
    "polymerpim-pipeline",
    "simplepim",
)

BENCHMARKS = Path(__file__).resolve().parent.parent
RESULTS = BENCHMARKS / "results" / "dynamic"
RUNS_CSV = RESULTS / "query-sweep.csv"
SECTIONS_CSV = RESULTS / "query-sweep.sections.csv"
SUMMARY_CSV = RESULTS / "query-sweep-summary.csv"
# Below ~1M elements the fixed JIT compile cost dominates every variant, so the
# curves measure compiler overhead rather than execution.  Smaller runs stay in
# the CSV; they are simply not plotted.
MIN_TOTAL_ELEMENTS = 1_000_000
FIGURE = RESULTS / "query-sweep.pdf"

RunKey = Tuple[str, str, str, str]
MeasurementKey = Tuple[str, int, int]


@dataclass(frozen=True)
class Measurement:
    model: str
    total_elements: int
    query_ops: int
    first_ms: float
    reuse_ms: float
    query_ms: float
    wall_s: Optional[float]


@dataclass(frozen=True)
class Boundary:
    model: str
    total_elements: int
    query_ops: int
    compile_ms: float
    pipeline_batch_ms: float
    compiled_batch_ms: float
    break_even_batches: float
    measured_query_ms: float
    process_wall_s: float


def parse_parameters(value: str) -> Dict[str, str]:
    return dict(field.split("=", 1) for field in value.split(";") if "=" in field)


def run_key(row: Mapping[str, str]) -> RunKey:
    return row["timestamp"], row["invocation"], row["variant"], row["trial"]


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def load_sections() -> Dict[RunKey, Dict[str, float]]:
    sections = defaultdict(dict)
    with SECTIONS_CSV.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["benchmark"] == BENCHMARK:
                sections[run_key(row)][row["section"]] = float(row["time_ms"])
    return dict(sections)


def load_measurements() -> Dict[MeasurementKey, Measurement]:
    sections = load_sections()
    grouped = defaultdict(lambda: defaultdict(list))

    # A rerun appends rows rather than replacing them, so keep only the newest
    # row per (variant, size, trial); otherwise old and new runs get averaged.
    newest = {}
    with RUNS_CSV.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["benchmark"] != BENCHMARK or row["status"] != "complete":
                continue
            if int(row["total_elements"]) < MIN_TOTAL_ELEMENTS:
                continue
            newest[(row["variant"], row["elements_per_dpu"], row["dpus"],
                    row["trial"])] = row

    for row in newest.values():

        query_ops = int(parse_parameters(row["parameters"])["query_ops"])
        measured = sections.get(run_key(row), {})
        if not {"query_first", "query_reuse"} <= measured.keys():
            continue

        key = row["variant"], int(row["total_elements"]), query_ops
        grouped[key]["first_ms"].append(measured["query_first"])
        grouped[key]["reuse_ms"].append(measured["query_reuse"])
        grouped[key]["query_ms"].append(float(row["time"]))
        if row["real_s"]:
            grouped[key]["wall_s"].append(float(row["real_s"]))

    measurements = {}
    for key, samples in grouped.items():
        model, total_elements, query_ops = key
        measurements[key] = Measurement(
            model=model,
            total_elements=total_elements,
            query_ops=query_ops,
            first_ms=average(samples["first_ms"]),
            reuse_ms=average(samples["reuse_ms"]),
            query_ms=average(samples["query_ms"]),
            wall_s=average(samples["wall_s"]) if samples["wall_s"] else None,
        )
    return measurements


def calculate_boundaries(
    measurements: Mapping[MeasurementKey, Measurement],
) -> List[Boundary]:
    rows = []
    cases = sorted({(measurement.total_elements, measurement.query_ops)
                    for measurement in measurements.values()})

    for total_elements, query_ops in cases:
        pipeline = measurements.get(("polymerpim-pipeline", total_elements, query_ops))
        if pipeline is None:
            continue
        for model in COMPILED_MODELS:
            result = measurements.get((model, total_elements, query_ops))
            if result is None:
                continue
            compile_ms = max(0.0, result.first_ms - result.reuse_ms)
            savings_ms = pipeline.reuse_ms - result.reuse_ms
            rows.append(Boundary(
                model=model,
                total_elements=total_elements,
                query_ops=query_ops,
                compile_ms=compile_ms,
                pipeline_batch_ms=pipeline.reuse_ms,
                compiled_batch_ms=result.reuse_ms,
                break_even_batches=(compile_ms / savings_ms
                                    if savings_ms > 0 else math.inf),
                measured_query_ms=result.query_ms,
                process_wall_s=(result.wall_s
                                if result.wall_s is not None else math.nan),
            ))
    return rows


def model_points(measurements, model, attribute):
    return sorted(
        (measurement.total_elements, getattr(measurement, attribute))
        for measurement in measurements.values()
        if measurement.model == model
        and getattr(measurement, attribute) is not None
    )


def format_elements(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    return f"{value / 1_000:.3g}K"


def plot_results(boundaries, measurements):
    plt = load_pyplot(FIGURE)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    query_axis, boundary_axis = axes

    for model in COMPILED_MODELS:
        rows = [row for row in boundaries if row.model == model]
        draw_series(boundary_axis, model, sorted(
            (row.total_elements, row.break_even_batches)
            for row in rows if math.isfinite(row.break_even_batches)
        ))

    for model in MODEL_ORDER:
        draw_series(query_axis, model, model_points(measurements, model, "query_ms"))

    element_counts = sorted({measurement.total_elements
                             for measurement in measurements.values()})
    for axis in axes:
        configure_axis(
            axis, element_counts,
            [format_elements(value) for value in element_counts],
            "Total elements", minor_grid=True,
        )

    query_axis.set_title("Mean Query Runtime\n"
                         "Cold first batch plus 4 reused batches",
                         fontweight="bold")
    query_axis.set_yscale("log")
    query_axis.set_ylabel("Mean Query Runtime (ms)")

    boundary_axis.set_title(
        "When compilation pays off vs Pipeline\n"
        "Batches to offset cold-start overhead",
        fontweight="bold",
    )
    boundary_axis.set_yscale("log")
    boundary_axis.set_ylabel("Batches to break even")
    # The benchmark reuses each compiled kernel this many times, so curves
    # below the line break even within a single query.
    reuse = {int(parse_parameters(row["parameters"])["batches_per_query"])
             for row in csv.DictReader(RUNS_CSV.open(newline=""))
             if row["benchmark"] == BENCHMARK and row["status"] == "complete"}
    if len(reuse) == 1:
        batches = reuse.pop()
        boundary_axis.axhline(batches, color="#777777", linewidth=1.2,
                              linestyle="--")
        boundary_axis.annotate(f"batches per query = {batches}",
                               xy=(0.02, batches), xycoords=("axes fraction",
                                                             "data"),
                               fontsize=8, color="#555555",
                               va="bottom")

    figure.suptitle("Dynamic query performance", fontsize=15,
                    fontweight="bold")
    figure.legend(handles=legend_handles(MODEL_ORDER), loc="upper center",
                  ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, 0.94))
    figure.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.5)
    figure.savefig(FIGURE)


def main():
    if not RUNS_CSV.is_file() or not SECTIONS_CSV.is_file():
        raise SystemExit("run query_sweep.sh before plotting")

    measurements = load_measurements()
    query_ops = {measurement.query_ops for measurement in measurements.values()}
    if len(query_ops) > 1:
        raise SystemExit("results contain multiple query_ops values; rerun with --reset")
    boundaries = calculate_boundaries(measurements)
    if not boundaries:
        raise SystemExit("no complete dynamic_query measurements found")

    write_summary(SUMMARY_CSV, boundaries)
    plot_results(boundaries, measurements)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
