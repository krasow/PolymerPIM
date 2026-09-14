import csv
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# The shared type scale and colours live one level up, with the main-suite
# plots; these figures sit beside those in the same paper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _plot_common import (  # noqa: E402
    ANNOTATION_PT, AXIS_LABEL_PT, FIGURE_TITLE_PT, GRID_COLOR, GRID_LW,
    GRID_MINOR_COLOR, GRID_MINOR_LW, LAYOUT_PAD, LEGEND_PT, TEXT_WIDTH_IN,
    TICK_PT, save_figure,
)

# Header geometry, as figure fractions.  Both dynamic figures are one row of
# two panels under a title and a single legend row.
TITLE_Y = 0.995
LEGEND_Y = 0.952
HEADER_TOP = 0.915
SERIES_LW = 2.2
MARKER_PT = 6


@dataclass(frozen=True)
class ModelStyle:
    label: str
    color: str
    marker: str


MODEL_STYLES = {
    "polymerpim-jit": ModelStyle("Blocking JIT", "#3264a8", "o"),
    "polymerpim-hybrid": ModelStyle("Hybrid", "#dd7f27", "D"),
    "polymerpim-pipeline": ModelStyle("Interpreter", "#3b8f5a", "s"),
    "polymerpim-eager": ModelStyle("Eager", "#8b5aa5", "^"),
    "simplepim": ModelStyle("SimplePIM", "#b94a48", "^"),
}


def load_pyplot(figure):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(f"matplotlib is required to write {figure}: {error}")
    return plt


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field.name for field in fields(rows[0])]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.3f}" if isinstance(value, float) else value
                for key, value in asdict(row).items()
            })


def draw_series(axis, model, points, yerr=None, linewidth=SERIES_LW):
    if not points:
        return
    style = MODEL_STYLES[model]
    options = dict(
        color=style.color,
        marker=style.marker,
        linewidth=linewidth,
        markersize=MARKER_PT,
        label=style.label,
    )
    x, y = zip(*points)
    if yerr is None:
        axis.plot(x, y, **options)
    else:
        axis.errorbar(x, y, yerr=yerr, capsize=3, **options)


def configure_axis(axis, ticks, labels, xlabel, minor_grid=False):
    axis.set_xscale("log")
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_PT)
    axis.tick_params(labelsize=TICK_PT)
    axis.grid(True, which="major", color=GRID_COLOR, linewidth=GRID_LW)
    if minor_grid:
        axis.grid(True, which="minor", color=GRID_MINOR_COLOR,
                  linewidth=GRID_MINOR_LW)


def draw_header(figure, title, handles, path, w_pad):
    """Title, then one legend row, then the panels; writes the figure."""
    figure.suptitle(title, fontsize=FIGURE_TITLE_PT, fontweight="bold",
                    y=TITLE_Y, va="top")
    figure.legend(handles=handles, fontsize=LEGEND_PT, loc="upper center",
                  ncol=len(handles), frameon=False,
                  bbox_to_anchor=(0.5, LEGEND_Y))
    figure.tight_layout(rect=(0, 0, 1, HEADER_TOP), pad=LAYOUT_PAD, w_pad=w_pad)
    save_figure(figure, path)


def legend_handles(models, linewidth=SERIES_LW):
    from matplotlib.lines import Line2D

    return [Line2D(
        [0], [0], color=MODEL_STYLES[model].color,
        marker=MODEL_STYLES[model].marker, linewidth=linewidth,
        markersize=MARKER_PT, label=MODEL_STYLES[model].label,
    ) for model in models]
