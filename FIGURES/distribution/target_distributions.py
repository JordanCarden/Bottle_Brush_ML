#!/usr/bin/env python3
"""Generate target-property distribution plots (histogram + smooth KDE) with overlays.

This mirrors the style of `FIGURES/distribution/feature_distributions.py`, but plots
the *target* columns from a raw dataset CSV:
  - Area AVG
  - RG AVG
  - RDF Peak

Optionally overlays additional raw CSVs as rug marks (e.g., extremes / categorical)
to show where those samples fall relative to the random dataset.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

TARGET_COLUMNS = ("Area AVG", "RG AVG", "RDF Peak")

TARGET_LABELS = {
    "Area AVG": "Area",
    "RG AVG": r"$R_g$",
    "RDF Peak": "RDF Peak",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-property distribution plots.")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=REPO_ROOT / "data" / "ES.csv",
        help="Raw dataset CSV containing Input List + targets (default: data/ES.csv).",
    )
    parser.add_argument(
        "--overlay",
        action="append",
        nargs=2,
        metavar=("LABEL", "RAW_CSV"),
        default=[],
        help="Overlay samples from a raw CSV as rug marks (repeatable).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "FIGURES" / "target_distribution_with_extremes",
        help="Directory where PNGs will be written.",
    )
    parser.add_argument(
        "--figwidth",
        type=float,
        default=3.25,
        help="Figure width in inches (default: 3.25, ACS single-column).",
    )
    parser.add_argument(
        "--figheight",
        type=float,
        default=3.25,
        help="Figure height in inches (default: 3.25, square single-column).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output (default: 300).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=0,
        help="Fixed bin count (0 = use Freedman–Diaconis; default: 0).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=256,
        help="Number of x points used for KDE curve (default: 256).",
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _sanitize_filename(text: str) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug or "target"


def _fd_bins(values: np.ndarray) -> int:
    """Freedman–Diaconis rule (with reasonable clamping)."""
    n = int(values.size)
    if n < 2:
        return 1
    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    span = data_max - data_min
    if span <= 0:
        return 1
    if iqr <= 0:
        return min(30, max(8, int(round(math.sqrt(n)))))
    bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
    if bin_width <= 0:
        return min(30, max(8, int(round(math.sqrt(n)))))
    bins = int(math.ceil(span / bin_width))
    return max(8, min(50, bins))


def gaussian_kde_1d(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Simple Gaussian KDE with Silverman's bandwidth."""
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n < 2:
        return np.zeros_like(grid, dtype=float)
    std = float(np.std(x, ddof=1))
    if not math.isfinite(std) or std <= 0:
        return np.zeros_like(grid, dtype=float)
    h = 1.06 * std * (n ** (-1.0 / 5.0))
    if not math.isfinite(h) or h <= 0:
        return np.zeros_like(grid, dtype=float)
    z = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (n * h * math.sqrt(2.0 * math.pi))
    return dens.astype(float)


def _set_count_axis_limits(ax: plt.Axes, max_count: float) -> None:
    """Make the top of the y-axis land exactly on a major tick."""
    if not np.isfinite(max_count) or max_count <= 0:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        return

    locator = MaxNLocator(nbins=5, integer=True, min_n_ticks=4)
    ticks = locator.tick_values(0, float(max_count))
    ticks = np.asarray([t for t in ticks if t >= 0])
    if ticks.size == 0:
        top = float(max_count)
    else:
        top = float(ticks[-1])
        if top < float(max_count):
            step = float(ticks[-1] - ticks[-2]) if ticks.size >= 2 else 1.0
            top = float(np.ceil(float(max_count) / step) * step)
            ticks = locator.tick_values(0, top)
            ticks = np.asarray([t for t in ticks if t >= 0])

    ax.set_ylim(0, top)
    ax.set_yticks(ticks)


def _draw_stacked_rug(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    y0: float,
    y1: float,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    """Draw rug marks that preserve multiplicity even when x values repeat."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    band = float(y1 - y0)
    if band <= 0:
        return

    rounded = np.round(values, 6)
    groups: dict[float, list[float]] = {}
    for key, val in zip(rounded, values):
        groups.setdefault(float(key), []).append(float(val))

    max_count = max(len(v) for v in groups.values())
    if max_count <= 0:
        return

    step = band / float(max_count)
    gap = 0.4 * step
    seg_height = max(0.0, step - gap)

    segments: list[list[tuple[float, float]]] = []
    for key in sorted(groups.keys()):
        stack_count = len(groups[key])
        x = float(np.median(groups[key]))
        for idx in range(stack_count):
            y_start = y0 + idx * step + gap / 2.0
            segments.append([(x, y_start), (x, y_start + seg_height)])

    if not segments:
        return

    collection = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
        transform=ax.get_xaxis_transform(),  # x in data, y in axes fraction
    )
    collection.set_capstyle("butt")
    ax.add_collection(collection)


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 1.2,
            "patch.edgecolor": "black",
            "patch.force_edgecolor": True,
            "patch.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_target_distribution(
    series: pd.Series,
    *,
    xlabel: str,
    outpath: Path,
    figwidth: float,
    figheight: float,
    bins: int,
    max_points: int,
    dpi: int,
    overlays: Sequence[Tuple[str, np.ndarray]] = (),
) -> None:
    values = series.dropna().astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    ax.grid(False)

    if values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        fig.tight_layout(pad=0.25)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    bin_spec = int(bins) if bins and bins > 0 else _fd_bins(values)
    counts, edges, _ = ax.hist(
        values,
        bins=bin_spec,
        density=False,
        color=mcolors.to_rgba("#24c1c6", 0.35),
        edgecolor="black",
        linewidth=1.2,
    )

    if len(counts):
        _set_count_axis_limits(ax, float(np.max(counts)))

    data_min = float(np.min(values))
    data_max = float(np.max(values))
    span = data_max - data_min
    if span > 0 and max_points >= 32:
        pad = 0.05 * span
        grid = np.linspace(data_min - pad, data_max + pad, int(max_points))
        kde = gaussian_kde_1d(values, grid)
        if np.any(kde > 0) and len(edges) >= 2:
            bin_width = float(edges[1] - edges[0])
            kde_counts = kde * float(values.size) * bin_width
            ax.plot(grid, kde_counts, color="#524ad3", linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    if overlays:
        rug_total_height = 0.32
        group_height = rug_total_height / max(1, len(overlays))
        overlay_colors = ["#f78515", "#ff0000", "#2ca02c", "#9467bd"]
        handles: List[Line2D] = []
        for idx, (label, overlay_values) in enumerate(overlays):
            color = overlay_colors[idx % len(overlay_colors)]
            overlay_values = np.asarray(overlay_values, dtype=float)
            overlay_values = overlay_values[np.isfinite(overlay_values)]
            if overlay_values.size == 0:
                continue
            y0 = float(idx) * group_height
            y1 = y0 + group_height
            _draw_stacked_rug(
                ax,
                overlay_values,
                y0=y0,
                y1=y1,
                color=color,
                linewidth=2.0,
                alpha=0.9,
                zorder=3,
            )
            handles.append(Line2D([0], [0], color=color, lw=1.2, label=label))

        if handles:
            legend = ax.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.06),
                ncols=max(1, min(3, len(handles))),
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                edgecolor="black",
                facecolor="white",
                borderaxespad=0.0,
            )
            legend.get_frame().set_linewidth(1.2)
            fig.tight_layout(pad=0.25, rect=(0, 0, 1, 0.88))
        else:
            fig.tight_layout(pad=0.25)
    else:
        fig.tight_layout(pad=0.25)

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    apply_publication_style()

    raw_csv = resolve_repo_path(args.raw_csv)
    df = pd.read_csv(raw_csv)

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            raise SystemExit(f"Missing required target column '{col}' in {raw_csv}")

    overlays: List[Tuple[str, pd.DataFrame]] = []
    for label, raw_path in args.overlay:
        overlay_df = pd.read_csv(resolve_repo_path(Path(raw_path)))
        overlays.append((label, overlay_df))

    outdir = resolve_repo_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for col in TARGET_COLUMNS:
        outname = f"dist_{_sanitize_filename(col)}.png"
        overlay_values: List[Tuple[str, np.ndarray]] = []
        for label, overlay_df in overlays:
            if col not in overlay_df.columns:
                continue
            overlay_values.append((label, overlay_df[col].dropna().to_numpy(dtype=float)))

        plot_target_distribution(
            df[col],
            xlabel=TARGET_LABELS.get(col, col),
            outpath=outdir / outname,
            figwidth=float(args.figwidth),
            figheight=float(args.figheight),
            bins=int(args.bins),
            max_points=int(args.max_points),
            dpi=int(args.dpi),
            overlays=overlay_values,
        )

    print(f"Wrote {len(TARGET_COLUMNS)} target distribution PNGs to {outdir}")


if __name__ == "__main__":
    main()

