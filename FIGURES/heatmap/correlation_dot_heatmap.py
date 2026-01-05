#!/usr/bin/env python3
"""Create a dot heatmap of the Pearson correlation matrix for engineered features.

Dot color encodes correlation sign/magnitude; dot size encodes |correlation|.

This figure is styled to match `plot_bars.py` (serif fonts, heavy spines) and is
intended for publication output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

TARGET_COLUMNS = {"Area AVG", "RG AVG", "RDF Peak"}

ABBREVIATIONS = {
    "backbone_length": r"$N_{bb}$",
    "mean_charge": r"$\langle q\rangle$",
    "sum_charge": r"$\sum_i q_i$",
    "max_charge": r"$q_{\max}$",
    "min_charge": r"$q_{\min}$",
    "std_charge": r"$\sigma_q$",
    "mean_length": r"$\langle |q| \rangle$",
    "max_length": r"$|q|_{\max}$",
    "std_length": r"$\sigma_{|q|}$",
    "max_S_block": r"$L_{\max}^{S}$",
    "max_E_block": r"$L_{\max}^{E}$",
    "transitions": r"$n_{\mathrm{trans}}$",
    "max_block_size": r"$L_{\max}$",
    "min_block_size": r"$L_{\min}$",
    "mean_block_size": r"$\langle L \rangle$",
    "std_block_size": r"$\sigma_L$",
    "blockiness": r"$B$",
    "gini": r"$G$",
    "max_fft_value": r"$\max|\mathcal{F}(q)|$",
    "mean_fft_value": r"$\langle|\mathcal{F}(q)|\rangle$",
    "sum_fft_value": r"$\sum|\mathcal{F}(q)|$",
    "std_fft_value": r"$\sigma_{|\mathcal{F}(q)|}$",
    "hydrophobic_ratio": r"$f_S$",
    "hydrophobic_ratio_weighted": r"$f_S^{(w)}$",
    "harwoods_blockiness": r"$B_{\mathrm{H}}$",
    "mayo_lewis": r"$r_{\mathrm{ML}}$",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dot heatmap of Pearson correlations.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=REPO_ROOT / "LR" / "processed" / "ES_features.csv",
        help="Path to engineered feature CSV (default: LR/processed/ES_features.csv).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_PATH.parent,
        help="Output directory (default: FIGURES/heatmap).",
    )
    parser.add_argument(
        "--outfile-stem",
        type=str,
        default="pearson_correlation_dot_heatmap",
        help="Output filename stem (default: pearson_correlation_dot_heatmap).",
    )
    parser.add_argument(
        "--figwidth",
        type=float,
        default=6.5,
        help="Figure width in inches (default: 6.5, ACS two-column).",
    )
    parser.add_argument(
        "--figheight",
        type=float,
        default=6.0,
        help="Figure height in inches (default: 6.0).",
    )
    parser.add_argument(
        "--triangle",
        choices=["lower", "upper", "full"],
        default="lower",
        help="Which part of the matrix to plot (default: lower).",
    )
    parser.add_argument(
        "--abbreviate",
        action="store_true",
        help="Use abbreviated feature labels (recommended for single-column figures).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Optional list of feature columns to exclude.",
    )
    parser.add_argument(
        "--max-dot-area",
        type=float,
        default=120.0,
        help="Maximum dot area (matplotlib 's' units) when |r|=1 (default: 120).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output (default: 300).",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit title (recommended when placing caption in the manuscript).",
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    """Resolve a user-supplied path relative to the repo root when not absolute."""
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _feature_columns(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    exclude_set = {c.strip() for c in exclude}
    cols: List[str] = []
    for col in df.columns:
        col = str(col).strip()
        if col == "Input List":
            continue
        if col in TARGET_COLUMNS:
            continue
        if col in exclude_set:
            continue
        cols.append(col)
    return cols


def _labels_for(columns: List[str], abbreviate: bool) -> List[str]:
    if not abbreviate:
        return columns
    labels: List[str] = []
    for col in columns:
        labels.append(ABBREVIATIONS.get(col, col))
    return labels


def plot_corr_dot_heatmap(
    corr: pd.DataFrame,
    *,
    outdir: Path,
    outfile_stem: str,
    max_dot_area: float,
    dpi: int,
    figwidth: float,
    figheight: float,
    show_title: bool,
    triangle: str,
    abbreviate: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    cols = list(corr.columns)
    n = len(cols)
    if n == 0:
        raise ValueError("No feature columns found for correlation plot.")

    # Match the general styling used in `plot_bars.py`.
    plt.rcParams.update(
        {
            "font.family": "serif",
            # Match `plot_bars.py` as rendered in this environment (Times fonts are not installed here),
            # so we explicitly use DejaVu Serif for consistent typography across figures.
            "font.serif": ["DejaVu Serif"],
            # Ensure mathtext (e.g., $N_{bb}$, $\langle q\rangle$) also uses a serif face.
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 1.2,
            "patch.edgecolor": "black",
            "patch.force_edgecolor": True,
            "patch.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    corr_np = corr.to_numpy()
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    x = xs.ravel()
    y = ys.ravel()
    r = corr_np.ravel()

    if triangle != "full":
        if triangle == "lower":
            keep = y >= x
        elif triangle == "upper":
            keep = y <= x
        else:
            raise ValueError(f"Unexpected triangle option: {triangle}")
        x = x[keep]
        y = y[keep]
        r = r[keep]

    sizes = np.clip(np.abs(r), 0.0, 1.0) * float(max_dot_area)
    ax.scatter(
        x,
        y,
        c=r,
        s=sizes,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        edgecolors="none",
    )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # invert y for matrix-style layout
    ax.set_aspect("equal", adjustable="box")

    tick_labels = _labels_for(cols, abbreviate)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tick_labels, rotation=90, ha="center")
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis="both", which="both", length=0)

    # Light grid to emphasize cells.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="0.85", linestyle="-", linewidth=0.7)
    ax.grid(which="major", visible=False)

    if show_title:
        ax.set_title("Pearson correlation", pad=6, fontsize=8)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-1.0, 1.0)),
        ax=ax,
        orientation="vertical",
        fraction=0.06,
        pad=0.02,
    )
    cbar.set_label("Pearson r")

    png_path = outdir / f"{outfile_stem}.png"
    fig.tight_layout(pad=0.4)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    features_csv = resolve_repo_path(args.features_csv)
    outdir = resolve_repo_path(args.outdir)
    df = pd.read_csv(features_csv)

    feature_cols = _feature_columns(df, args.exclude)
    corr = df.loc[:, feature_cols].corr(method="pearson")
    plot_corr_dot_heatmap(
        corr,
        outdir=outdir,
        outfile_stem=args.outfile_stem,
        max_dot_area=args.max_dot_area,
        dpi=args.dpi,
        figwidth=args.figwidth,
        figheight=args.figheight,
        show_title=not args.no_title,
        triangle=args.triangle,
        abbreviate=bool(args.abbreviate),
    )


if __name__ == "__main__":
    main()
