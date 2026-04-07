#!/usr/bin/env python3
"""Generate only the grouped R² bar chart."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

R2_COLUMNS: Dict[str, str] = {
    "Area AVG": "Area AVG R2",
    "RG AVG": "RG AVG R2",
    "RDF Peak": "RDF Peak R2",
}

GROUPED_R2_STYLE = {
    # Colors + hatch are chosen to mimic classic materials journal bar plots.
    "Area AVG": {"facecolor": "#24c1c6", "hatch": ""},
    "RG AVG": {"facecolor": "#524ad3", "hatch": ""},
    "RDF Peak": {"facecolor": "#f78515", "hatch": ""},
}

TARGETS = list(R2_COLUMNS.keys())
DEFAULT_MODEL_ORDER = ["CNN", "GAT", "GCN", "GIN", "LR", "MLP"]

# ---- Figure/style knobs (single place to tune appearance) --------------------
PLOT_STYLE: Dict[str, object] = {
    # Figure sizing
    "default_figwidth": 3.25,  # ACS single-column ≈ 3.25 in
    "default_figheight": 3.0,
    # Bar geometry
    "group_spacing": 1.0,
    "bar_width": 0.22,
    # Axes/labels
    "y_label": "R\u00b2 score",
    # R² cannot exceed 1.0, so keep the axis capped at 1.0 to avoid visual headroom above the max tick.
    "y_lim": (0.0, 1.0),
    "y_ticks": np.arange(0.0, 1.01, 0.2),
    "y_minor_subdivisions": 2,
    # Reference line (set to None to disable)
    "ref_line_y": None,
    "ref_line_color": "black",
    "ref_line_width": 1.0,
    "ref_line_alpha": 0.25,
    "ref_line_linestyle": (0, (3, 2)),  # dashed
    # Legend placement
    # Place legend outside the axes on the top.
    "legend_loc": "lower center",
    "legend_bbox_to_anchor": (0.5, 1.06),
    "legend_ncols": 3,
    "legend_borderaxespad": 0.0,
    # Optional overlay markers (e.g., CV values on top of test-set bars)
    "overlay_marker": ".",
    "overlay_marker_size": 32,
    "overlay_marker_edgecolor": "black",
    "overlay_marker_facecolor": "#FFFFFF",
    "overlay_marker_linewidth": 0.6,
    "overlay_label": "CV",
    # Layout + output
    "tight_layout_pad": 0.3,
    # Leave room on the right for the outside legend.
    "tight_layout_rect": (0, 0, 1, 0.84),
    "output_filename": "r2_grouped.png",
    # Preferred model ordering on the x-axis (any missing models are skipped; any extra models are appended).
    "model_order": DEFAULT_MODEL_ORDER,
    # Matplotlib rcParams (global styling)
    "rcparams": {
        "font.family": "Nimbus Sans",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans",
        "mathtext.bf": "Nimbus Sans",
        "mathtext.sf": "Nimbus Sans",
        "mathtext.tt": "Nimbus Sans",
        "mathtext.cal": "Nimbus Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.2,
        # Patch/bar outlines (bars + legend frame use these defaults).
        "patch.edgecolor": "black",
        "patch.force_edgecolor": True,
        "patch.linewidth": 1.2,
        # Tick styling (no top/right; no x tick marks)
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.bottom": False,
        "xtick.top": False,
        "xtick.labelbottom": True,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "hatch.linewidth": 1.2,
        # Legend frame
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        # Figure/axes backgrounds
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    },
}


def apply_publication_style() -> None:
    """Apply a classic, heavy-spine bar-chart style (similar to many materials journals)."""

    plt.rcParams.update(PLOT_STYLE["rcparams"])


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grouped R² bar chart.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=SCRIPT_DIR / "test_performance.csv",
        help="Path to model performance summary CSV.",
    )
    parser.add_argument(
        "--overlay-summary",
        type=Path,
        default=None,
        help="Optional second summary CSV; if provided, overlay its R² values as markers.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "figures",
        help="Directory where charts will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Image resolution (dots per inch) for saved figures.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=str(PLOT_STYLE["output_filename"]),
        help="Output filename (written inside --outdir).",
    )
    parser.add_argument(
        "--figwidth",
        type=float,
        default=float(PLOT_STYLE["default_figwidth"]),
        help="Figure width in inches (ACS single-column ≈ 3.25 in).",
    )
    parser.add_argument(
        "--figheight",
        type=float,
        default=float(PLOT_STYLE["default_figheight"]),
        help="Figure height in inches.",
    )
    return parser.parse_args(argv)


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip())
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.strip()
    return df


def plot_grouped_r2(
    df: pd.DataFrame,
    overlay_df: pd.DataFrame | None,
    outdir: Path,
    dpi: int,
    figwidth: float,
    figheight: float,
    outfile: str,
) -> None:
    required_cols = list(R2_COLUMNS.values())
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "Summary CSV is missing required columns: " + ", ".join(missing)
        )

    outdir.mkdir(parents=True, exist_ok=True)

    models_in_df = (
        df["model"].astype(str).tolist()
        if "model" in df.columns
        else df.index.astype(str).tolist()
    )
    models_in_df = [m.strip() for m in models_in_df]
    unique_models = list(dict.fromkeys(models_in_df))

    preferred_order = [str(m).strip() for m in PLOT_STYLE.get("model_order", [])]
    ordered_models: list[str] = []
    for model in preferred_order:
        if model in unique_models:
            ordered_models.append(model)
    for model in unique_models:
        if model not in ordered_models:
            ordered_models.append(model)

    df_ordered = (
        df.set_index("model").reindex(ordered_models).dropna(how="all").reset_index()
        if "model" in df.columns
        else df.loc[ordered_models].reset_index()
    )

    group_spacing = float(PLOT_STYLE["group_spacing"])
    x = np.arange(len(ordered_models)) * group_spacing
    width = float(PLOT_STYLE["bar_width"])
    offsets = np.linspace(-width, width, len(TARGETS))

    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    legend_labels = {
        "Area AVG": "Area",
        "RG AVG": r"$R_g$",
        "RDF Peak": "RDF Peak",
    }
    for target, offset in zip(TARGETS, offsets):
        col = R2_COLUMNS[target]
        values = df_ordered[col].astype(float).to_numpy()
        style = GROUPED_R2_STYLE.get(target, {})
        ax.bar(
            x + offset,
            values,
            width=width,
            label=legend_labels.get(target, target.replace(" AVG", "")),
            color=style.get("facecolor", "#cccccc"),
            hatch=style.get("hatch", ""),
        )

        if overlay_df is not None:
            overlay_col = col
            if overlay_col not in overlay_df.columns:
                raise KeyError(
                    "Overlay summary CSV is missing required column: " + str(overlay_col)
                )
            if "model" not in overlay_df.columns:
                raise KeyError("Overlay summary CSV is missing required column: model")

            overlay_map = {
                str(m).strip(): float(v)
                for m, v in zip(
                    overlay_df["model"].astype(str).tolist(),
                    overlay_df[overlay_col].astype(float).tolist(),
                )
            }
            overlay_vals = [overlay_map.get(model, np.nan) for model in ordered_models]
            ax.scatter(
                x + offset,
                overlay_vals,
                marker=str(PLOT_STYLE["overlay_marker"]),
                s=float(PLOT_STYLE["overlay_marker_size"]),
                edgecolors=str(PLOT_STYLE["overlay_marker_edgecolor"]),
                facecolors=str(PLOT_STYLE["overlay_marker_facecolor"]),
                linewidths=float(PLOT_STYLE["overlay_marker_linewidth"]),
                label=str(PLOT_STYLE["overlay_label"]) if target == TARGETS[0] else None,
                zorder=4,
            )

    ax.set_ylabel(str(PLOT_STYLE["y_label"]), fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_models)
    ax.set_ylim(*PLOT_STYLE["y_lim"])
    ax.set_yticks(PLOT_STYLE["y_ticks"])
    ax.yaxis.set_minor_locator(AutoMinorLocator(int(PLOT_STYLE["y_minor_subdivisions"])))
    ref_line_y = PLOT_STYLE.get("ref_line_y")
    if ref_line_y is not None:
        ax.axhline(
            float(ref_line_y),
            color=str(PLOT_STYLE["ref_line_color"]),
            linewidth=float(PLOT_STYLE["ref_line_width"]),
            alpha=float(PLOT_STYLE["ref_line_alpha"]),
            linestyle=PLOT_STYLE["ref_line_linestyle"],
            zorder=0,
        )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc=str(PLOT_STYLE["legend_loc"]),
        bbox_to_anchor=tuple(PLOT_STYLE["legend_bbox_to_anchor"]),
        ncols=int(PLOT_STYLE.get("legend_ncols", 1)) + (1 if overlay_df is not None else 0),
        borderaxespad=float(PLOT_STYLE["legend_borderaxespad"]),
    )
    fig.tight_layout(
        pad=float(PLOT_STYLE["tight_layout_pad"]),
        rect=tuple(PLOT_STYLE["tight_layout_rect"]),
    )
    out_path = outdir / outfile
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(argv: Iterable[str] | None = None) -> None:
    apply_publication_style()
    args = parse_args(argv)
    df = load_summary(args.summary)
    overlay_df = load_summary(args.overlay_summary) if args.overlay_summary else None
    plot_grouped_r2(
        df,
        overlay_df,
        args.outdir,
        args.dpi,
        args.figwidth,
        args.figheight,
        args.outfile,
    )


if __name__ == "__main__":
    main()
