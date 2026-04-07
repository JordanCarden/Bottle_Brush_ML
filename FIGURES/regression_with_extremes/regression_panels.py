#!/usr/bin/env python3
"""Create 3-panel predicted-vs-true regression plots with dataset overlays.

Each panel corresponds to one target (Area, R_g, RDF Peak) and overlays:
  - Random (train-like) samples: light gray circles (many points)
  - Optimized: red squares (combined Extreme 1 + Extreme 2)
  - Categorical: green triangles

The plot includes a dashed y=x reference line in each panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

TARGETS: Tuple[Tuple[str, str, str], ...] = (
    ("Area AVG", "Area", "area"),
    ("RG AVG", r"$R_g$", "rg"),
    ("RDF Peak", "RDF Peak", "rdf"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-panel predicted-vs-true regression plots with overlays."
    )
    parser.add_argument(
        "--model",
        default="GIN",
        help="Model name subdirectory (e.g., GIN, CNN, GCN, GAT, LR, MLP).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "RESULTS_RANDOM_ONLY",
        help="Base results directory (default: RESULTS_RANDOM_ONLY).",
    )
    parser.add_argument(
        "--extreme-1-csv",
        type=Path,
        default=REPO_ROOT / "data" / "extreme_test_set_1.csv",
        help="Ground-truth CSV for optimized set (part 1) (default: data/extreme_test_set_1.csv).",
    )
    parser.add_argument(
        "--extreme-2-csv",
        type=Path,
        default=REPO_ROOT / "data" / "extreme_test_set_2.csv",
        help="Ground-truth CSV for optimized set (part 2) (default: data/extreme_test_set_2.csv).",
    )
    parser.add_argument(
        "--categorical-csv",
        type=Path,
        default=REPO_ROOT / "data" / "test_set.csv",
        help="Ground-truth CSV for categorical set (default: data/test_set.csv).",
    )
    parser.add_argument(
        "--categorical-dirname",
        type=str,
        default="catagorical",
        help="Results subdirectory name for categorical set (default: catagorical).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_PATH.parent,
        help="Output directory (default: FIGURES/regression_with_extremes).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Output filename (default: pred_vs_true_<MODEL>_overlay.png).",
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
        default=2.45,
        help="Figure height in inches (default: 2.45).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for PNG output (default: 600).",
    )
    parser.add_argument(
        "--random-marker-size",
        type=float,
        default=6.0,
        help="Scatter marker size ('s') for random points (default: 6).",
    )
    parser.add_argument(
        "--random-alpha",
        type=float,
        default=0.25,
        help="Alpha transparency for random points (default: 0.25).",
    )
    parser.add_argument(
        "--random-color",
        type=str,
        default="#5F5F5F",
        help="Color for random points (default: #5F5F5F).",
    )
    parser.add_argument(
        "--special-marker-size",
        type=float,
        default=None,
        help="Scatter marker size ('s') for extremes/categorical (default: match random).",
    )
    parser.add_argument(
        "--legend",
        choices=("below", "above", "inside", "none"),
        default="below",
        help="Legend placement (default: below).",
    )
    parser.add_argument(
        "--legend-cols",
        type=int,
        default=4,
        help="Number of legend columns when placed above/below (default: 4).",
    )
    parser.add_argument(
        "--legend-gap",
        type=float,
        default=0.06,
        help="Vertical gap between legend and plots (figure fraction; default: 0.06).",
    )
    parser.add_argument(
        "--score-dataset",
        "--rmse-dataset",
        choices=("random", "extreme", "extreme_1", "extreme_2", "categorical", "all", "none"),
        default="none",
        help=argparse.SUPPRESS,  # Deprecated: R² annotation removed from plots.
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _normalize_input(text: str) -> str:
    return "".join(str(text).split())


def _input_key(df: pd.DataFrame) -> pd.Series | None:
    for col in ("Input List", "Input"):
        if col in df.columns:
            return df[col].astype(str).map(_normalize_input)
    return None


def _load_cv_true_pred(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: str(c).strip())
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"CV predictions file missing y_true/y_pred columns: {path}")
    y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _load_set_true_pred(
    ground_truth_csv: Path,
    predictions_csv: Path,
    *,
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    gt = pd.read_csv(ground_truth_csv).rename(columns=lambda c: str(c).strip())
    pred = pd.read_csv(predictions_csv).rename(columns=lambda c: str(c).strip())

    if target_column not in gt.columns:
        raise ValueError(f"Missing target column {target_column!r} in {ground_truth_csv}")
    if target_column not in pred.columns:
        raise ValueError(f"Missing target column {target_column!r} in {predictions_csv}")

    gt_key = _input_key(gt)
    pred_key = _input_key(pred)
    if gt_key is not None and pred_key is not None:
        gt_frame = pd.DataFrame({"__key": gt_key, "__true": gt[target_column]})
        pred_frame = pd.DataFrame({"__key": pred_key, "__pred": pred[target_column]})

        if gt_frame["__key"].duplicated().any():
            gt_frame = gt_frame.drop_duplicates("__key", keep="first")
            print(f"[WARN] Duplicate Input rows in {ground_truth_csv}; keeping first.")
        if pred_frame["__key"].duplicated().any():
            pred_frame = pred_frame.drop_duplicates("__key", keep="first")
            print(f"[WARN] Duplicate Input rows in {predictions_csv}; keeping first.")

        merged = gt_frame.merge(pred_frame, on="__key", how="inner")
        if merged.empty:
            raise ValueError(
                f"Failed to align predictions to ground truth by Input: "
                f"{ground_truth_csv} vs {predictions_csv}"
            )
        if len(merged) != len(gt_frame) or len(merged) != len(pred_frame):
            print(
                "[WARN] Input mismatch when aligning by Input: "
                f"matched {len(merged)} of {len(gt_frame)} ground-truth rows and "
                f"{len(pred_frame)} prediction rows ({ground_truth_csv.name} vs {predictions_csv})."
            )

        y_true = pd.to_numeric(merged["__true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(merged["__pred"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        return y_true[mask], y_pred[mask]

    if len(gt) != len(pred):
        raise ValueError(
            f"Row mismatch ({len(gt)} vs {len(pred)}) and no Input column to align on: "
            f"{ground_truth_csv} vs {predictions_csv}"
        )

    y_true = pd.to_numeric(gt[target_column], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(pred[target_column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _data_limits(*arrays: np.ndarray) -> Tuple[float, float]:
    vals: list[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.size:
            vals.append(arr)
    if not vals:
        return 0.0, 1.0
    joined = np.concatenate(vals, axis=0)
    joined = joined[np.isfinite(joined)]
    if joined.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(joined))
    vmax = float(np.max(joined))
    if np.isclose(vmin, vmax):
        pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def _nice_equal_ticks(
    vmin: float,
    vmax: float,
    *,
    nbins: int = 4,
    max_ticks: int = 5,
) -> Tuple[float, float, np.ndarray]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        ticks = np.asarray([0.0, 0.5, 1.0], dtype=float)
        return 0.0, 1.0, ticks

    if vmin > vmax:
        vmin, vmax = vmax, vmin

    nbins_i = max(2, int(nbins))
    max_ticks_i = max(2, int(max_ticks))
    ticks: np.ndarray | None = None
    for _ in range(8):
        locator = MaxNLocator(nbins=max(2, nbins_i))
        candidate = locator.tick_values(vmin, vmax)
        candidate = np.asarray([t for t in candidate if np.isfinite(t)], dtype=float)
        if candidate.size >= 2 and candidate.size <= max_ticks_i:
            ticks = candidate
            break
        ticks = candidate
        if candidate.size > max_ticks_i and nbins_i > 2:
            nbins_i -= 1
            continue
        break

    if ticks is None:
        ticks = np.asarray([vmin, vmax], dtype=float)
    if ticks.size < 2:
        lo = float(vmin)
        hi = float(vmax if not np.isclose(vmin, vmax) else (vmin + 1.0))
        ticks = np.asarray([lo, hi], dtype=float)
        return lo, hi, ticks

    lo = float(ticks[0])
    hi = float(ticks[-1])
    if lo == hi:
        hi = lo + 1.0
        ticks = np.asarray([lo, hi], dtype=float)
    return lo, hi, ticks


def _lock_pred_at_zero_for_true_zero(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isclose(y_true, 0.0)
    if not mask.any():
        return y_pred
    y_pred = y_pred.copy()
    y_pred[mask] = 0.0
    return y_pred


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else float("nan")
    return float(1.0 - ss_res / ss_tot)


def _format_r2(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    results_root = resolve_repo_path(args.results_root)
    outdir = resolve_repo_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = str(args.model).strip()
    outfile = args.outfile or f"pred_vs_true_{model}_overlay.png"
    outpath = outdir / outfile

    extreme_1_csv = resolve_repo_path(args.extreme_1_csv)
    extreme_2_csv = resolve_repo_path(args.extreme_2_csv)
    categorical_csv = resolve_repo_path(args.categorical_csv)

    cv_dir = results_root / "cv" / model / "results"
    extreme_1_pred = results_root / "extreme_1" / model / "predictions.csv"
    extreme_2_pred = results_root / "extreme_2" / model / "predictions.csv"
    categorical_pred = results_root / args.categorical_dirname / model / "predictions.csv"

    missing: list[Path] = [
        p
        for p in (
            extreme_1_csv,
            extreme_2_csv,
            categorical_csv,
            extreme_1_pred,
            extreme_2_pred,
            categorical_pred,
        )
        if not p.is_file()
    ]
    for _, _, suffix in TARGETS:
        path = cv_dir / f"cv_predictions_{suffix}.csv"
        if not path.is_file():
            missing.append(path)

    if missing:
        missing_str = "\n  - ".join(str(p) for p in sorted(set(missing)))
        raise SystemExit(f"Missing required inputs:\n  - {missing_str}")

    # Match the general styling used in `plot_bars.py` / other FIGURES scripts.
    plt.rcParams.update(
        {
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
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    colors = {
        "random": str(args.random_color),
        # Keep marker colors aligned with FIGURES/clustering_with_extremes/umap_architecture_descriptors.py
        "extreme": "#D62728",
        "categorical": "#2CA02C",
    }

    random_marker_size = float(args.random_marker_size)
    special_marker_size = (
        float(args.special_marker_size)
        if args.special_marker_size is not None
        else random_marker_size
    )

    legend_random_ms = float(max(3.0, np.sqrt(max(random_marker_size, 0.0)) * 1.7))
    legend_special_ms = float(max(3.0, np.sqrt(max(special_marker_size, 0.0)) * 1.7))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(float(args.figwidth), float(args.figheight)),
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=legend_random_ms,
            markerfacecolor=colors["random"],
            markeredgecolor="none",
            alpha=float(args.random_alpha),
            label="Random",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=legend_special_ms,
            markerfacecolor=colors["extreme"],
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="Optimized",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markersize=legend_special_ms,
            markerfacecolor=colors["categorical"],
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="Categorical",
        ),
    ]

    for ax, (target_col, title, suffix) in zip(axes, TARGETS):
        rand_true, rand_pred = _load_cv_true_pred(cv_dir / f"cv_predictions_{suffix}.csv")
        ex1_true, ex1_pred = _load_set_true_pred(
            extreme_1_csv, extreme_1_pred, target_column=target_col
        )
        ex2_true, ex2_pred = _load_set_true_pred(
            extreme_2_csv, extreme_2_pred, target_column=target_col
        )
        ex_true = np.concatenate([ex1_true, ex2_true])
        ex_pred = np.concatenate([ex1_pred, ex2_pred])
        cat_true, cat_pred = _load_set_true_pred(
            categorical_csv, categorical_pred, target_column=target_col
        )

        if target_col == "RDF Peak":
            rand_pred = _lock_pred_at_zero_for_true_zero(rand_true, rand_pred)
            ex_pred = _lock_pred_at_zero_for_true_zero(ex_true, ex_pred)
            cat_pred = _lock_pred_at_zero_for_true_zero(cat_true, cat_pred)

        data_min, data_max = _data_limits(
            rand_true,
            rand_pred,
            ex_true,
            ex_pred,
            cat_true,
            cat_pred,
        )
        lim_lo, lim_hi, ticks = _nice_equal_ticks(data_min, data_max, nbins=4, max_ticks=5)

        # Intentionally omit per-panel R² annotation: this figure overlays multiple
        # datasets, so a single R² label is easy to misread.

        ax.scatter(
            rand_true,
            rand_pred,
            s=random_marker_size,
            marker="o",
            color=colors["random"],
            alpha=float(args.random_alpha),
            linewidths=0.0,
            zorder=1,
        )
        ax.scatter(
            ex_true,
            ex_pred,
            s=special_marker_size,
            marker="s",
            facecolor=colors["extreme"],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.95,
            zorder=3,
        )
        ax.scatter(
            cat_true,
            cat_pred,
            s=special_marker_size,
            marker="^",
            facecolor=colors["categorical"],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.95,
            zorder=3,
        )

        ax.plot(
            [lim_lo, lim_hi],
            [lim_lo, lim_hi],
            linestyle="--",
            color="black",
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
        )

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        # Avoid duplicated "corner" label at (xmin, ymin): keep the y-axis label and hide x's first label.
        if ticks.size:
            ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(title)
        ax.set_xlabel("True")
        ax.tick_params(direction="in", length=3, width=1.0, top=False, right=False)

    axes[0].set_ylabel("Predicted")
    if args.legend == "inside":
        axes[-1].legend(handles=legend_handles, loc="lower right", frameon=False)
    else:
        ncols = max(1, int(args.legend_cols))
        legend_gap = max(0.0, float(args.legend_gap))

        legend = None
        if args.legend == "above":
            legend = fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncols=ncols,
                frameon=False,
                borderaxespad=0.0,
            )
        elif args.legend == "below":
            legend = fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncols=ncols,
                frameon=False,
                borderaxespad=0.0,
            )

        fig.tight_layout(pad=0.115)

        if legend is not None:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = legend.get_window_extent(renderer=renderer).transformed(
                fig.transFigure.inverted()
            )
            if args.legend == "above":
                top = max(0.0, min(1.0, float(bbox.y0 - legend_gap)))
                fig.tight_layout(pad=0.115, rect=(0, 0, 1, top))
            else:
                bottom = max(0.0, min(1.0, float(bbox.y1 + legend_gap)))
                fig.tight_layout(pad=0.115, rect=(0, bottom, 1, 1))

    fig.savefig(outpath, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
