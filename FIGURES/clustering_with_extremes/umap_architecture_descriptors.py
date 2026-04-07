#!/usr/bin/env python3
"""UMAP projection of LR/MLP engineered architecture descriptors (with overlays).

This script is intended for the "are these test sets out-of-distribution?" story:
it embeds bottlebrush *input* descriptors (no targets, no errors) into 2D using
UMAP, then overlays the categorical and optimized sets on top of the random set.
Samples from extreme_test_set_1/2 are combined into a single "Optimized" group.

By default:
  - Features are engineered from the same "Input List" strings used elsewhere.
  - Preprocessing follows the LR pipeline's monotonic transforms + scaling.
  - PCA is applied before UMAP (common best practice for stability).
  - UMAP is fit on the random set and used to transform the other sets.

Example:
  ./venv/bin/python FIGURES/clustering_with_extremes/umap_architecture_descriptors.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

TARGET_COLUMNS = {"Area AVG", "RG AVG", "RDF Peak"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UMAP projection of engineered architecture descriptors."
    )
    parser.add_argument(
        "--random-csv",
        type=Path,
        default=REPO_ROOT / "data" / "ES.csv",
        help="Random (training-like) raw CSV (default: data/ES.csv).",
    )
    parser.add_argument(
        "--categorical-csv",
        type=Path,
        default=REPO_ROOT / "data" / "test_set.csv",
        help="Categorical/structured raw CSV (default: data/test_set.csv).",
    )
    parser.add_argument(
        "--extreme-1-csv",
        type=Path,
        default=REPO_ROOT / "data" / "extreme_test_set_1.csv",
        help="Optimized set (part 1) raw CSV (default: data/extreme_test_set_1.csv).",
    )
    parser.add_argument(
        "--extreme-2-csv",
        type=Path,
        default=REPO_ROOT / "data" / "extreme_test_set_2.csv",
        help="Optimized set (part 2) raw CSV (default: data/extreme_test_set_2.csv).",
    )
    parser.add_argument(
        "--fit-on",
        choices=("random", "all"),
        default="random",
        help="Which samples define the UMAP fit (default: random).",
    )
    parser.add_argument(
        "--include-redundant",
        action="store_true",
        help="Keep 'max_length' and 'min_block_size' (default: drop them).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=20,
        help="PCA components before UMAP (0 disables PCA; default: 20).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors (default: 30).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="UMAP metric (default: euclidean).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA/UMAP (default: 42).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_PATH.parent,
        help="Output directory (default: FIGURES/clustering_with_extremes).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="umap_architecture_descriptors.png",
        help="Output filename (default: umap_architecture_descriptors.png).",
    )
    parser.add_argument(
        "--save-embedding-csv",
        type=Path,
        default=None,
        help="Optional path to write embedding coordinates as CSV.",
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
        default=3.25,
        help="Figure height in inches (default: 3.25).",
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
        default=8.0,
        help="Scatter marker size ('s') for random points (default: 8).",
    )
    parser.add_argument(
        "--random-alpha",
        type=float,
        default=0.25,
        help="Alpha transparency for random points (default: 0.25).",
    )
    parser.add_argument(
        "--special-marker-size",
        type=float,
        default=26.0,
        help="Scatter marker size ('s') for extremes/categorical (default: 26).",
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
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _import_lr_preprocess():
    lr_dir = (REPO_ROOT / "LR").resolve()
    if str(lr_dir) not in sys.path:
        sys.path.insert(0, str(lr_dir))
    import preprocess as lr_preprocess  # type: ignore

    return lr_preprocess


def _import_lr_train():
    lr_dir = (REPO_ROOT / "LR").resolve()
    if str(lr_dir) not in sys.path:
        sys.path.insert(0, str(lr_dir))
    import train as lr_train  # type: ignore

    return lr_train


def _engineer_features(raw_csv: Path) -> pd.DataFrame:
    lr_preprocess = _import_lr_preprocess()
    df = lr_preprocess.create_feature_dataframe(str(raw_csv))
    df = df.rename(columns=lambda c: str(c).strip())
    return df


def _numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        col = str(col).strip()
        if col == "Input List":
            continue
        if col in TARGET_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        cols.append(col)
    return cols


def _stack_datasets(
    datasets: Sequence[Tuple[str, Path]],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for label, csv_path in datasets:
        df = _engineer_features(csv_path)
        df["dataset"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _prep_features(
    df_all: pd.DataFrame,
    *,
    feature_cols: List[str],
    fit_mask: np.ndarray,
    seed: int,
    pca_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lr_train = _import_lr_train()

    pre = Pipeline(
        steps=[
            ("prep", lr_train.build_preprocessor(feature_cols)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    X_all = df_all.loc[:, feature_cols]
    X_fit = df_all.loc[fit_mask, feature_cols]

    X_fit_t = pre.fit_transform(X_fit)
    X_all_t = pre.transform(X_all)

    if pca_components <= 0:
        return X_fit_t, X_all_t

    n_components = min(int(pca_components), X_fit_t.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    X_fit_pca = pca.fit_transform(X_fit_t)
    X_all_pca = pca.transform(X_all_t)
    return X_fit_pca, X_all_pca


def compute_umap_embedding(
    X_fit: np.ndarray,
    X_all: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    try:
        import umap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: umap-learn. Install in your venv with:\n"
            "  ./venv/bin/pip install umap-learn\n"
            "Then rerun this script using ./venv/bin/python."
        ) from exc

    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(seed),
        n_components=2,
    )
    reducer.fit(X_fit)
    return reducer.transform(X_all)


def _dataset_styles(
    random_alpha: float,
    random_marker_size: float,
    special_marker_size: float,
) -> Dict[str, Dict[str, object]]:
    return {
        "Random": {
            "marker": "o",
            "color": "#5F5F5F",
            "alpha": random_alpha,
            "s": random_marker_size,
            "edgecolors": "none",
            "linewidths": 0.0,
        },
        "Categorical": {
            "marker": "^",
            "facecolor": "#2CA02C",
            "edgecolor": "black",
            "alpha": 1.0,
            "s": special_marker_size,
            "linewidths": 0.6,
        },
        "Optimized": {
            "marker": "s",
            "facecolor": "#D62728",
            "edgecolor": "black",
            "alpha": 1.0,
            "s": special_marker_size,
            "linewidths": 0.6,
        },
    }


def plot_umap_overlay(
    df: pd.DataFrame,
    *,
    outdir: Path,
    outfile: str,
    figwidth: float,
    figheight: float,
    dpi: int,
    legend: str,
    legend_cols: int,
    random_alpha: float,
    random_marker_size: float,
    special_marker_size: float,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / outfile

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

    styles = _dataset_styles(
        random_alpha=random_alpha,
        random_marker_size=random_marker_size,
        special_marker_size=special_marker_size,
    )

    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    # Plot random first as a background cloud.
    order = ["Random", "Categorical", "Optimized"]
    for label in order:
        mask = df["dataset"].astype(str) == label
        if not mask.any():
            continue
        style = styles.get(label, {})
        if label == "Random":
            ax.scatter(df.loc[mask, "umap_1"], df.loc[mask, "umap_2"], **style)
        else:
            ax.scatter(
                df.loc[mask, "umap_1"],
                df.loc[mask, "umap_2"],
                marker=style["marker"],
                s=style["s"],
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidths=style["linewidths"],
                alpha=style["alpha"],
                zorder=3,
            )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if legend != "none":
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label="Random",
                markerfacecolor=styles["Random"]["color"],
                markeredgecolor="none",
                markersize=5,
                alpha=0.9,
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="",
                label="Categorical",
                markerfacecolor=styles["Categorical"]["facecolor"],
                markeredgecolor="black",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                label="Optimized",
                markerfacecolor=styles["Optimized"]["facecolor"],
                markeredgecolor="black",
                markersize=6,
            ),
        ]

        if legend == "inside":
            ax.legend(handles=handles, loc="best", frameon=False, fontsize=9)
        else:
            bbox = (0.5, -0.18) if legend == "below" else (0.5, 1.15)
            ax.legend(
                handles=handles,
                loc="upper center" if legend == "below" else "lower center",
                bbox_to_anchor=bbox,
                frameon=False,
                ncol=int(legend_cols),
                fontsize=9,
                columnspacing=1.2,
                handletextpad=0.4,
            )
            fig.subplots_adjust(bottom=0.27 if legend == "below" else 0.12, top=0.9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return outpath


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    datasets = [
        ("Random", resolve_repo_path(args.random_csv)),
        ("Categorical", resolve_repo_path(args.categorical_csv)),
        ("Optimized", resolve_repo_path(args.extreme_1_csv)),
        ("Optimized", resolve_repo_path(args.extreme_2_csv)),
    ]
    df_all = _stack_datasets(datasets)

    feature_cols = _numeric_feature_columns(df_all)
    if not args.include_redundant:
        for col in ("max_length", "min_block_size"):
            if col in feature_cols:
                feature_cols.remove(col)

    if not feature_cols:
        raise SystemExit("No numeric engineered feature columns found to embed.")

    fit_mask = df_all["dataset"].astype(str).eq("Random").to_numpy()
    if args.fit_on == "all":
        fit_mask = np.ones(len(df_all), dtype=bool)

    X_fit, X_all = _prep_features(
        df_all,
        feature_cols=feature_cols,
        fit_mask=fit_mask,
        seed=int(args.seed),
        pca_components=int(args.pca_components),
    )

    embedding = compute_umap_embedding(
        X_fit,
        X_all,
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=str(args.metric),
        seed=int(args.seed),
    )

    df_plot = df_all.copy()
    df_plot["umap_1"] = embedding[:, 0]
    df_plot["umap_2"] = embedding[:, 1]

    outdir = resolve_repo_path(args.outdir)
    outpath = plot_umap_overlay(
        df_plot,
        outdir=outdir,
        outfile=str(args.outfile),
        figwidth=float(args.figwidth),
        figheight=float(args.figheight),
        dpi=int(args.dpi),
        legend=str(args.legend),
        legend_cols=int(args.legend_cols),
        random_alpha=float(args.random_alpha),
        random_marker_size=float(args.random_marker_size),
        special_marker_size=float(args.special_marker_size),
    )
    print(f"Saved UMAP overlay figure to: {outpath}")

    if args.save_embedding_csv is not None:
        csv_path = resolve_repo_path(args.save_embedding_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keep_cols = ["dataset", "umap_1", "umap_2"]
        if "Input List" in df_plot.columns:
            keep_cols.insert(0, "Input List")
        df_plot.loc[:, keep_cols].to_csv(csv_path, index=False)
        print(f"Saved embedding coordinates to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
