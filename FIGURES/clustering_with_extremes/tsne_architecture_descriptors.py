#!/usr/bin/env python3
"""t-SNE projection of LR/MLP engineered architecture descriptors (with overlays).

t-SNE is non-parametric (no reliable out-of-sample transform), so this script
fits t-SNE on the concatenated dataset (Random + Categorical + Extreme sets)
and then overlays points by dataset label. Extreme samples from
extreme_test_set_1/2 are combined into a single "Extreme" group.

The input representation is identical to the LR/MLP engineered descriptor set:
we engineer features from "Input List", apply the same monotonic transforms as
the LR pipeline, impute, standardize, optionally PCA-reduce, then run t-SNE.

Example:
  ./venv/bin/python FIGURES/clustering_with_extremes/tsne_architecture_descriptors.py
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
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

TARGET_COLUMNS = {"Area AVG", "RG AVG", "RDF Peak"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="t-SNE projection of engineered architecture descriptors."
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
        help="Extreme set (part 1) raw CSV (default: data/extreme_test_set_1.csv).",
    )
    parser.add_argument(
        "--extreme-2-csv",
        type=Path,
        default=REPO_ROOT / "data" / "extreme_test_set_2.csv",
        help="Extreme set (part 2) raw CSV (default: data/extreme_test_set_2.csv).",
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
        help="PCA components before t-SNE (0 disables PCA; default: 20).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30).",
    )
    parser.add_argument(
        "--learning-rate",
        type=str,
        default="auto",
        help="t-SNE learning_rate (float or 'auto'; default: auto).",
    )
    parser.add_argument(
        "--early-exaggeration",
        type=float,
        default=12.0,
        help="t-SNE early exaggeration (default: 12).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="t-SNE max iterations (default: 2000).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="t-SNE metric (default: euclidean).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA/t-SNE (default: 42).",
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
        default="tsne_architecture_descriptors.png",
        help="Output filename (default: tsne_architecture_descriptors.png).",
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
        "Extreme": {
            "marker": "s",
            "facecolor": "#D62728",
            "edgecolor": "black",
            "alpha": 1.0,
            "s": special_marker_size,
            "linewidths": 0.6,
        },
    }


def plot_tsne_overlay(
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

    order = ["Random", "Categorical", "Extreme"]
    for label in order:
        mask = df["dataset"].astype(str) == label
        if not mask.any():
            continue
        style = styles.get(label, {})
        if label == "Random":
            ax.scatter(df.loc[mask, "tsne_1"], df.loc[mask, "tsne_2"], **style)
        else:
            ax.scatter(
                df.loc[mask, "tsne_1"],
                df.loc[mask, "tsne_2"],
                marker=style["marker"],
                s=style["s"],
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidths=style["linewidths"],
                alpha=style["alpha"],
                zorder=3,
            )

    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

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
                label="Extreme",
                markerfacecolor=styles["Extreme"]["facecolor"],
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


def _parse_learning_rate(text: str) -> float | str:
    value = str(text).strip().lower()
    if value == "auto":
        return "auto"
    return float(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    datasets = [
        ("Random", resolve_repo_path(args.random_csv)),
        ("Categorical", resolve_repo_path(args.categorical_csv)),
        ("Extreme", resolve_repo_path(args.extreme_1_csv)),
        ("Extreme", resolve_repo_path(args.extreme_2_csv)),
    ]
    df_all = _stack_datasets(datasets)

    feature_cols = _numeric_feature_columns(df_all)
    if not args.include_redundant:
        for col in ("max_length", "min_block_size"):
            if col in feature_cols:
                feature_cols.remove(col)

    if not feature_cols:
        raise SystemExit("No numeric engineered feature columns found to embed.")

    lr_train = _import_lr_train()
    pre = Pipeline(
        steps=[
            ("prep", lr_train.build_preprocessor(feature_cols)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    X = pre.fit_transform(df_all.loc[:, feature_cols])

    if int(args.pca_components) > 0:
        n_components = min(int(args.pca_components), X.shape[1])
        pca = PCA(n_components=n_components, random_state=int(args.seed))
        X = pca.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=float(args.perplexity),
        early_exaggeration=float(args.early_exaggeration),
        learning_rate=_parse_learning_rate(args.learning_rate),
        max_iter=int(args.max_iter),
        metric=str(args.metric),
        init="pca",
        random_state=int(args.seed),
    )
    embedding = tsne.fit_transform(X)

    df_plot = df_all.copy()
    df_plot["tsne_1"] = embedding[:, 0]
    df_plot["tsne_2"] = embedding[:, 1]

    outdir = resolve_repo_path(args.outdir)
    outpath = plot_tsne_overlay(
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
    print(f"Saved t-SNE overlay figure to: {outpath}")

    if args.save_embedding_csv is not None:
        csv_path = resolve_repo_path(args.save_embedding_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keep_cols = ["dataset", "tsne_1", "tsne_2"]
        if "Input List" in df_plot.columns:
            keep_cols.insert(0, "Input List")
        df_plot.loc[:, keep_cols].to_csv(csv_path, index=False)
        print(f"Saved embedding coordinates to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
