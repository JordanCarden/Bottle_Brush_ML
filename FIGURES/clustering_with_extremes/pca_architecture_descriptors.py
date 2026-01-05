#!/usr/bin/env python3
"""PCA projection of LR/MLP engineered architecture descriptors (with overlays).

This is a companion to `umap_architecture_descriptors.py` that uses PCA instead
of UMAP, and (optionally) summarizes "out-of-distribution-ness" via nearest-
neighbor distance to the Random set in the same standardized PCA space.

By default:
  - Features are engineered from the same "Input List" strings used elsewhere.
  - Preprocessing follows the LR pipeline's monotonic transforms + scaling.
  - PCA is fit on the Random set and used to transform the other sets.
  - A two-panel figure is produced:
      (left) PC1 vs PC2 overlay scatter
      (right) boxplot of nearest-neighbor distance to Random (PCA space)

Example:
  ./venv/bin/python FIGURES/clustering_with_extremes/pca_architecture_descriptors.py
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
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

TARGET_COLUMNS = {"Area AVG", "RG AVG", "RDF Peak"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA projection of engineered architecture descriptors."
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
        default=REPO_ROOT / "test_set.csv",
        help="Categorical/structured raw CSV (default: test_set.csv).",
    )
    parser.add_argument(
        "--extreme-1-csv",
        type=Path,
        default=REPO_ROOT / "extreme_test_set_1.csv",
        help="Extreme 1 raw CSV (default: extreme_test_set_1.csv).",
    )
    parser.add_argument(
        "--extreme-2-csv",
        type=Path,
        default=REPO_ROOT / "extreme_test_set_2.csv",
        help="Extreme 2 raw CSV (default: extreme_test_set_2.csv).",
    )
    parser.add_argument(
        "--fit-on",
        choices=("random", "all"),
        default="random",
        help="Which samples define the preprocessing + PCA fit (default: random).",
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
        help="Number of PCA components for distance calculations (min 2; default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA (default: 42).",
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
        default="pca_architecture_descriptors.png",
        help="Output filename (default: pca_architecture_descriptors.png).",
    )
    parser.add_argument(
        "--save-embedding-csv",
        type=Path,
        default=None,
        help="Optional path to write PC1/PC2 and NN distance as CSV.",
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
        default=300,
        help="DPI for PNG output (default: 300).",
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


def _prep_and_pca(
    df_all: pd.DataFrame,
    *,
    feature_cols: List[str],
    fit_mask: np.ndarray,
    seed: int,
    pca_components: int,
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    lr_train = _import_lr_train()

    pre = Pipeline(
        steps=[
            ("prep", lr_train.build_preprocessor(feature_cols)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    X_fit = df_all.loc[fit_mask, feature_cols]
    X_all = df_all.loc[:, feature_cols]

    X_fit_t = pre.fit_transform(X_fit)
    X_all_t = pre.transform(X_all)

    n_components = max(2, min(int(pca_components), X_fit_t.shape[1]))
    pca = PCA(n_components=n_components, random_state=int(seed))
    X_fit_pca = pca.fit_transform(X_fit_t)
    X_all_pca = pca.transform(X_all_t)
    return pca, X_fit_pca, X_all_pca


def _nn_distance_to_random(
    df_all: pd.DataFrame,
    X_all_pca: np.ndarray,
    *,
    random_label: str = "Random",
) -> np.ndarray:
    random_mask = df_all["dataset"].astype(str).eq(random_label).to_numpy()
    X_random = X_all_pca[random_mask]
    if X_random.shape[0] < 2:
        raise ValueError("Random dataset must contain at least 2 samples for NN distances.")

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_random)
    dists, _ = nn.kneighbors(X_all_pca)
    # For random points, neighbor 0 is itself (distance 0); use neighbor 1 instead.
    distances = dists[:, 0].copy()
    distances[random_mask] = dists[random_mask, 1]
    return distances


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
        "Extreme 1": {
            "marker": "s",
            "facecolor": "#FF7F0E",
            "edgecolor": "black",
            "alpha": 1.0,
            "s": special_marker_size,
            "linewidths": 0.6,
        },
        "Extreme 2": {
            "marker": "s",
            "facecolor": "#D62728",
            "edgecolor": "black",
            "alpha": 1.0,
            "s": special_marker_size,
            "linewidths": 0.6,
        },
    }


def plot_pca_with_nn_panel(
    df: pd.DataFrame,
    *,
    explained_var: Tuple[float, float],
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
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
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

    fig, (ax_scatter, ax_box) = plt.subplots(
        1,
        2,
        figsize=(figwidth, figheight),
        gridspec_kw={"width_ratios": [2.2, 1.2]},
    )

    # Scatter panel.
    order = ["Random", "Categorical", "Extreme 1", "Extreme 2"]
    for label in order:
        mask = df["dataset"].astype(str) == label
        if not mask.any():
            continue
        style = styles.get(label, {})
        if label == "Random":
            ax_scatter.scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"], **style)
        else:
            ax_scatter.scatter(
                df.loc[mask, "pc1"],
                df.loc[mask, "pc2"],
                marker=style["marker"],
                s=style["s"],
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidths=style["linewidths"],
                alpha=style["alpha"],
                zorder=3,
            )

    ax_scatter.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    ax_scatter.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    for spine in ax_scatter.spines.values():
        spine.set_linewidth(1.2)

    # NN-distance boxplot panel.
    data = []
    labels = []
    for label in order:
        mask = df["dataset"].astype(str) == label
        if not mask.any():
            continue
        data.append(df.loc[mask, "nn_dist"].to_numpy(dtype=float))
        labels.append(label)

    boxplot_kwargs = dict(
        patch_artist=True,
        widths=0.65,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    # Matplotlib renamed 'labels' -> 'tick_labels' in 3.9; keep compatibility.
    try:
        box = ax_box.boxplot(data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:  # pragma: no cover
        box = ax_box.boxplot(data, labels=labels, **boxplot_kwargs)
    facecolors = [styles[l].get("facecolor", styles[l].get("color", "#5F5F5F")) for l in labels]
    alphas = [0.35 if l == "Random" else 0.9 for l in labels]
    for patch, fc, a in zip(box["boxes"], facecolors, alphas, strict=False):
        patch.set_facecolor(fc)
        patch.set_alpha(a)
        patch.set_edgecolor("black")

    ax_box.set_ylabel("NN distance to Random\n(PCA space)")
    ax_box.tick_params(axis="x", rotation=45)
    for spine in ax_box.spines.values():
        spine.set_linewidth(1.2)

    # Legend handling.
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
                label="Extreme 1",
                markerfacecolor=styles["Extreme 1"]["facecolor"],
                markeredgecolor="black",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                label="Extreme 2",
                markerfacecolor=styles["Extreme 2"]["facecolor"],
                markeredgecolor="black",
                markersize=6,
            ),
        ]

        if legend == "inside":
            ax_scatter.legend(handles=handles, loc="best", frameon=False, fontsize=8)
        else:
            loc = "lower center" if legend == "above" else "upper center"
            bbox = (0.5, 1.12) if legend == "above" else (0.5, -0.18)
            fig.legend(
                handles=handles,
                loc=loc,
                bbox_to_anchor=bbox,
                frameon=False,
                ncol=int(legend_cols),
                fontsize=8,
                columnspacing=1.2,
                handletextpad=0.4,
            )

    if legend == "below":
        fig.subplots_adjust(bottom=0.3)
    elif legend == "above":
        fig.subplots_adjust(top=0.83)

    fig.tight_layout()
    fig.savefig(outpath, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return outpath


def _print_nn_summary(df: pd.DataFrame) -> None:
    order = ["Random", "Categorical", "Extreme 1", "Extreme 2"]
    print("Nearest-neighbor distance to Random (PCA space):")
    for label in order:
        mask = df["dataset"].astype(str).eq(label)
        if not mask.any():
            continue
        values = df.loc[mask, "nn_dist"].to_numpy(dtype=float)
        print(
            f"{label:11s} n={values.size:4d} "
            f"median={np.median(values):.3f} p90={np.percentile(values,90):.3f} "
            f"max={np.max(values):.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    datasets = [
        ("Random", resolve_repo_path(args.random_csv)),
        ("Categorical", resolve_repo_path(args.categorical_csv)),
        ("Extreme 1", resolve_repo_path(args.extreme_1_csv)),
        ("Extreme 2", resolve_repo_path(args.extreme_2_csv)),
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

    pca, _X_fit_pca, X_all_pca = _prep_and_pca(
        df_all,
        feature_cols=feature_cols,
        fit_mask=fit_mask,
        seed=int(args.seed),
        pca_components=int(args.pca_components),
    )

    nn_dist = _nn_distance_to_random(df_all, X_all_pca, random_label="Random")

    df_plot = df_all.copy()
    df_plot["pc1"] = X_all_pca[:, 0]
    df_plot["pc2"] = X_all_pca[:, 1]
    df_plot["nn_dist"] = nn_dist

    _print_nn_summary(df_plot)

    explained = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))

    outdir = resolve_repo_path(args.outdir)
    outpath = plot_pca_with_nn_panel(
        df_plot,
        explained_var=explained,
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
    print(f"Saved PCA overlay figure to: {outpath}")

    if args.save_embedding_csv is not None:
        csv_path = resolve_repo_path(args.save_embedding_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keep_cols = ["dataset", "pc1", "pc2", "nn_dist"]
        if "Input List" in df_plot.columns:
            keep_cols.insert(0, "Input List")
        df_plot.loc[:, keep_cols].to_csv(csv_path, index=False)
        print(f"Saved PCA coordinates to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
