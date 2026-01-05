"""Linear regression pipeline for the polymer dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess import shifted_log1p
from utils.splits import load_splits


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and R2 for the provided arrays."""
    if y_true.size == 0 or y_pred.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        r2 = float("nan")
    else:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        r2 = 1 - ss_res / ss_tot

    return {"rmse": rmse, "mae": mae, "r2": float(r2)}


def write_cv_predictions(
    predictions: List[Dict[str, float]],
    results_dir: Path,
    target_alias: str,
) -> Optional[Dict[str, float]]:
    """Persist CV predictions for a target and return aggregate metrics."""
    if not predictions:
        return None

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.sort(key=lambda row: (row["fold"], row["sample_index"]))

    csv_path = results_dir / f"cv_predictions_{target_alias}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fold", "epoch", "sample_index", "y_true", "y_pred"])
        for row in predictions:
            writer.writerow(
                [
                    int(row["fold"]),
                    int(row["epoch"]),
                    int(row["sample_index"]),
                    float(row["y_true"]),
                    float(row["y_pred"]),
                ]
            )

    y_true = np.array([row["y_true"] for row in predictions], dtype=float)
    y_pred = np.array([row["y_pred"] for row in predictions], dtype=float)
    metrics = compute_regression_metrics(y_true, y_pred)
    print(f"Saved CV predictions for {target_alias} to: {csv_path}")
    return metrics


def write_performance_summary(
    rows: List[Tuple[str, Dict[str, float]]],
    results_dir: Path,
) -> None:
    """Write aggregate metrics per target to a summary CSV."""
    if not rows:
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "model_performance_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["target", "rmse", "mae", "r2"])
        for target_name, metrics in rows:
            writer.writerow(
                [
                    target_name,
                    metrics.get("rmse", float("nan")),
                    metrics.get("mae", float("nan")),
                    metrics.get("r2", float("nan")),
                ]
            )
    print(f"Saved performance summary to: {summary_path}")

def build_preprocessor(columns: Iterable[str]) -> ColumnTransformer:
    """Builds the preprocessing transformer for the model."""
    orig_log = ["max_fft_value", "mayo_lewis"]
    orig_cube = ["harwoods_blockiness"]
    orig_square = ["sum_fft_value"]
    orig_yj1 = ["mean_block_size", "std_block_size"]
    orig_yj2 = ["min_charge", "max_charge"]

    log_cols = [c for c in orig_log if c in columns]
    cube_cols = [c for c in orig_cube if c in columns]
    square_cols = [c for c in orig_square if c in columns]
    yeoj_cols_size = [c for c in orig_yj1 if c in columns]
    yeoj_cols_charge = [c for c in orig_yj2 if c in columns]

    other_cols = [c for c in columns if c not in (log_cols + cube_cols + square_cols + yeoj_cols_size + yeoj_cols_charge)]

    transformers = []
    if log_cols:
        transformers.append(("log", FunctionTransformer(shifted_log1p, feature_names_out="one-to-one"), log_cols))
    if cube_cols:
        transformers.append(("cube", FunctionTransformer(np.cbrt, feature_names_out="one-to-one"), cube_cols))
    if square_cols:
        transformers.append(("square", FunctionTransformer(np.square, feature_names_out="one-to-one"), square_cols))
    if yeoj_cols_size:
        transformers.append(("yj1", PowerTransformer(method="yeo-johnson", standardize=False), yeoj_cols_size))
    if yeoj_cols_charge:
        transformers.append(("yj2", PowerTransformer(method="yeo-johnson", standardize=False), yeoj_cols_charge))
    transformers.append(("pass", "passthrough", other_cols))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def train(
    df: pd.DataFrame,
    target: str,
    model_path: Path,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    """Train a linear regression model, returning metrics for summary writing."""
    target_columns = {"Area AVG", "RG AVG", "RDF Peak", target}
    feature_columns = [
        col for col in df.columns if col not in target_columns and col != "Input List"
    ]

    X = df.loc[:, feature_columns]
    y = df[target]

    initial_drop = ["max_length", "min_block_size"]
    X = X.drop(columns=initial_drop, errors="ignore")

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    target_corr = X.join(y).corr()[y.name].abs()
    multi_drop = {i if target_corr[i] < target_corr[j] else j for i in upper.index for j in upper.columns if upper.loc[i, j] > 0.90}
    X = X.drop(columns=list(multi_drop), errors="ignore")

    feature_cols = list(X.columns)

    def make_pipeline() -> Pipeline:
        return Pipeline([
            ("prep", build_preprocessor(feature_cols)),
            ("scale", StandardScaler()),
            ("lr", LinearRegression()),
        ])

    if not splits:
        raise ValueError("Splits list must not be empty.")

    oof_predictions = np.zeros(len(X), dtype=float)
    fold_assignments = np.zeros(len(X), dtype=int)
    for fold_id, (train_idx, val_idx) in enumerate(splits, 1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        fold_pipeline = make_pipeline()
        fold_pipeline.fit(X_train, y_train)
        predictions = fold_pipeline.predict(X_val)
        oof_predictions[val_idx] = predictions
        fold_assignments[val_idx] = fold_id

    y_array = y.to_numpy(dtype=float)
    cv_metrics = compute_regression_metrics(y_array, oof_predictions)

    full_pipeline = make_pipeline()

    full_pipeline.fit(X, y)
    joblib.dump(full_pipeline, model_path)
    in_sample_pred = full_pipeline.predict(X)
    in_sample_metrics = compute_regression_metrics(y_array, in_sample_pred)

    cv_prediction_records = [
        {
            "fold": int(fold_assignments[idx]),
            "epoch": 0,
            "sample_index": int(idx),
            "y_true": float(target_value),
            "y_pred": float(oof_predictions[idx]),
        }
        for idx, target_value in enumerate(y_array)
    ]

    return {
        "target": target,
        "cv_metrics": cv_metrics,
        "final_fit_metrics": {
            "rmse": float(in_sample_metrics["rmse"]),
            "mae": float(in_sample_metrics["mae"]),
            "r2": float(in_sample_metrics["r2"]),
        },
        "columns_used": list(full_pipeline.feature_names_in_),
        "cv_predictions": cv_prediction_records,
    }


def train_all(
    df: pd.DataFrame,
    target_map: dict[str, str],
    models_dir: Path,
    results_dir: Path,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    """Trains models for multiple targets.

    Args:
        df: Feature dataframe.
        target_map: Mapping from target column name to simplified alias.
        models_dir: Directory for model artifacts.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    summary_rows: List[Tuple[str, Dict[str, float]]] = []
    for target, alias in target_map.items():
        metrics = train(
            df,
            target,
            models_dir / f"lr_{alias}.joblib",
            splits,
        )
        predictions = metrics.pop("cv_predictions", [])
        metrics_from_cv = write_cv_predictions(predictions, results_dir, alias)
        if metrics_from_cv:
            summary_rows.append((target, metrics_from_cv))
        summaries[alias] = metrics

    write_performance_summary(summary_rows, results_dir)
    return summaries


def main() -> None:
    """Entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Train linear regression models with shared defaults"
    )
    parser.add_argument(
        "--data",
        default="processed/ES_features.csv",
        help="Path to the feature CSV file.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["Area AVG", "RG AVG", "RDF Peak"],
        help="Target names to train",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds used for cross-validation.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "splits" / "es_kfold_5_seed42.json"),
        help="Path to a JSON file containing predefined CV splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling during cross-validation.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    splits = load_splits(args.splits)
    if args.folds and args.folds != len(splits):
        print(
            f"Warning: --folds={args.folds} ignored; using {len(splits)} folds from {args.splits}",
        )
    args.folds = len(splits)

    target_alias_map = {
        "Area AVG": "area",
        "RG AVG": "rg",
        "RDF Peak": "rdf",
    }
    for target in args.targets:
        if target not in target_alias_map:
            raise ValueError(
                f"Unsupported target '{target}'. Expected one of {list(target_alias_map)}."
            )

    target_map = {t: target_alias_map[t] for t in args.targets}

    models_dir = Path("models")
    results_dir = Path(__file__).resolve().parent / "results"
    summaries = train_all(df, target_map, models_dir, results_dir, splits)
    for target, metrics in summaries.items():
        cv_rmse = metrics["cv_metrics"].get("rmse", float("nan"))
        cv_mae = metrics["cv_metrics"].get("mae", float("nan"))
        cv_r2 = metrics["cv_metrics"].get("r2", float("nan"))
        final_rmse = metrics["final_fit_metrics"].get("rmse", float("nan"))
        final_mae = metrics["final_fit_metrics"].get("mae", float("nan"))
        final_r2 = metrics["final_fit_metrics"].get("r2", float("nan"))
        print(
            f"{target}: CV_RMSE={cv_rmse:.3f}, CV_MAE={cv_mae:.3f}, CV_R2={cv_r2:.3f}, "
            f"Train_RMSE={final_rmse:.3f}, Train_MAE={final_mae:.3f}, Train_R2={final_r2:.3f}"
        )


if __name__ == "__main__":
    main()
