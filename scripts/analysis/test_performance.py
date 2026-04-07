#!/usr/bin/env python3
"""Evaluate model predictions against the ground truth and summarize metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]

TARGET_COLUMNS: Tuple[str, ...] = ("Area AVG", "RG AVG", "RDF Peak")
DEFAULT_DATASET = REPO_ROOT / "data" / "test_set.csv"
DEFAULT_SUMMARY = SCRIPT_DIR / "test_performance.csv"
PREDICT_SCRIPTS: Tuple[Tuple[str, Path], ...] = (
    ("CNN", REPO_ROOT / "CNN" / "predict.py"),
    ("GAT", REPO_ROOT / "GAT" / "predict.py"),
    ("GCN", REPO_ROOT / "GCN" / "predict.py"),
    ("GIN", REPO_ROOT / "GIN" / "predict.py"),
    ("LR", REPO_ROOT / "LR" / "predict.py"),
    ("MLP", REPO_ROOT / "MLP" / "predict.py"),
)


def _find_prediction_files(base_dir: Path) -> Dict[str, Path]:
    """Return a mapping of model name to prediction CSV path."""
    prediction_files: Dict[str, Path] = {}
    for path in base_dir.glob("*/predictions.csv"):
        if path.is_file():
            prediction_files[path.parent.name] = path
    return prediction_files


def _run_predict_scripts(dataset_path: Path, *, skip: bool) -> None:
    """Execute each model's predict.py to regenerate predictions for the dataset."""
    if skip:
        print("Skipping automatic prediction generation (--skip-predict).")
        return

    for model_name, script_path in PREDICT_SCRIPTS:
        if not script_path.is_file():
            print(f"[WARN] Skipping {model_name}: missing predictor at {script_path}")
            continue

        out_path = (script_path.parent / "predictions.csv").resolve()
        cmd = [
            "python3",
            str(script_path.resolve()),
            "--csv",
            str(dataset_path),
            "--out",
            str(out_path),
        ]
        print(f"Running {model_name} predictor...")
        try:
            subprocess.run(cmd, check=True, cwd=script_path.parent)
        except FileNotFoundError as exc:
            raise SystemExit(f"Failed to run {model_name} predictor: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"{model_name} predictor exited with code {exc.returncode}"
            ) from exc


def _validate_columns(df: pd.DataFrame, df_name: str) -> None:
    """Ensure the dataframe has all required target columns."""
    missing = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing columns: {', '.join(missing)}")


def _select_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe containing only the target columns."""
    columns = [col for col in TARGET_COLUMNS if col in df.columns]
    return df.loc[:, columns]


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate model predictions for a dataset and summarize test performance."
        )
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_DATASET),
        help="Path to the dataset CSV containing Input List + target columns.",
    )
    parser.add_argument(
        "--include-rdf-zeros",
        action="store_true",
        help="Include rows with zero RDF Peak values when computing metrics.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Use existing predictions and skip running predict.py scripts.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_SUMMARY),
        help="Path to write the performance summary CSV.",
    )
    return parser.parse_args(list(argv))


def evaluate_model(
    model: str,
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    exclude_rdf_zeros: bool,
) -> Dict[str, float]:
    """Compute metrics for a single model and return a flattened record."""
    metrics: Dict[str, float] = {"model": model}
    mae_values: List[float] = []
    rmse_values: List[float] = []
    r2_values: List[float] = []

    if ground_truth.empty:
        for column in TARGET_COLUMNS:
            metrics[f"{column} MAE"] = float("nan")
            metrics[f"{column} RMSE"] = float("nan")
            metrics[f"{column} R2"] = float("nan")
        metrics["Mean MAE"] = float("nan")
        metrics["Mean RMSE"] = float("nan")
        metrics["Mean R2"] = float("nan")
        return metrics

    for column in TARGET_COLUMNS:
        y_true_series = pd.to_numeric(ground_truth[column], errors="coerce")
        y_pred_series = pd.to_numeric(predictions[column], errors="coerce")
        mask = np.isfinite(y_true_series) & np.isfinite(y_pred_series)

        if column == "RDF Peak" and exclude_rdf_zeros:
            mask = mask & (y_true_series.to_numpy() != 0)
            metrics["RDF Peak Count"] = int(mask.sum())

        if not mask.any():
            mae = float("nan")
            rmse = float("nan")
            r2 = float("nan")
            metrics[f"{column} MAE"] = mae
            metrics[f"{column} RMSE"] = rmse
            metrics[f"{column} R2"] = r2
            mae_values.append(mae)
            rmse_values.append(rmse)
            r2_values.append(r2)
            continue

        y_true = y_true_series.to_numpy()[mask]
        y_pred = y_pred_series.to_numpy()[mask]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
        ss_res = float(np.sum(np.square(y_true - y_pred)))
        ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
        if np.isclose(ss_tot, 0.0):
            r2 = 1.0 if np.isclose(ss_res, 0.0) else float("nan")
        else:
            r2 = 1.0 - ss_res / ss_tot

        metrics[f"{column} MAE"] = float(mae)
        metrics[f"{column} RMSE"] = float(rmse)
        metrics[f"{column} R2"] = float(r2)

        mae_values.append(mae)
        rmse_values.append(rmse)
        r2_values.append(r2)

    metrics["Mean MAE"] = float(np.nanmean(mae_values))
    metrics["Mean RMSE"] = float(np.nanmean(rmse_values))
    metrics["Mean R2"] = float(np.nanmean(r2_values))
    return metrics


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    dataset_path = Path(args.csv)
    if not dataset_path.is_file():
        print(f"Dataset file not found at {dataset_path}", file=sys.stderr)
        return 1
    dataset_path = dataset_path.resolve()

    _run_predict_scripts(dataset_path, skip=args.skip_predict)
    print(f"Evaluating predictions against {dataset_path}")

    ground_truth = pd.read_csv(dataset_path)
    ground_truth = ground_truth.rename(columns=lambda c: c.strip())
    _validate_columns(ground_truth, "Ground truth")
    exclude_rdf_zeros = not args.include_rdf_zeros

    prediction_files = _find_prediction_files(REPO_ROOT)
    if not prediction_files:
        print("No prediction files found.", file=sys.stderr)
        return 1

    records: List[Dict[str, float]] = []
    for model, pred_path in sorted(prediction_files.items()):
        predictions = pd.read_csv(pred_path)
        predictions = predictions.rename(columns=lambda c: c.strip())
        predictions = _select_targets(predictions)
        _validate_columns(predictions, f"{model} predictions")

        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Row count mismatch for {model}: "
                f"{len(predictions)} predictions vs {len(ground_truth)} ground truth."
            )

        # Drop rows that have missing target values in either ground truth or predictions.
        target_cols = list(TARGET_COLUMNS)
        valid_mask = ground_truth[target_cols].notna().all(axis=1) & predictions[target_cols].notna().all(axis=1)
        dropped = int((~valid_mask).sum())
        if dropped:
            print(f"[WARN] {model}: skipping {dropped} rows with NaN targets.")
        gt_filtered = ground_truth.loc[valid_mask].reset_index(drop=True)
        pred_filtered = predictions.loc[valid_mask].reset_index(drop=True)

        if gt_filtered.empty:
            print(f"[WARN] {model}: no rows left after filtering; metrics set to NaN.")
            records.append(evaluate_model(model, pred_filtered, gt_filtered, exclude_rdf_zeros=exclude_rdf_zeros))
            continue

        records.append(
            evaluate_model(
                model,
                pred_filtered,
                gt_filtered,
                exclude_rdf_zeros=exclude_rdf_zeros,
            )
        )

    summary_path = Path(args.out)
    summary = pd.DataFrame(records).sort_values("Mean RMSE")
    summary_path.write_text(summary.to_csv(index=False))
    print(f"Wrote summary metrics to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
