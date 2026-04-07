#!/usr/bin/env python3
"""Combine per-model performance summaries into a single comparison table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]


TARGETS: Tuple[Tuple[str, str], ...] = (
    ("Area AVG", "area"),
    ("RG AVG", "rg"),
    ("RDF Peak", "rdf"),
)
TARGET_METRICS: Tuple[str, ...] = ("mae", "rmse", "r2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-model CV performance summaries with optional RDF filtering."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(REPO_ROOT),
        help="Repository root to search for per-model results directories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "cv_performance.csv"),
        help="Path (relative or absolute) for the combined summary.",
    )
    parser.add_argument(
        "--include-rdf-zeros",
        action="store_true",
        help="Include samples with zero ground-truth RDF values when computing metrics.",
    )
    return parser.parse_args()


def iter_model_dirs(root: Path) -> Iterable[Path]:
    """Yield each child directory that could contain model results."""
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            yield child


def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Return MAE, RMSE, R2 for the provided value lists."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan}

    n = len(y_true)
    diff = [yp - yt for yt, yp in zip(y_true, y_pred)]
    mae = sum(abs(d) for d in diff) / n
    rmse = math.sqrt(sum(d * d for d in diff) / n)

    mean_true = sum(y_true) / n
    ss_tot = sum((yt - mean_true) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    if math.isclose(ss_tot, 0.0, abs_tol=1e-12):
        r2 = 1.0 if math.isclose(ss_res, 0.0, abs_tol=1e-12) else math.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def load_metrics_from_predictions(
    results_dir: Path,
    include_rdf_zeros: bool,
) -> Optional[Tuple[Dict[str, float], List[str]]]:
    """Compute metrics from stored CV prediction CSVs if available."""
    metrics: Dict[str, float] = {}
    target_order: List[str] = []

    for display_name, alias in TARGETS:
        csv_path = results_dir / f"cv_predictions_{alias}.csv"
        if not csv_path.is_file():
            return None

        y_true: List[float] = []
        y_pred: List[float] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "y_true" not in reader.fieldnames or "y_pred" not in reader.fieldnames:
                raise ValueError(f"{csv_path} is missing required columns 'y_true' and/or 'y_pred'.")
            for row in reader:
                try:
                    yt = float(row["y_true"])
                    yp = float(row["y_pred"])
                except (TypeError, ValueError):
                    continue
                if display_name == "RDF Peak" and not include_rdf_zeros and math.isclose(yt, 0.0, abs_tol=1e-12):
                    continue
                y_true.append(yt)
                y_pred.append(yp)

        target_order.append(display_name)
        metrics[f"{display_name} Count"] = float(len(y_true))
        computed = compute_regression_metrics(y_true, y_pred)
        for metric_name, value in computed.items():
            metrics[f"{display_name} {metric_name.upper()}"] = value

    return metrics, target_order


def read_summary(summary_path: Path) -> Tuple[Dict[str, float], List[str]]:
    """Read a per-model summary CSV and return metric mapping + ordered targets."""
    metrics: Dict[str, float] = {}
    target_order: List[str] = []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = row.get("target")
            if not target:
                continue
            target_order.append(target)
            for metric in TARGET_METRICS:
                key = f"{target} {metric.upper()}"
                value = row.get(metric)
                try:
                    metrics[key] = float(value) if value is not None else math.nan
                except (TypeError, ValueError):
                    metrics[key] = math.nan
    return metrics, target_order


def compute_means(row: Dict[str, float], targets: Iterable[str]) -> Dict[str, float]:
    """Compute mean per-metric values across targets for the given row."""
    means: Dict[str, float] = {}
    for metric in TARGET_METRICS:
        values: List[float] = []
        for target in targets:
            value = row.get(f"{target} {metric.upper()}")
            if value is None or math.isnan(value):
                continue
            values.append(value)
        means[f"Mean {metric.upper()}"] = float(sum(values) / len(values)) if values else math.nan
    return means


def write_combined_summary(
    output_path: Path,
    rows: List[Dict[str, float]],
    targets: List[str],
    models_order: List[str],
) -> None:
    """Write the combined CSV with consistent column ordering."""
    columns: List[str] = ["model"]
    for target in targets:
        for metric in TARGET_METRICS:
            columns.append(f"{target} {metric.upper()}")
    for metric in TARGET_METRICS:
        columns.append(f"Mean {metric.upper()}")

    # Append any additional columns (e.g., counts) in sorted order for consistency.
    extra_columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns and key != "model":
                if key not in extra_columns:
                    extra_columns.append(key)
    columns.extend(sorted(extra_columns))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model in models_order:
            row = next((entry for entry in rows if entry["model"] == model), None)
            if row is None:
                continue
            writer.writerow({column: row.get(column, math.nan) for column in columns})


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    combined_rows: List[Dict[str, float]] = []
    all_targets: List[str] = []
    models_in_order: List[str] = []

    for model_dir in iter_model_dirs(root):
        results_dir = model_dir / "results"
        metrics: Dict[str, float] = {}
        target_order: List[str] = []

        prediction_metrics = None
        if results_dir.is_dir():
            prediction_metrics = load_metrics_from_predictions(results_dir, args.include_rdf_zeros)

        if prediction_metrics is not None:
            metrics, target_order = prediction_metrics
        else:
            summary_path = results_dir / "model_performance_summary.csv"
            if not summary_path.is_file():
                continue
            metrics, target_order = read_summary(summary_path)

        if not metrics:
            continue

        if not all_targets:
            all_targets = target_order
        else:
            for target in target_order:
                if target not in all_targets:
                    all_targets.append(target)

        row: Dict[str, float] = {"model": model_dir.name}
        row.update(metrics)
        row.update(compute_means(row, all_targets))
        combined_rows.append(row)
        models_in_order.append(model_dir.name)

    if not combined_rows:
        raise SystemExit("No per-model summaries found. Run individual training scripts first.")

    write_combined_summary(output_path, combined_rows, all_targets, models_in_order)
    print(f"Wrote combined summary to {output_path}")


if __name__ == "__main__":
    main()
