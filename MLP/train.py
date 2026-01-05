#!/usr/bin/env python3
"""Train MLP models on LR-engineered features using PyTorch."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LR_DIR = REPO_ROOT / "LR"
if str(LR_DIR) not in sys.path:
    sys.path.insert(0, str(LR_DIR))

from preprocess import shifted_log1p  # type: ignore[attr-defined]
from utils.splits import load_splits

TARGET_ALIASES = {"Area AVG": "area", "RG AVG": "rg", "RDF Peak": "rdf"}


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return RMSE, MAE, and R2 computed from the provided arrays."""
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_batch_size(n_samples: int, batch_size: int) -> int:
    if batch_size is None or batch_size <= 0:
        return n_samples
    return min(batch_size, n_samples)


# ---------------------------------------------------------------------------
# Feature preprocessing (mirrors LR pipeline)
# ---------------------------------------------------------------------------
def _build_preprocessor(columns: Iterable[str]) -> ColumnTransformer:
    log_cols = [c for c in ("max_fft_value", "mayo_lewis") if c in columns]
    cube_cols = [c for c in ("harwoods_blockiness",) if c in columns]
    square_cols = [c for c in ("sum_fft_value",) if c in columns]
    yeoj_cols_size = [c for c in ("mean_block_size", "std_block_size") if c in columns]
    yeoj_cols_charge = [c for c in ("min_charge", "max_charge") if c in columns]

    other_cols = [
        c
        for c in columns
        if c
        not in (
            log_cols + cube_cols + square_cols + yeoj_cols_size + yeoj_cols_charge
        )
    ]

    transformers = []
    if log_cols:
        transformers.append(
            (
                "log",
                FunctionTransformer(shifted_log1p, feature_names_out="one-to-one"),
                log_cols,
            )
        )
    if cube_cols:
        transformers.append(
            (
                "cube",
                FunctionTransformer(np.cbrt, feature_names_out="one-to-one"),
                cube_cols,
            )
        )
    if square_cols:
        transformers.append(
            (
                "square",
                FunctionTransformer(np.square, feature_names_out="one-to-one"),
                square_cols,
            )
        )
    if yeoj_cols_size:
        transformers.append(
            (
                "yj_size",
                PowerTransformer(method="yeo-johnson", standardize=False),
                yeoj_cols_size,
            )
        )
    if yeoj_cols_charge:
        transformers.append(
            (
                "yj_charge",
                PowerTransformer(method="yeo-johnson", standardize=False),
                yeoj_cols_charge,
            )
        )
    transformers.append(("passthrough", "passthrough", other_cols))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _select_base_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    target_columns = {"Area AVG", "RG AVG", "RDF Peak", target}
    feature_columns = [
        col for col in df.columns if col not in target_columns and col != "Input List"
    ]
    X = df.loc[:, feature_columns]
    initial_drop = ["max_length", "min_block_size"]
    return X.drop(columns=initial_drop, errors="ignore")


def _apply_collinearity_filter(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    target_corr = X.join(y).corr()[y.name].abs()
    drop = {
        i if target_corr[i] < target_corr[j] else j
        for i in upper.index
        for j in upper.columns
        if upper.loc[i, j] > 0.90
    }
    return X.drop(columns=list(drop), errors="ignore")


def prepare_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = _select_base_features(df, target)
    y = df[target]
    X = _apply_collinearity_filter(X, y)
    return X, y


def build_pipeline(columns: Sequence[str]) -> Pipeline:
    preprocessor = _build_preprocessor(columns)
    return Pipeline(
        [
            ("prep", preprocessor),
            ("scale", StandardScaler()),
        ]
    )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
    return total_loss / max(total, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    preds: list[float] = []
    gts: list[float] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            preds.extend(outputs.cpu().numpy().tolist())
            gts.extend(targets.cpu().numpy().tolist())

    if total == 0:
        return float("nan"), {}, np.array([]), np.array([])

    preds_arr = np.array(preds, dtype=float)
    gts_arr = np.array(gts, dtype=float)
    metrics = compute_regression_metrics(gts_arr, preds_arr)

    avg_loss = total_loss / total
    return avg_loss, metrics, preds_arr, gts_arr


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Cross-validation to estimate epoch count
# ---------------------------------------------------------------------------
def cross_validate_epochs(
    X: pd.DataFrame,
    y: pd.Series,
    columns: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    splits: list[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[int, Dict[str, float], List[Dict[str, float]]]:
    best_epochs: List[int] = []
    best_val_losses: List[float] = []
    best_val_rmses: List[float] = []
    all_predictions: List[Dict[str, float]] = []
    criterion = nn.MSELoss()

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\nFold {fold}/{args.folds}")
        set_seed(args.seed + fold)

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        pipeline = build_pipeline(columns)
        pipeline.fit(X_train)
        X_train_t = pipeline.transform(X_train)
        X_val_t = pipeline.transform(X_val)

        train_bs = resolve_batch_size(len(X_train_t), args.batch_size)
        val_bs = resolve_batch_size(len(X_val_t), args.batch_size)

        train_loader = make_dataloader(X_train_t, y_train.to_numpy(), train_bs, True)
        val_loader = make_dataloader(X_val_t, y_val.to_numpy(), val_bs, False)

        model = FeedForwardNet(
            input_dim=X_train_t.shape[1],
            hidden_sizes=args.hidden_sizes,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_epoch = 0
        best_val_loss = float("inf")
        best_val_rmse = float("inf")
        patience = args.patience
        epochs_without_improve = 0
        best_val_preds: Optional[np.ndarray] = None
        best_val_targets: Optional[np.ndarray] = None

        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics, val_preds, val_targets = evaluate(
                model, val_loader, criterion, device
            )
            val_rmse = val_metrics.get("rmse", float("nan"))

            should_log = (
                epoch == 1
                or epoch % 10 == 0
                or (patience > 0 and epochs_without_improve + 1 >= patience)
            )
            if should_log:
                train_rmse = float(np.sqrt(train_loss))
                print(
                    f"  Epoch {epoch:4d} | train_rmse={train_rmse:.6f} | "
                    f"val_rmse={val_rmse:.6f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_val_rmse = val_rmse
                best_val_preds = val_preds
                best_val_targets = val_targets
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if patience > 0 and epochs_without_improve >= patience:
                    break

        best_epochs.append(best_epoch)
        best_val_losses.append(float(best_val_loss))
        best_val_rmses.append(float(best_val_rmse))
        print(
            f"Best epoch for fold {fold}: {best_epoch} "
            f"(val_rmse={best_val_rmse:.6f})"
        )
        if best_val_preds is not None and best_val_targets is not None:
            for sample_idx, pred, target_val in zip(
                val_idx.tolist(), best_val_preds.tolist(), best_val_targets.tolist()
            ):
                all_predictions.append(
                    {
                        "fold": int(fold),
                        "epoch": int(best_epoch),
                        "sample_index": int(sample_idx),
                        "y_true": float(target_val),
                        "y_pred": float(pred),
                    }
                )

    optimal_epochs = max(1, int(round(float(np.mean(best_epochs))))) if best_epochs else args.max_epochs
    summary: Dict[str, float] = {
        "mean_val_loss": float(np.mean(best_val_losses)) if best_val_losses else float("inf"),
        "mean_val_rmse": float(np.mean(best_val_rmses)) if best_val_rmses else float("inf"),
    }
    print(f"\nSuggested epoch count: {optimal_epochs}")
    return optimal_epochs, summary, all_predictions


def run_lr_weight_decay_search(
    X: pd.DataFrame,
    y: pd.Series,
    columns: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    splits: list[Tuple[np.ndarray, np.ndarray]],
    target_alias: str,
) -> Tuple[Dict[str, float], int, Dict[str, float], List[Dict[str, float]]]:
    if not args.lr_grid or not args.wd_grid:
        raise ValueError("Both --lr_grid and --wd_grid must provide at least one value when using --tune.")

    combos = [(lr, wd) for lr in args.lr_grid for wd in args.wd_grid]
    total_trials = len(combos)
    if total_trials == 0:
        raise ValueError("No hyperparameter combinations generated for tuning.")

    original_lr = args.lr
    original_wd = args.weight_decay

    trials: List[Dict[str, float]] = []
    best_config: Optional[Dict[str, float]] = None
    best_summary: Dict[str, float] = {}
    best_epochs = args.max_epochs
    best_metric = float("inf")
    best_predictions: List[Dict[str, float]] = []

    try:
        for idx, (lr, wd) in enumerate(combos, 1):
            print(f"\n=== Tuning trial {idx}/{total_trials} ===")
            print(f"Learning rate: {lr:.6g}, Weight decay: {wd:.6g}")
            args.lr = lr
            args.weight_decay = wd
            epochs, summary, predictions = cross_validate_epochs(
                X, y, columns, args, device, splits
            )
            record = {
                "trial": idx,
                "lr": float(lr),
                "weight_decay": float(wd),
                "suggested_epochs": int(epochs),
                "mean_val_loss": summary.get("mean_val_loss", float("inf")),
                "mean_val_rmse": summary.get("mean_val_rmse", float("inf")),
            }
            trials.append(record)

            metric = summary.get("mean_val_loss", float("inf"))
            if metric < best_metric:
                best_metric = metric
                best_config = {"lr": float(lr), "weight_decay": float(wd)}
                best_summary = summary
                best_epochs = epochs
                best_predictions = predictions
    finally:
        args.lr = original_lr
        args.weight_decay = original_wd

    if best_config is None:
        raise RuntimeError("Tuning did not evaluate any valid hyperparameter combinations.")

    if args.record_trials:
        record_path = Path(args.record_trials)
        if record_path.is_dir() or record_path.suffix == "":
            record_path = record_path / f"{target_alias}_lr_wd_trials.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("w", encoding="utf-8") as f:
            json.dump(trials, f, indent=2)
        print(f"Saved tuning trials to: {record_path}")

    return best_config, best_epochs, best_summary, best_predictions


# ---------------------------------------------------------------------------
# Final training on full dataset
# ---------------------------------------------------------------------------
def train_full_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    columns: Sequence[str],
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FeedForwardNet, Pipeline, dict[str, float]]:
    pipeline = build_pipeline(columns)
    pipeline.fit(X)
    X_t = pipeline.transform(X)

    batch_size = resolve_batch_size(len(X_t), args.batch_size)
    loader = make_dataloader(X_t, y.to_numpy(), batch_size, True)

    model = FeedForwardNet(
        input_dim=X_t.shape[1],
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loader, criterion, optimizer, device)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            train_rmse = float(np.sqrt(train_loss))
            print(
                f"Final training pass: epoch {epoch}/{epochs} "
                f"train_rmse={train_rmse:.6f}"
            )

    # Evaluate on the training set
    train_loss, metrics, _, _ = evaluate(model, loader, criterion, device)
    metrics["train_loss"] = train_loss
    return model, pipeline, metrics


def save_artifacts(
    model: FeedForwardNet,
    pipeline: Pipeline,
    columns: Sequence[str],
    alias: str,
    args: argparse.Namespace,
    model_metrics: dict[str, float],
) -> None:
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = models_dir / f"mlp_{alias}_prep.joblib"
    model_path = models_dir / f"mlp_{alias}.pt"
    meta_path = models_dir / f"mlp_{alias}.json"

    joblib.dump(pipeline, pipeline_path)
    torch.save(model.state_dict(), model_path)

    first_linear = next((m for m in model.net if isinstance(m, nn.Linear)), None)
    input_dim = int(first_linear.in_features) if first_linear is not None else None

    metrics_clean: dict[str, float | int | str | None] = {}
    for key, value in model_metrics.items():
        if isinstance(value, (int, np.integer)):
            metrics_clean[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            metrics_clean[key] = float(value)
        else:
            metrics_clean[key] = value

    metadata = {
        "columns": list(columns),
        "hidden_sizes": list(args.hidden_sizes),
        "dropout": float(args.dropout),
        "pipeline": pipeline_path.name,
        "model": model_path.name,
        "input_dim": input_dim,
        "epochs_trained": model_metrics.get("epochs_trained"),
        "target": alias,
        "metrics": metrics_clean,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to: {model_path}")
    print(f"Saved preprocessor to: {pipeline_path}")


def run_training(args: argparse.Namespace) -> None:
    if args.tune and args.skip_cv:
        raise ValueError("Cannot use --tune together with --skip_cv.")

    df = pd.read_csv(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.splits)
    if args.folds and args.folds != len(splits):
        print(
            f"Warning: --folds={args.folds} ignored; using {len(splits)} folds from {args.splits}",
        )
    args.folds = len(splits)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.tune and not args.record_trials:
        default_record_dir = models_dir / "tuning_trials"
        default_record_dir.mkdir(parents=True, exist_ok=True)
        args.record_trials = str(default_record_dir)

    summary_rows: List[Tuple[str, Dict[str, float]]] = []
    for target in args.targets:
        alias = TARGET_ALIASES[target]
        print(f"\n{'=' * 60}")
        print(f"Processing target: {target}")
        print(f"{'=' * 60}")

        X, y = prepare_features(df, target)
        columns = list(X.columns)

        set_seed(args.seed)
        original_lr = args.lr
        original_weight_decay = args.weight_decay
        cv_summary: Dict[str, float] = {}
        cv_predictions: List[Dict[str, float]] = []

        if args.tune:
            best_config, optimal_epochs, cv_summary, cv_predictions = run_lr_weight_decay_search(
                X, y, columns, args, device, splits, alias
            )
            args.lr = best_config["lr"]
            args.weight_decay = best_config["weight_decay"]
            print(
                "Selected optimizer hyperparameters: "
                f"lr={args.lr:.6g}, weight_decay={args.weight_decay:.6g}"
            )
        elif args.skip_cv:
            if args.full_epochs is None or args.full_epochs <= 0:
                raise ValueError(
                    "When using --skip_cv you must provide a positive --full_epochs value."
                )
            optimal_epochs = args.full_epochs
        else:
            optimal_epochs, cv_summary, cv_predictions = cross_validate_epochs(
                X, y, columns, args, device, splits
            )

        if cv_summary:
            mean_rmse = cv_summary.get("mean_val_rmse")
            mean_loss = cv_summary.get("mean_val_loss")
            if mean_rmse is not None and mean_loss is not None:
                print(
                    f"Cross-validation summary: mean_val_rmse={mean_rmse:.6f}, "
                    f"mean_val_loss={mean_loss:.6f}"
                )

        metrics_from_cv = write_cv_predictions(cv_predictions, results_dir, alias)
        if metrics_from_cv:
            summary_rows.append((target, metrics_from_cv))

        model, pipeline, metrics = train_full_dataset(
            X, y, columns, optimal_epochs, args, device
        )

        metrics_summary = {
            "epochs_trained": optimal_epochs,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        }
        print(
            f"Final training metrics for {target}: "
            f"epochs={metrics_summary['epochs_trained']}, "
            f"rmse={metrics_summary['rmse']:.4f}, "
            f"mae={metrics_summary['mae']:.4f}, "
            f"r2={metrics_summary['r2']:.4f}"
        )

        save_artifacts(model, pipeline, columns, alias, args, metrics_summary)

        args.lr = original_lr
        args.weight_decay = original_weight_decay

    write_performance_summary(summary_rows, results_dir)


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train feedforward MLP models on LR-engineered features."
    )
    parser.add_argument(
        "--data",
        default=str(Path("..") / "LR" / "processed" / "ES_features.csv"),
        help="Path to engineered feature CSV.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["Area AVG", "RG AVG", "RDF Peak"],
        help="Target names to train.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5000,
        help="Maximum epochs for cross-validation and training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini-batch size (<=0 uses full batch).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="L2 weight decay for Adam.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=250,
        help="Early stopping patience during CV (<=0 disables).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "splits" / "es_kfold_5_seed42.json"),
        help="Path to a JSON file containing predetermined CV splits.",
    )
    parser.add_argument(
        "--skip_cv",
        action="store_true",
        help="Skip cross-validation and train for a fixed number of epochs.",
    )
    parser.add_argument(
        "--full_epochs",
        type=int,
        default=None,
        help="Epoch count when --skip_cv is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splits and initialization.",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[128, 128, 128],
        help="Hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate applied after each hidden layer.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Search over learning rate and weight decay combinations using cross-validation.",
    )
    parser.add_argument(
        "--lr_grid",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4, 1e-4],
        help="Candidate learning rates used when --tune is enabled.",
    )
    parser.add_argument(
        "--wd_grid",
        type=float,
        nargs="+",
        default=[5e-4, 1e-5, 1e-6],
        help="Candidate weight decay values used when --tune is enabled.",
    )
    parser.add_argument(
        "--record_trials",
        type=str,
        default=None,
        help="Optional path to write tuning trial summaries as JSON.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Unused placeholder for CLI compatibility.",
    )
    parser.add_argument(
        "--models_dir",
        default="models",
        help="Directory for saving model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    for target in args.targets:
        if target not in TARGET_ALIASES:
            raise ValueError(
                f"Unsupported target '{target}'. Expected one of {sorted(TARGET_ALIASES)}."
            )

    run_training(args)


if __name__ == "__main__":
    main()
