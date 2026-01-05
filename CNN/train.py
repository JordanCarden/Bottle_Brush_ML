import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.splits import load_splits


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and R2 from provided arrays."""
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
    """Write CV predictions for a target and return aggregate metrics."""
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
    """Persist aggregate metrics per target."""
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


class MatrixDataset(Dataset):
    """Dataset for processed JSON matrices.

    Args:
        json_file: Path to the JSON file.
        target_key: Target value key in each record.
    """

    def __init__(self, json_file: str, target_key: str):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        matrix = torch.tensor(item["input_matrix"], dtype=torch.float32)
        target = torch.tensor([item[self.target_key]], dtype=torch.float32)
        return matrix, target


class Simple1DCNN(nn.Module):
    """Three-layer 1D CNN with shared hidden width."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2, output_size: int = 1) -> None:
        super().__init__()
        channels = hidden_dim
        drop = float(dropout)

        self.features = nn.Sequential(
            nn.Conv1d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.size(0), 3, -1)
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.fc2(x)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Perform one training epoch."""
    model.train()
    total_loss = 0.0
    for data, target_vals in loader:
        data = data.to(device)
        target_vals = target_vals.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target_vals)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate the model on a loader."""
    model.eval()
    total_loss = 0.0
    preds: List[float] = []
    gts: List[float] = []
    with torch.no_grad():
        for data, target_vals in loader:
            data = data.to(device)
            target_vals = target_vals.to(device)
            outputs = model(data)
            loss = criterion(outputs, target_vals)
            total_loss += loss.item() * data.size(0)
            preds.extend(outputs.cpu().numpy().flatten().tolist())
            gts.extend(target_vals.cpu().numpy().flatten().tolist())

    if not preds:
        return float("nan"), {}, np.array([]), np.array([])

    avg_loss = total_loss / len(loader.dataset)
    preds_arr = np.array(preds, dtype=float)
    gts_arr = np.array(gts, dtype=float)
    metrics = compute_regression_metrics(gts_arr, preds_arr)
    return avg_loss, metrics, preds_arr, gts_arr


def cross_validate_epochs(
    dataset: MatrixDataset,
    target: str,
    args: argparse.Namespace,
    device: torch.device,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[int, Dict[str, float], List[Dict[str, float]]]:
    """Run K-fold CV to estimate the ideal number of epochs."""
    best_epochs: List[int] = []
    best_val_losses: List[float] = []
    best_val_rmses: List[float] = []
    criterion = nn.MSELoss()
    all_predictions: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\nFold {fold}/{args.folds} for {target}")
        set_seed(args.seed + fold)

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False
        )

        model = Simple1DCNN(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        best_epoch = 0
        best_val_loss = float("inf")
        best_val_rmse = float("inf")
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
                or (args.patience and epochs_without_improve + 1 >= args.patience)
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
                if args.patience and epochs_without_improve >= args.patience:
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
    print(f"\nSuggested epoch count for {target}: {optimal_epochs}")
    return optimal_epochs, summary, all_predictions


def run_lr_weight_decay_search(
    dataset: MatrixDataset,
    target: str,
    args: argparse.Namespace,
    device: torch.device,
    splits: List[Tuple[np.ndarray, np.ndarray]],
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
                dataset, target, args, device, splits
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


def train_full_dataset(
    dataset: MatrixDataset,
    target: str,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Simple1DCNN, Dict[str, float]]:
    """Train on the full dataset for the chosen number of epochs."""
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    criterion = nn.MSELoss()

    model = Simple1DCNN(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(model, loader, criterion, optimizer, device)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            train_rmse = float(np.sqrt(avg_loss))
            print(
                f"Final training pass {target}: epoch {epoch}/{epochs} "
                f"train_rmse={train_rmse:.6f}"
            )

    full_loss, metrics, _, _ = evaluate(model, loader, criterion, device)
    return model, {"epochs_trained": epochs, "train_loss": full_loss, **metrics}


def run_training(args: argparse.Namespace) -> None:
    if args.tune and args.skip_cv:
        raise ValueError("Cannot use --tune together with --skip_cv.")

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Could not find dataset JSON at {args.data}. "
            "Set --data to the combined training file (e.g., processed/data.json)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    target_map = {
        "Area AVG": "area",
        "RG AVG": "rg",
        "RDF Peak": "rdf",
    }

    splits = load_splits(args.splits)
    if args.folds and args.folds != len(splits):
        print(
            f"Warning: --folds={args.folds} ignored; using {len(splits)} folds from {args.splits}",
        )
    args.folds = len(splits)

    if args.tune and not args.record_trials:
        default_record_dir = models_dir / "tuning_trials"
        default_record_dir.mkdir(parents=True, exist_ok=True)
        args.record_trials = str(default_record_dir)

    summary_rows: List[Tuple[str, Dict[str, float]]] = []
    for target in args.targets:
        if target not in target_map:
            raise ValueError(
                f"Unsupported target '{target}'. Expected one of {sorted(target_map)}."
            )

        print(f"\n{'='*60}")
        print(f"Processing target: {target}")
        print(f"{'='*60}")

        full_dataset = MatrixDataset(args.data, target_key=target)

        set_seed(args.seed)
        original_lr = args.lr
        original_weight_decay = args.weight_decay
        cv_summary: Dict[str, float] = {}
        cv_predictions: List[Dict[str, float]] = []

        if args.tune:
            best_config, optimal_epochs, cv_summary, cv_predictions = run_lr_weight_decay_search(
                full_dataset,
                target,
                args,
                device,
                splits,
                target_map[target],
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
                full_dataset, target, args, device, splits
            )

        if cv_summary:
            mean_rmse = cv_summary.get("mean_val_rmse")
            mean_loss = cv_summary.get("mean_val_loss")
            if mean_rmse is not None and mean_loss is not None:
                print(
                    f"Cross-validation summary: mean_val_rmse={mean_rmse:.6f}, "
                    f"mean_val_loss={mean_loss:.6f}"
                )

        metrics = write_cv_predictions(cv_predictions, results_dir, target_map[target])
        if metrics:
            summary_rows.append((target, metrics))

        final_model, training_stats = train_full_dataset(
            full_dataset, target, optimal_epochs, args, device
        )

        model_path = models_dir / f"model_{target_map[target]}.pth"
        torch.save(final_model.state_dict(), model_path)
        print(f"Saved model to: {model_path}")
        print(
            f"Final training metrics for {target}: "
            f"epochs={training_stats['epochs_trained']}, "
            f"rmse={training_stats['rmse']:.4f}, "
            f"mae={training_stats['mae']:.4f}, "
            f"r2={training_stats['r2']:.4f}"
        )

        args.lr = original_lr
        args.weight_decay = original_weight_decay

    write_performance_summary(summary_rows, results_dir)


def main() -> None:
    """Train and save one model per target."""
    parser = argparse.ArgumentParser(
        description="Train CNN models with K-fold epoch selection"
    )
    parser.add_argument(
        "--data",
        default="processed/data.json",
        help="Path to the JSON dataset used for training",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["Area AVG", "RG AVG", "RDF Peak"],
        help="Target names to train",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=5000, help="Upper bound for epoch search"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument(
        "--patience", type=int, default=250, help="Early stopping patience during CV"
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of CV folds used for epoch search"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "splits" / "es_kfold_5_seed42.json"),
        help="Path to a JSON file containing fixed train/val splits.",
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
        help="Epoch count to use when --skip_cv is set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Shared hidden width for convolutional channels and dense layer.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability applied after convolutional blocks and the penultimate linear layer.",
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
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
