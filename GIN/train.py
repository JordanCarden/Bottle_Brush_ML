import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # Prefer package-relative import when available.
    from .preprocess import PolymerDataset  # type: ignore[import-not-found]
except ImportError:
    from preprocess import PolymerDataset  # type: ignore[no-redef]
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


class GIN(nn.Module):
    """Simple 3-layer GIN network."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        def make_block(in_features: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.conv1 = GINConv(make_block(in_dim))
        self.conv2 = GINConv(make_block(hidden_dim))
        self.conv3 = GINConv(make_block(hidden_dim))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
        self.dropout = float(dropout)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv3(h, edge_index)
        h = global_add_pool(h, batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h).view(-1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    target_ID: int,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        targets = data.y.view(data.num_graphs, -1)[:, target_ID]
        targets_norm = (targets - mean) / std
        loss = F.mse_loss(out, targets_norm, reduction="mean")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    target_ID: int,
    return_predictions: bool = False,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model and report metrics on original scale."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds: List[float] = []
    gts: List[float] = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            targets = data.y.view(data.num_graphs, -1)[:, target_ID]
            targets_norm = (targets - mean) / std
            loss = F.mse_loss(out, targets_norm, reduction="mean")
            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs

            preds.extend(((out * std) + mean).cpu().numpy().tolist())
            gts.extend(targets.cpu().numpy().tolist())

    if total_samples == 0:
        return float("nan"), {}, np.array([]), np.array([])

    avg_loss = total_loss / total_samples
    preds_arr = np.array(preds, dtype=float)
    gts_arr = np.array(gts, dtype=float)
    metrics = compute_regression_metrics(gts_arr, preds_arr)
    if return_predictions:
        return avg_loss, metrics, preds_arr, gts_arr
    return avg_loss, metrics, np.array([]), np.array([])


def compute_normalization(
    dataset: PolymerDataset, indices: np.ndarray, target_ID: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean/std for target using the provided indices."""
    values = torch.tensor(
        [dataset[int(i)].y.view(-1)[target_ID].item() for i in indices],
        dtype=torch.float32,
    )
    mean = values.mean()
    std = values.std(unbiased=False)
    if std < 1e-6:
        std = torch.tensor(1.0, dtype=torch.float32)
    return mean.to(device), std.to(device)


def cross_validate_epochs(
    dataset: PolymerDataset,
    target_ID: int,
    args: argparse.Namespace,
    device: torch.device,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[int, Dict[str, float], List[Dict[str, float]]]:
    """Use K-fold CV to estimate a suitable epoch count."""

    best_epochs: List[int] = []
    best_val_losses: List[float] = []
    best_val_rmses: List[float] = []
    all_predictions: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\nFold {fold}/{args.folds}")

        random.seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        torch.manual_seed(args.seed + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + fold)

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False
        )

        mean, std = compute_normalization(dataset, train_idx, target_ID, device)
        model = GIN(in_dim=3, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_epoch = 0
        best_val_loss = float("inf")
        best_val_rmse = float("inf")
        epochs_without_improve = 0
        best_val_preds: Optional[np.ndarray] = None
        best_val_targets: Optional[np.ndarray] = None

        std_scalar = float(std.item())
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_epoch(model, train_loader, mean, std, device, optimizer, target_ID)
            val_loss, val_metrics, val_preds, val_targets = evaluate(
                model, val_loader, mean, std, device, target_ID, return_predictions=True
            )

            should_log = (
                epoch == 1
                or epoch % 10 == 0
                or (args.patience and epochs_without_improve + 1 >= args.patience)
            )
            val_rmse = val_metrics.get("rmse", float("nan"))
            if should_log:
                train_rmse = float(np.sqrt(train_loss)) * std_scalar
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
            for sample_idx, pred, target in zip(
                val_idx.tolist(), best_val_preds.tolist(), best_val_targets.tolist()
            ):
                all_predictions.append(
                    {
                        "fold": int(fold),
                        "epoch": int(best_epoch),
                        "sample_index": int(sample_idx),
                        "y_true": float(target),
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
    dataset: PolymerDataset,
    target_ID: int,
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
            epochs, summary, predictions = cross_validate_epochs(dataset, target_ID, args, device, splits)
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
    dataset: PolymerDataset,
    target_ID: int,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[GIN, Dict[str, float], Dict[str, float]]:
    """Train the final model on the full dataset."""
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    mean, std = compute_normalization(dataset, np.arange(len(dataset)), target_ID, device)

    model = GIN(in_dim=3, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    std_scalar = float(std.item())
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, mean, std, device, optimizer, target_ID)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            train_rmse = float(np.sqrt(loss)) * std_scalar
            print(
                f"Final training pass: epoch {epoch}/{epochs} "
                f"train_rmse={train_rmse:.6f}"
            )

    full_loss, metrics, _, _ = evaluate(model, loader, mean, std, device, target_ID)
    training_summary = {"epochs_trained": epochs, "train_loss": full_loss, **metrics}
    normalization = {"mean": float(mean.item()), "std": float(std.item())}
    return model, training_summary, normalization


def run_training(args: argparse.Namespace) -> None:
    if args.tune and args.skip_cv:
        raise ValueError("Cannot use --tune together with --skip_cv.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_dir = Path(__file__).resolve().parent
    results_dir = repo_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data).resolve()
    default_token = "processed/train.json"
    if args.data == default_token:
        root_path = repo_dir
    elif data_path.is_dir():
        root_path = data_path
    else:
        root_path = data_path.parent

    if not root_path.exists():
        raise FileNotFoundError(f"Could not locate dataset root at {root_path}")

    dataset = PolymerDataset(root=str(root_path))

    splits = load_splits(args.splits)
    if args.folds and args.folds != len(splits):
        print(
            f"Warning: --folds={args.folds} ignored; using {len(splits)} folds from {args.splits}",
        )
    args.folds = len(splits)

    target_index_map = {
        "Area AVG": 0,
        "RG AVG": 1,
        "RDF Peak": 2,
    }
    target_alias_map = {
        "Area AVG": "area",
        "RG AVG": "rg",
        "RDF Peak": "rdf",
    }

    models_dir = root_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.tune and not args.record_trials:
        default_record_dir = models_dir / "tuning_trials"
        default_record_dir.mkdir(parents=True, exist_ok=True)
        args.record_trials = str(default_record_dir)

    stats = {}
    summary_rows: List[Tuple[str, Dict[str, float]]] = []
    for target_name in args.targets:
        if target_name not in target_index_map:
            raise ValueError(
                f"Unsupported target '{target_name}'. Expected one of {list(target_index_map)}."
            )

        idx = target_index_map[target_name]
        alias = target_alias_map[target_name]
        print(f"\n{'='*60}")
        print(f"Processing target: {target_name}")
        print(f"{'='*60}")

        original_lr = args.lr
        original_weight_decay = args.weight_decay
        cv_summary: Dict[str, float] = {}

        cv_predictions: List[Dict[str, float]] = []

        if args.tune:
            best_config, optimal_epochs, cv_summary, cv_predictions = run_lr_weight_decay_search(
                dataset, idx, args, device, splits, alias
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
                dataset, idx, args, device, splits
            )

        if cv_summary:
            mean_rmse = cv_summary.get("mean_val_rmse")
            mean_loss = cv_summary.get("mean_val_loss")
            if mean_rmse is not None and mean_loss is not None:
                print(
                    f"Cross-validation summary: mean_val_rmse={mean_rmse:.6f}, "
                    f"mean_val_loss={mean_loss:.6f}"
                )

        metrics = write_cv_predictions(cv_predictions, results_dir, alias)
        if metrics:
            summary_rows.append((target_name, metrics))

        model, training_summary, normalization = train_full_dataset(
            dataset, idx, optimal_epochs, args, device
        )

        model_path = models_dir / f"model_{alias}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to: {model_path}")

        stats[alias] = normalization
        print(
            f"Final training metrics for {target_name}: "
            f"epochs={training_summary['epochs_trained']}, "
            f"rmse={training_summary['rmse']:.4f}, "
            f"mae={training_summary['mae']:.4f}, "
            f"r2={training_summary['r2']:.4f}"
        )

        args.lr = original_lr
        args.weight_decay = original_weight_decay

    with (models_dir / "normalization_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    write_performance_summary(summary_rows, results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GIN models with K-fold epoch selection"
    )
    parser.add_argument(
        "--data",
        default="processed/train.json",
        help=(
            "Dataset location (default mirrors CNN). "
            "If this points to a JSON/PT file, its parent directory is used as the dataset root."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["Area AVG", "RG AVG", "RDF Peak"],
        help="Target names to train",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5000,
        help="Maximum epochs considered during cross-validation.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate for Adam optimizer."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for Adam optimizer."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=500,
        help="Patience for early stopping during cross-validation.",
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of folds used for cross-validation."
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "splits" / "es_kfold_5_seed42.json"),
        help="Path to a JSON file containing K-fold indices.",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension size."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability applied after convolutional blocks and the penultimate linear layer.",
    )
    parser.add_argument(
        "--skip_cv",
        action="store_true",
        help="Skip cross-validation and train for a fixed number of epochs.",
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
        "--full_epochs",
        type=int,
        default=None,
        help="Epoch count to use when --skip_cv is set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
