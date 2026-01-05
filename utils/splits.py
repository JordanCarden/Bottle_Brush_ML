"""Utilities for loading existing dataset splits and generating new ones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


FoldIndices = Tuple[np.ndarray, np.ndarray]


def _kfold_indices(
    n_samples: int,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int | None,
) -> List[FoldIndices]:
    """Generate K-fold train/validation indices matching scikit-learn's KFold.

    This avoids requiring scikit-learn at runtime while keeping the split
    behavior consistent with ``sklearn.model_selection.KFold``.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot exceed n_samples")

    indices = np.arange(n_samples, dtype=np.int64)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int64)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    folds: List[FoldIndices] = []
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, val_idx))
        current = stop

    return folds


def load_splits(path: str | Path) -> List[FoldIndices]:
    """Load train/validation index splits from a JSON file.

    The expected format is::

        {
          "folds": [
            {"train": [...], "val": [...]},
            ...
          ]
        }

    Returns a list of ``(train_indices, val_indices)`` tuples as ``np.ndarray``.
    """

    split_path = Path(path).expanduser().resolve()
    if not split_path.exists():
        raise FileNotFoundError(f"Splits file not found: {split_path}")

    with split_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    folds_data: Iterable[dict] = data.get("folds", [])
    splits: List[FoldIndices] = []
    for fold_id, fold in enumerate(folds_data, 1):
        try:
            train = np.asarray(fold["train"], dtype=np.int64)
            val = np.asarray(fold["val"], dtype=np.int64)
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Fold {fold_id} missing key: {exc}") from exc

        if train.ndim != 1 or val.ndim != 1:
            raise ValueError(f"Fold {fold_id} indices must be 1-D arrays")
        splits.append((train, val))

    if not splits:
        raise ValueError(f"No folds found in splits file: {split_path}")

    return splits


def build_kfold_splits(
    n_samples: int,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = None,
) -> List[FoldIndices]:
    """Return K-fold train/validation indices mirroring scikit-learn's ``KFold``."""
    return _kfold_indices(
        n_samples,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def write_splits_json(
    folds: Iterable[FoldIndices],
    output_path: str | Path,
    *,
    meta: dict | None = None,
) -> None:
    """Serialize splits to JSON using the format expected by :func:`load_splits`."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "folds": [
            {"train": train.astype(np.int64).tolist(), "val": val.astype(np.int64).tolist()}
            for train, val in folds
        ]
    }
    if meta:
        payload["meta"] = meta

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _infer_sample_count(data_path: Path) -> int:
    """Infer the number of samples from a structured data file."""

    data_path = data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    suffix = data_path.suffix.lower()
    try:
        if suffix == ".npy":
            array = np.load(data_path)
            return int(array.shape[0])

        if suffix == ".npz":
            with np.load(data_path) as archive:
                if not archive.files:
                    raise ValueError(f"NPZ file contains no arrays: {data_path}")
                first = archive.files[0]
                return int(archive[first].shape[0])

        read_kwargs = {"sep": "\t"} if suffix == ".tsv" else {}
        frame = pd.read_csv(data_path, **read_kwargs)
        return int(len(frame))
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Failed to infer sample count from {data_path}") from exc


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for generating K-fold JSON splits."""

    parser = argparse.ArgumentParser(
        description="Generate K-fold JSON splits identical to scikit-learn's KFold output."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to the dataset file used to determine the number of samples.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of samples. Overrides --data-path when provided.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("splits/kfold_splits.json"),
        help="Destination JSON file (created if missing).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds to generate.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed passed to sklearn.model_selection.KFold.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before creating folds.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for generating K-fold split JSON files."""

    args = _parse_args(argv)

    if args.n_samples is not None:
        n_samples = int(args.n_samples)
    elif args.data_path is not None:
        n_samples = _infer_sample_count(args.data_path)
    else:
        raise ValueError("Either --n-samples or --data-path must be provided.")

    folds = build_kfold_splits(
        n_samples,
        n_splits=int(args.n_splits),
        shuffle=not args.no_shuffle,
        random_state=int(args.random_state),
    )

    meta = {
        "generator": "sklearn.model_selection.KFold",
        "n_samples": n_samples,
        "n_splits": int(args.n_splits),
        "shuffle": bool(not args.no_shuffle),
        "random_state": int(args.random_state),
    }
    if args.data_path is not None:
        meta["data_path"] = str(Path(args.data_path).expanduser().resolve())

    write_splits_json(folds, args.output_path, meta=meta)


if __name__ == "__main__":  # pragma: no cover - CLI utility
    main()
