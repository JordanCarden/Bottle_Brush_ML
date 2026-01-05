#!/usr/bin/env python3
"""Predict polymer properties with PyTorch-trained MLP models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
LR_DIR = REPO_ROOT / "LR"
if str(LR_DIR) not in sys.path:
    sys.path.insert(0, str(LR_DIR))

from preprocess import feature_dataframe_from_string  # type: ignore[attr-defined]

TARGET_MAP = {"area": "Area AVG", "rg": "RG AVG", "rdf": "RDF Peak"}
CSV_INPUT_COLUMN = "Input List"


class FeedForwardNet(nn.Module):
    """Matches the architecture used during training."""

    def __init__(self, input_dim: int, hidden_sizes: List[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
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


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _read_inputs_from_csv(path: str, column: str = CSV_INPUT_COLUMN) -> List[str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"CSV file {path} does not have '{column}' column")
        inputs = []
        for row in reader:
            value = (row.get(column) or "").strip()
            if value and not value.startswith("#"):
                inputs.append(value)
        if not inputs:
            raise ValueError(f"No rows found in column '{column}' of {path}")
    return inputs


def _load_bundle(
    models_dir: Path,
    alias: str,
    device: torch.device,
) -> Tuple[object, FeedForwardNet, dict]:
    meta_path = models_dir / f"mlp_{alias}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata for '{alias}' at {meta_path}")
    metadata = json.loads(meta_path.read_text())

    pipeline_name = metadata.get("pipeline", f"mlp_{alias}_prep.joblib")
    pipeline_path = models_dir / pipeline_name
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Missing preprocessor pipeline at {pipeline_path}")
    pipeline = joblib.load(pipeline_path)

    model_name = metadata.get("model", f"mlp_{alias}.pt")
    model_path = models_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights at {model_path}")

    hidden_sizes = [int(h) for h in metadata.get("hidden_sizes", [])]
    dropout = float(metadata.get("dropout", 0.2))
    input_dim = metadata.get("input_dim")
    if input_dim is None:
        raise ValueError(f"Metadata for '{alias}' missing 'input_dim'.")

    model = FeedForwardNet(int(input_dim), hidden_sizes, dropout).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return pipeline, model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict polymer properties using the trained MLP ensemble."
    )
    parser.add_argument("--polymer", help="Polymer input list string.")
    parser.add_argument(
        "--file",
        help="Path to a text file of polymer inputs (one per line).",
        default=None,
    )
    parser.add_argument(
        "--csv",
        help=f"Path to a CSV file containing a '{CSV_INPUT_COLUMN}' column.",
        default=None,
    )
    parser.add_argument(
        "--out",
        default="predictions.csv",
        help="Destination CSV path for predictions.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing mlp_{alias}.pt/.json/.joblib artifacts.",
    )
    args = parser.parse_args()

    if args.csv:
        inputs = _read_inputs_from_csv(args.csv)
    elif args.file:
        inputs = _read_lines(args.file)
    elif args.polymer:
        inputs = [args.polymer]
    else:
        raise SystemExit("Provide --polymer, --file, or --csv with input definitions.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir)

    # Precompute engineered feature frames so we only featurize once.
    raw_feature_frames = [
        feature_dataframe_from_string(inp) for inp in inputs
    ]

    per_target_preds: Dict[str, List[float]] = {TARGET_MAP[a]: [float("nan")] * len(inputs) for a in TARGET_MAP}

    for alias, pretty_name in TARGET_MAP.items():
        pipeline, model, metadata = _load_bundle(models_dir, alias, device)
        columns = metadata.get("columns")
        if not columns:
            raise ValueError(f"Metadata for '{alias}' missing 'columns'.")

        feature_rows = []
        for df in raw_feature_frames:
            missing = [col for col in columns if col not in df.columns]
            if missing:
                raise KeyError(
                    f"Required columns {missing} not present in engineered features."
                )
            feature_rows.append(df.loc[:, columns])

        combined_features = pd.concat(feature_rows, ignore_index=True)
        transformed = pipeline.transform(combined_features)
        inputs_tensor = torch.from_numpy(transformed.astype(np.float32)).to(device)

        with torch.no_grad():
            preds = model(inputs_tensor).cpu().numpy().astype(float)

        per_target_preds[pretty_name] = preds.tolist()

    ordered_targets = ["Area AVG", "RG AVG", "RDF Peak"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Input"] + ordered_targets)
        for idx, raw_input in enumerate(inputs):
            row = [
                raw_input,
                *[f"{per_target_preds[target][idx]:.4f}" for target in ordered_targets],
            ]
            writer.writerow(row)

    print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
