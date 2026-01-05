import argparse
import ast
import csv
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    """Three-layer 1D CNN (mirrors training architecture)."""

    def __init__(
        self, hidden_dim: int = 128, dropout: float = 0.2, output_size: int = 1
    ):
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


def parse_input_list(input_str: str) -> List[tuple]:
    """Parse an input list string.

    Args:
        input_str: String formatted like "[(1, 'E4'), (2, 'S2'), ...]".

    Returns:
        A list of (position, type) tuples.
    """
    input_str = input_str.strip()
    # Strip wrapping quotes if the line came from a text file with quotes
    if (input_str.startswith("\"") and input_str.endswith("\"")) or (
        input_str.startswith("'") and input_str.endswith("'")
    ):
        input_str = input_str[1:-1]
    try:
        return ast.literal_eval(input_str)
    except (SyntaxError, ValueError):
        return []


def convert_to_matrix(
    input_list: List[tuple], max_length: int = 20
) -> np.ndarray:
    """Convert parsed tuples into the 3x20 matrix format.

    Args:
        input_list: Parsed (position, type) tuples.
        max_length: Maximum sequence length.

    Returns:
        A numpy.ndarray representing the sequence.
    """
    matrix = np.zeros((3, max_length))
    for pos, type_code in input_list:
        if 1 <= pos <= max_length and type_code not in {"E0", "S0"}:
            idx = pos - 1
            matrix[0, idx] = 1
            if type_code.startswith("E"):
                matrix[1, idx] = int(type_code[1:])
            elif type_code.startswith("S"):
                matrix[2, idx] = int(type_code[1:])
    return matrix


CSV_INPUT_COLUMN = "Input List"


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
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


def main() -> None:
    """Run inference on one or many polymer inputs."""
    parser = argparse.ArgumentParser(description="Predict polymer properties using CNN")
    parser.add_argument("--polymer", help="Polymer input list string")
    parser.add_argument("--file", help="Path to a text file of polymer inputs", default=None)
    parser.add_argument(
        "--csv",
        help=f"Path to a CSV file containing a '{CSV_INPUT_COLUMN}' column",
        default=None,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Computation device",
    )
    parser.add_argument(
        "--out",
        default="predictions.csv",
        help="Path to output CSV file (written for single or batch inputs)",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )

    # Gather inputs
    inputs: List[str]
    if args.csv:
        inputs = _read_inputs_from_csv(args.csv)
    elif args.file:
        inputs = _read_lines(args.file)
    elif args.polymer:
        inputs = [args.polymer]
    else:
        raise SystemExit("Provide --polymer, --file, or --csv with input definitions.")

    matrices = [convert_to_matrix(parse_input_list(s)) for s in inputs]
    input_tensor = torch.tensor(np.stack(matrices, axis=0), dtype=torch.float32).to(device)

    # Target mapping
    targets = {
        "Area AVG": "area",
        "RG AVG": "rg",
        "RDF Peak": "rdf"
    }

    # Load models and predict
    model_cache = {}
    preds_per_target = {}
    with torch.no_grad():
        for target_name, model_name in targets.items():
            model_path = f"models/model_{model_name}.pth"
            model = model_cache.get(model_path)
            if model is None:
                model = Simple1DCNN(hidden_dim=128, dropout=0.2)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                model_cache[model_path] = model
            out = model(input_tensor).view(-1).cpu().numpy().tolist()
            preds_per_target[target_name] = out

    # Write CSV output
    order = ["Area AVG", "RG AVG", "RDF Peak"]
    header = ["Input"] + order
    rows = []
    for i, s in enumerate(inputs):
        row_vals = [preds_per_target[k][i] for k in order]
        rows.append([s] + [f"{v:.4f}" for v in row_vals])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
