import argparse
import ast
import csv
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from train import GAT  # reuse architecture


def input_to_graph(polymer_str: str) -> Data:
    s = polymer_str.strip()
    if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    input_list: List[Tuple[int, str]] = ast.literal_eval(s)

    features = []
    backbone_indices = {}
    for idx, (n, label) in enumerate(input_list):
        features.append([1, 0, 0])
        backbone_indices[n] = idx

    edge_index = [[], []]
    for i in range(len(input_list) - 1):
        edge_index[0].append(i)
        edge_index[1].append(i + 1)
        edge_index[0].append(i + 1)
        edge_index[1].append(i)

    next_node = len(features)
    for n, label in input_list:
        if label not in ("E0", "S0"):
            m = int(label[1:])
            bead_type = label[0]
            prev = backbone_indices[n]
            for _ in range(m):
                features.append([0, 1, 0] if bead_type == "S" else [0, 0, 1])
                curr = next_node
                edge_index[0].append(prev)
                edge_index[1].append(curr)
                edge_index[0].append(curr)
                edge_index[1].append(prev)
                prev = curr
                next_node += 1

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


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
    parser = argparse.ArgumentParser(description="Predict polymer properties using GAT")
    parser.add_argument("--polymer", help="Polymer input list string")
    parser.add_argument("--file", help="Path to a text file of polymer inputs", default=None)
    parser.add_argument(
        "--csv",
        help=f"Path to a CSV file containing a '{CSV_INPUT_COLUMN}' column",
        default=None,
    )
    parser.add_argument("--out", default="predictions.csv", help="Output CSV path")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension used during training.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Attention heads used during training.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout applied after attention layers.",
    )
    parser.add_argument(
        "--attn-dropout",
        type=float,
        default=0.2,
        help="Attention dropout used in GATConv layers.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_map = {"area": 0, "rg": 1, "rdf": 2}

    stats_path = Path("models/normalization_stats.json")
    if not stats_path.exists():
        raise FileNotFoundError(f"Could not find normalization file at {stats_path}")

    with stats_path.open() as f:
        stats = json.load(f)

    models = {}
    for name in target_map:
        model = GAT(
            in_dim=3,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
        ).to(device)
        state_dict = torch.load(f"models/model_{name}.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models[name] = model

    if args.csv:
        inputs = _read_inputs_from_csv(args.csv)
    elif args.file:
        inputs = _read_lines(args.file)
    elif args.polymer:
        inputs = [args.polymer]
    else:
        raise SystemExit("Provide --polymer, --file, or --csv with input definitions.")

    data_list = [input_to_graph(s) for s in inputs]
    batch = Batch.from_data_list(data_list).to(device)

    preds_per_target = {}
    with torch.no_grad():
        for name, idx in target_map.items():
            out = models[name](batch).view(-1)
            mean = stats[name]["mean"]
            std = stats[name]["std"]
            pred = (out.cpu().numpy() * std) + mean
            preds_per_target[name] = pred.tolist()

    order = ["area", "rg", "rdf"]
    header_map = {"area": "Area AVG", "rg": "RG AVG", "rdf": "RDF Peak"}
    header = ["Input"] + [header_map[k] for k in order]
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
