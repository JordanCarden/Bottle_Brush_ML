"""CLI for predicting polymer properties using linear regression."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List
import joblib
import pandas as pd

from preprocess import feature_dataframe_from_string

TARGET_MAP = {
    "area": "Area AVG",
    "rg": "RG AVG",
    "rdf": "RDF Peak",
}


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
    parser = argparse.ArgumentParser(description="Predict polymer properties using Linear Regression")
    parser.add_argument("--polymer", help="Polymer input list string")
    parser.add_argument("--file", help="Path to a text file of polymer inputs", default=None)
    parser.add_argument(
        "--csv",
        help=f"Path to a CSV file containing a '{CSV_INPUT_COLUMN}' column",
        default=None,
    )
    parser.add_argument("--out", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

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

    # Load models once
    models = {
        name: joblib.load(Path("models") / f"lr_{name}.joblib") for name in TARGET_MAP
    }

    # Build predictions
    per_target_preds = {TARGET_MAP[name]: [] for name in TARGET_MAP}
    for s in inputs:
        df = feature_dataframe_from_string(s)
        for name, target_name in TARGET_MAP.items():
            model = models[name]
            X = df[model.feature_names_in_]
            pred = float(model.predict(X)[0])
            per_target_preds[target_name].append(pred)

    # Write CSV
    order = ["Area AVG", "RG AVG", "RDF Peak"]
    header = ["Input"] + order
    rows = []
    for i, s in enumerate(inputs):
        row_vals = [per_target_preds[k][i] for k in order]
        rows.append([s] + [f"{v:.4f}" for v in row_vals])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
