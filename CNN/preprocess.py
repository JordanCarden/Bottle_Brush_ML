import argparse
import json
import os
from typing import Dict, List

import ast

import numpy as np
import pandas as pd


def parse_input_list(input_str: str) -> List[tuple]:
    """Parse an input list string from the CSV.

    Args:
        input_str: String formatted like "[(1, 'E4'), (2, 'S2'), ...]".

    Returns:
        A list of ``(position, type)`` tuples.
    """
    input_str = input_str.strip()
    try:
        return ast.literal_eval(input_str)
    except (SyntaxError, ValueError):
        return []


def convert_to_matrix(
    input_list: List[tuple], max_length: int = 20
) -> np.ndarray:
    """Convert parsed tuples into the 3x``max_length`` matrix format.

    Args:
        input_list: Parsed ``(position, type)`` tuples.
        max_length: Maximum sequence length.

    Returns:
        A ``numpy.ndarray`` representing the sequence.
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


def build_sample(row: pd.Series, row_id: int) -> Dict[str, object]:
    """Build a single dataset sample from a CSV row.

    Args:
        row: A row from the raw CSV.
        row_id: Row index to use if Name column doesn't exist.

    Returns:
        A dictionary with the model input matrix and target values.
    """
    input_list = parse_input_list(row["Input List"])
    matrix = convert_to_matrix(input_list)

    # Handle missing Name column
    if "Name" in row.index:
        sample_id = row["Name"]
    else:
        sample_id = f"sample_{row_id}"

    sample = {
        "id": sample_id,
        "input_matrix": matrix.tolist(),
    }

    for col in row.index:
        if col in {"Input List", "Name"}:
            continue
        sample[col] = float(row[col])
    return sample


def main() -> None:
    """Entry point for CSV preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess CSV into model-ready datasets"
    )
    parser.add_argument("--csv", default="../data/ES.csv", help="Path to raw CSV file")
    parser.add_argument(
        "--outdir", default="processed", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    samples = [build_sample(row, idx) for idx, (_, row) in enumerate(df.iterrows())]

    dataset_path = os.path.join(args.outdir, "data.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)


if __name__ == "__main__":
    main()
