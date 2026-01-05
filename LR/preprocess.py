"""Feature engineering utilities for polymer dataset."""

from __future__ import annotations

import ast
import argparse
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd


def shifted_log1p(arr: np.ndarray) -> np.ndarray:
    """Applies ``log1p`` after shifting array to eliminate negatives."""
    shift = (-np.minimum(0, arr.min(axis=0))) + 1
    return np.log1p(arr + shift)


def parse_vectors(s: str) -> list[int]:
    """Parses encoded vectors from the dataset or a text file line.

    Accepts strings like "[(1, 'S4'), (2, 'E3')]" and tolerates extra
    wrapping quotes (e.g., lines copied with surrounding quotes).

    Returns a list of signed integers: positive for "S", negative for "E".
    """
    text = s.strip()
    # Strip one layer of wrapping quotes if present
    if (text.startswith("\"") and text.endswith("\"")) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()

    # First parse attempt
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []

    # Some inputs may be double-encoded strings; unwrap once more
    if isinstance(parsed, str):
        try:
            parsed = ast.literal_eval(parsed)
        except Exception:
            return []

    vector: list[int] = []
    try:
        for item in parsed:
            # Expect (index, code)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                code = str(item[1])
                match = re.search(r"([ES])(\d+)", code)
                if not match:
                    continue
                sign, num = match.groups()
                value = (-1 if sign == "E" else 1) * int(num)
                vector.append(value)
    except TypeError:
        return []

    return vector



def _max_consecutive_by_condition(values: Iterable[int], condition: Callable[[int], bool]) -> int:
    """Calculates longest consecutive run satisfying a condition."""
    max_len = 0
    curr_len = 0
    for value in values:
        if condition(value):
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 0
    return max_len


def _extract_blocks(values: Iterable[int]) -> list[int]:
    """Extracts consecutive blocks of the same sign."""
    blocks = []
    curr_len = 0
    curr_sign = 0
    for value in values:
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign == 0:
            if curr_len:
                blocks.append(curr_len)
            curr_len = 0
            curr_sign = 0
        else:
            if sign != curr_sign:
                if curr_len:
                    blocks.append(curr_len)
                curr_len = 1
                curr_sign = sign
            else:
                curr_len += 1
    if curr_len:
        blocks.append(curr_len)
    return blocks


def _apply_block_stat(values: Iterable[int], fn: Callable[[Iterable[int]], float]) -> float:
    blocks = _extract_blocks(values)
    return fn(blocks) if blocks else 0.0


def max_block_size(values: Iterable[int]) -> float:
    return _apply_block_stat(values, max)


def min_block_size(values: Iterable[int]) -> float:
    return _apply_block_stat(values, min)


def mean_block_size(values: Iterable[int]) -> float:
    return _apply_block_stat(values, lambda b: sum(b) / len(b))


def std_block_size(values: Iterable[int]) -> float:
    return _apply_block_stat(values, np.std)


def count_transitions(values: Iterable[int]) -> int:
    """Counts sign transitions in the sequence."""
    transitions = 0
    last_sign = 0
    for value in values:
        current_sign = 1 if value > 0 else -1 if value < 0 else 0
        if last_sign != 0 and current_sign != 0 and current_sign != last_sign:
            transitions += 1
        if current_sign:
            last_sign = current_sign
    return transitions


def blockiness(values: Iterable[int]) -> float:
    """Computes normalized blockiness metric."""
    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in values]
    filtered = [s for s in signs if s != 0]
    if len(filtered) < 2:
        return 0.0
    same_pairs = sum(1 for i in range(1, len(filtered)) if filtered[i] == filtered[i - 1])
    return same_pairs / (len(filtered) - 1)


def gini_coefficient(values: Iterable[int]) -> float:
    """Calculates the Gini coefficient of absolute values."""
    arr = np.abs(np.array(list(values), dtype=float))
    if arr.sum() == 0:
        return 0.0
    arr_sorted = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return (2 * (index * arr_sorted).sum()) / (n * arr.sum()) - (n + 1) / n


def fft_stats(values: Iterable[int]) -> dict[str, float]:
    """Returns basic statistics of FFT magnitudes."""
    arr = np.array(list(values), dtype=float)
    ft = np.fft.fft(arr)
    abs_ft = np.abs(ft)
    return {
        "max": float(np.max(abs_ft)),
        "mean": float(np.mean(abs_ft)),
        "sum": float(np.sum(abs_ft)),
        "std": float(np.std(abs_ft)),
    }


def hydrophobic_hydrophilic_ratio(values: Iterable[int]) -> float:
    """Ratio of positive entries over all non-zero entries."""
    vector = np.array(list(values))
    pos = np.sum(vector > 0)
    neg = np.sum(vector < 0)
    total = pos + neg
    return float(pos / total) if total else 0.0


def hydrophobic_hydrophilic_ratio_weighted(values: Iterable[int]) -> float:
    """Weighted ratio using magnitudes."""
    vector = np.array(list(values))
    pos = np.sum(vector[vector > 0])
    neg = np.sum(np.abs(vector[vector < 0]))
    total = pos + neg
    return float(pos / total) if total else 0.0


def harwoods_blockiness(values: Iterable[int]) -> float:
    """Harwood's blockiness metric."""
    n = len(list(values))
    if n < 2:
        return 0.0
    count_a = sum(1 for v in values if v > 0)
    p_a = count_a / n
    count_aa = sum(1 for i in range(n - 1) if values[i] > 0 and values[i + 1] > 0)
    p_aa = count_aa / (n - 1)
    if 1 - p_a == 0:
        return 0.0
    return (p_aa - p_a) / (1 - p_a)


def mayo_lewis(values: Iterable[int], r1: float, r2: float) -> float:
    """Mayo-Lewis reactivity ratio."""
    n = len(list(values))
    if n == 0:
        return 0.0
    count_m1 = sum(1 for v in values if v > 0)
    count_m2 = sum(1 for v in values if v < 0)
    if count_m2 == 0:
        return 1.0
    m1 = count_m1 / n
    m2 = count_m2 / n
    denominator = m2 * (r2 * m2 + m1)
    if denominator == 0:
        return 0.0
    return (m1 * (r1 * m1 + m2)) / denominator


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes engineered features for a dataframe."""
    df = df.copy()

    df["backbone_length"] = df["Input List"].apply(len)

    df["mean_charge"] = df["Input List"].apply(np.mean)
    df["sum_charge"] = df["Input List"].apply(np.sum)
    df["max_charge"] = df["Input List"].apply(np.max)
    df["min_charge"] = df["Input List"].apply(np.min)
    df["std_charge"] = df["Input List"].apply(np.std)

    df["mean_length"] = df["Input List"].apply(lambda x: np.mean(np.abs(x)))
    df["max_length"] = df["Input List"].apply(lambda x: np.max(np.abs(x)))
    df["std_length"] = df["Input List"].apply(lambda x: np.std(np.abs(x)))

    df["max_S_block"] = df["Input List"].apply(
        lambda x: _max_consecutive_by_condition(x, lambda v: v > 0)
    )
    df["max_E_block"] = df["Input List"].apply(
        lambda x: _max_consecutive_by_condition(x, lambda v: v < 0)
    )

    df["transitions"] = df["Input List"].apply(count_transitions)
    df["max_block_size"] = df["Input List"].apply(max_block_size)
    df["min_block_size"] = df["Input List"].apply(min_block_size)
    df["mean_block_size"] = df["Input List"].apply(mean_block_size)
    df["std_block_size"] = df["Input List"].apply(std_block_size)
    df["blockiness"] = df["Input List"].apply(blockiness)
    df["gini"] = df["Input List"].apply(gini_coefficient)

    df["max_fft_value"] = df["Input List"].apply(lambda x: fft_stats(x)["max"])
    df["mean_fft_value"] = df["Input List"].apply(lambda x: fft_stats(x)["mean"])
    df["sum_fft_value"] = df["Input List"].apply(lambda x: fft_stats(x)["sum"])
    df["std_fft_value"] = df["Input List"].apply(lambda x: fft_stats(x)["std"])

    df["hydrophobic_ratio"] = df["Input List"].apply(hydrophobic_hydrophilic_ratio)
    df["hydrophobic_ratio_weighted"] = df["Input List"].apply(
        hydrophobic_hydrophilic_ratio_weighted
    )

    df["harwoods_blockiness"] = df["Input List"].apply(harwoods_blockiness)
    df["mayo_lewis"] = df["Input List"].apply(lambda x: mayo_lewis(x, r1=1, r2=1))

    return df


def feature_dataframe_from_string(polymer_str: str) -> pd.DataFrame:
    """Creates a feature dataframe from a polymer string."""
    df = pd.DataFrame({"Input List": [parse_vectors(polymer_str)]})
    return add_features(df)


def create_feature_dataframe(path: str) -> pd.DataFrame:
    """Loads the raw dataset and generates additional features."""
    df = pd.read_csv(path)
    df["Input List"] = df["Input List"].apply(parse_vectors)
    return add_features(df)


def save_feature_dataframe(df: pd.DataFrame, path: str) -> None:
    """Saves feature dataframe to CSV."""
    df.to_csv(path, index=False)


def main() -> None:
    """Create engineered feature CSV for the LR model."""
    parser = argparse.ArgumentParser(
        description="Create engineered feature CSV for the LR/MLP models."
    )
    parser.add_argument(
        "--csv",
        default="../data/ES.csv",
        help="Path to raw dataset CSV (Input List + targets).",
    )
    parser.add_argument(
        "--out",
        default="processed/ES_features.csv",
        help="Output path for engineered feature CSV.",
    )
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = create_feature_dataframe(args.csv)
    save_feature_dataframe(df, str(output_path))


if __name__ == "__main__":
    main()
