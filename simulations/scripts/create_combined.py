#!/usr/bin/env python3
"""
Rebuild data/combined.csv from the constituent CSV inputs.

The original combined.csv pulls numeric summaries from several source files
under data/.  This script reproduces that behaviour and writes the refreshed
results back to data/combined.csv.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "combined.csv"

EXCLUDED_INPUT_LISTS: frozenset[str] = frozenset(
    (
        "[(1, 'S5'), (2, 'S5'), (3, 'S5'), (4, 'S5'), (5, 'S5'), (6, 'S5'), (7, 'S5'), (8, 'S5'), (9, 'S5'), (10, 'S5')]",
        "[(1, 'E4'), (2, 'E4'), (3, 'E4'), (4, 'E4'), (5, 'E4'), (6, 'E4'), (7, 'E4'), (8, 'E4'), (9, 'E4'), (10, 'E4'), (11, 'E4'), (12, 'E4')]",
        "[(1, 'S8'), (2, 'S8'), (3, 'S8'), (4, 'S8'), (5, 'S8'), (6, 'S8'), (7, 'S8'), (8, 'S8'), (9, 'S8'), (10, 'S8'), (11, 'S8'), (12, 'S8'), (13, 'S8'), (14, 'S8'), (15, 'S8'), (16, 'S8'), (17, 'S8'), (18, 'S8')]",
        "[(1, 'E10'), (2, 'E10'), (3, 'E10'), (4, 'E10'), (5, 'E10'), (6, 'E10'), (7, 'E10'), (8, 'E10'), (9, 'E10'), (10, 'E10'), (11, 'E10'), (12, 'E10'), (13, 'E10'), (14, 'E10'), (15, 'E10'), (16, 'E10'), (17, 'E10'), (18, 'E10'), (19, 'E10'), (20, 'E10')]",
        "[(1, 'S20'), (2, 'S20'), (3, 'S20'), (4, 'S20'), (5, 'S20'), (6, 'S20'), (7, 'S20'), (8, 'S20'), (9, 'S20'), (10, 'S20')]",
        "[(1, 'E6'), (2, 'E6'), (3, 'E6'), (4, 'E6'), (5, 'S6'), (6, 'S6'), (7, 'S6'), (8, 'S6'), (9, 'E6'), (10, 'E6'), (11, 'E6'), (12, 'E6'), (13, 'S6'), (14, 'S6'), (15, 'S6'), (16, 'S6')]",
        "[(1, 'S7'), (2, 'S7'), (3, 'S7'), (4, 'S7'), (5, 'S7'), (6, 'E7'), (7, 'E7'), (8, 'E7'), (9, 'E7'), (10, 'E7'), (11, 'S7'), (12, 'S7'), (13, 'S7'), (14, 'S7'), (15, 'S7')]",
        "[(1, 'E9'), (2, 'E9'), (3, 'E9'), (4, 'E9'), (5, 'E9'), (6, 'S1'), (7, 'S1'), (8, 'S1'), (9, 'S1'), (10, 'S1'), (11, 'S1'), (12, 'S1'), (13, 'S1'), (14, 'S1'), (15, 'S1'), (16, 'S1'), (17, 'S1'), (18, 'S1')]",
        "[(1, 'S5'), (2, 'E7'), (3, 'S5'), (4, 'E7'), (5, 'E5'), (6, 'E0'), (7, 'E0'), (8, 'E0'), (9, 'E0'), (10, 'E0'), (11, 'E0'), (12, 'E0'), (13, 'E0'), (14, 'E0'), (15, 'E0'), (16, 'E0'), (17, 'E0'), (18, 'E0')]",
        "[(1, 'E0'), (2, 'E0'), (3, 'E0'), (4, 'S3'), (5, 'E3'), (6, 'S3'), (7, 'E3'), (8, 'S3'), (9, 'E3'), (10, 'S3')]",
        "[(1, 'S8'), (2, 'E0'), (3, 'E0'), (4, 'E0'), (5, 'E0'), (6, 'E0'), (7, 'E0'), (8, 'E0'), (9, 'E0'), (10, 'E0'), (11, 'E0'), (12, 'S8')]",
        "[(1, 'E9'), (2, 'S3'), (3, 'S3'), (4, 'S3'), (5, 'S3'), (6, 'S3'), (7, 'S3'), (8, 'S3'), (9, 'S3'), (10, 'S3'), (11, 'S3'), (12, 'S3'), (13, 'S3'), (14, 'E9')]",
        "[(1, 'S8'), (2, 'S8'), (3, 'S8'), (4, 'S8'), (5, 'E1'), (6, 'E1'), (7, 'E1'), (8, 'E1'), (9, 'S8'), (10, 'S8'), (11, 'S8'), (12, 'S8')]",
        "[(1, 'S6'), (2, 'S6'), (3, 'S6'), (4, 'S6'), (5, 'S6'), (6, 'S6'), (7, 'S6'), (8, 'E1'), (9, 'E1'), (10, 'S6'), (11, 'S6'), (12, 'S6'), (13, 'S6'), (14, 'S6'), (15, 'S6'), (16, 'S6')]",
        "[(1, 'E2'), (2, 'E2'), (3, 'E2'), (4, 'E2'), (5, 'E2'), (6, 'E2'), (7, 'E2'), (8, 'S2'), (9, 'S2'), (10, 'S2'), (11, 'S2'), (12, 'S2'), (13, 'S2'), (14, 'S2')]",
        "[(1, 'E1'), (2, 'S1'), (3, 'E1'), (4, 'S1'), (5, 'E1'), (6, 'S1'), (7, 'E1'), (8, 'S1'), (9, 'E1'), (10, 'S1'), (11, 'E1')]",
        "[(1, 'E0'), (2, 'E0'), (3, 'E0'), (4, 'E0'), (5, 'E0'), (6, 'E0'), (7, 'E0'), (8, 'E0'), (9, 'E0'), (10, 'S10')]",
        "[(1, 'E0'), (2, 'E0'), (3, 'E0'), (4, 'E0'), (5, 'E0'), (6, 'E0'), (7, 'E0'), (8, 'E0'), (9, 'E0'), (10, 'E5')]",
        "[(1, 'E0'), (2, 'E0'), (3, 'E0'), (4, 'S4'), (5, 'S4'), (6, 'S4'), (7, 'S4'), (8, 'S4'), (9, 'S4'), (10, 'S4')]",
        "[(1, 'E0'), (2, 'E0'), (3, 'E0'), (4, 'S7'), (5, 'E7'), (6, 'S7'), (7, 'E7'), (8, 'S7'), (9, 'E7'), (10, 'S7')]",
        "[(1, 'S1'), (2, 'S1'), (3, 'S1'), (4, 'S1'), (5, 'S1'), (6, 'S1'), (7, 'S1'), (8, 'S1'), (9, 'S1'), (10, 'S1'), (11, 'E8'), (12, 'E8'), (13, 'E8')]",
        "[(1, 'S6'), (2, 'S0'), (3, 'S0'), (4, 'S0'), (5, 'S0'), (6, 'S0'), (7, 'S0'), (8, 'S0'), (9, 'S0'), (10, 'E6')]",
        "[(1, 'S8'), (2, 'S8'), (3, 'S8'), (4, 'E2'), (5, 'E2'), (6, 'E2'), (7, 'E2'), (8, 'E2'), (9, 'E2'), (10, 'E2'), (11, 'E2'), (12, 'E2'), (13, 'E2'), (14, 'E2'), (15, 'E2'), (16, 'E8'), (17, 'E8'), (18, 'E8')]",
        "[(1, 'E4'), (2, 'E4'), (3, 'E4'), (4, 'E4'), (5, 'S8'), (6, 'S8'), (7, 'S8'), (8, 'S8'), (9, 'S8'), (10, 'S8'), (11, 'S8'), (12, 'S8'), (13, 'S8'), (14, 'S8'), (15, 'E0'), (16, 'E0'), (17, 'S4'), (18, 'S4'), (19, 'S4'), (20, 'S4')]",
        "[(1, 'E7'), (2, 'E7'), (3, 'S2'), (4, 'S2'), (5, 'S2'), (6, 'S2'), (7, 'S2'), (8, 'S2'), (9, 'S2'), (10, 'S2'), (11, 'S2'), (12, 'S2'), (13, 'S2'), (14, 'E7'), (15, 'E7')]",
        "[(1, 'S1'), (2, 'S2'), (3, 'S3'), (4, 'S4'), (5, 'S5'), (6, 'S6'), (7, 'S7'), (8, 'S8'), (9, 'S9'), (10, 'S10')]",
        "[(1, 'S2'), (2, 'S2'), (3, 'S3'), (4, 'S3'), (5, 'S4'), (6, 'S4'), (7, 'S5'), (8, 'S5'), (9, 'S6'), (10, 'S6'), (11, 'S7'), (12, 'S7'), (13, 'S8'), (14, 'S8'), (15, 'S8')]",
        "[(1, 'E0'), (2, 'E1'), (3, 'E2'), (4, 'E3'), (5, 'S4'), (6, 'E4'), (7, 'S5'), (8, 'E5'), (9, 'S6'), (10, 'E6'), (11, 'S7'), (12, 'E7'), (13, 'S8'), (14, 'E8'), (15, 'S9'), (16, 'E9'), (17, 'S10'), (18, 'E10')]",
        "[(1, 'S0'), (2, 'S0'), (3, 'S1'), (4, 'S1'), (5, 'S2'), (6, 'S2'), (7, 'S3'), (8, 'S3'), (9, 'S4'), (10, 'S4'), (11, 'S5'), (12, 'S5'), (13, 'S6'), (14, 'S6'), (15, 'S7'), (16, 'S7'), (17, 'S8'), (18, 'S8'), (19, 'S9'), (20, 'S10')]",
        "[(1, 'E0'), (2, 'E1'), (3, 'E2'), (4, 'E3'), (5, 'E4'), (6, 'E5'), (7, 'E6'), (8, 'S7'), (9, 'S8'), (10, 'S9'), (11, 'S10'), (12, 'S10'), (13, 'S10')]",
        "[(1, 'S0'), (2, 'S0'), (3, 'S0'), (4, 'S0'), (5, 'S3'), (6, 'S3'), (7, 'S3'), (8, 'S3'), (9, 'S6'), (10, 'S6'), (11, 'S6'), (12, 'S6'), (13, 'S9'), (14, 'S9'), (15, 'S9'), (16, 'S9')]",
        "[(1, 'E0'), (2, 'E1'), (3, 'E2'), (4, 'E3'), (5, 'E4'), (6, 'E5'), (7, 'E6'), (8, 'E7'), (9, 'E8'), (10, 'E9'), (11, 'E10'), (12, 'E10')]",
        "[(1, 'E10'), (2, 'E9'), (3, 'E8'), (4, 'E7'), (5, 'E6'), (6, 'E5'), (7, 'E4'), (8, 'E3'), (9, 'E2'), (10, 'E1'), (11, 'E0')]",
    )
)

SOURCES: Sequence[str] = (
    "yxan.csv",
    "short.csv",
    "parimal.csv",
    "naz.csv",
)

FIELDNAMES: Sequence[str] = ("Input List", "Area AVG", "RG AVG", "RDF Peak")


def parse_numeric_fields(row: Dict[str, str]) -> Optional[Dict[str, float]]:
    """Return the numeric fields of interest if they can be parsed."""
    try:
        area = float(row["Area AVG"])
        rg = float(row["RG AVG"])
        rdf = float(row["RDF Peak"])
    except (KeyError, TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in (area, rg, rdf)):
        return None

    return {"Area AVG": area, "RG AVG": rg, "RDF Peak": rdf}


def load_source_rows(filename: str) -> List[Dict[str, str]]:
    """Yield formatted rows from a single source CSV."""
    source_path = DATA_DIR / filename
    rows: List[Dict[str, str]] = []

    with source_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for raw_row in reader:
            input_list = raw_row.get("Input List")
            if input_list in EXCLUDED_INPUT_LISTS:
                continue

            numeric = parse_numeric_fields(raw_row)
            if not numeric:
                continue

            rows.append(
                {
                    "Input List": input_list,
                    "Area AVG": f"{numeric['Area AVG']:.8f}",
                    "RG AVG": f"{numeric['RG AVG']:.8f}",
                    "RDF Peak": f"{numeric['RDF Peak']:.8f}",
                }
            )
    return rows


def build_combined_rows() -> List[Dict[str, str]]:
    combined: List[Dict[str, str]] = []
    for source in SOURCES:
        combined.extend(load_source_rows(source))
    return combined


def write_combined(rows: Sequence[Dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    rows = build_combined_rows()
    write_combined(rows)


if __name__ == "__main__":
    main()
