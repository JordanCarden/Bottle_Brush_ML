#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from .arch import (
    Token,
    blockiness,
    coefficient_of_variation,
    centroid_separation,
    end_loading,
    flip_chemistry_greedy,
    gini_coefficient,
    gini_coefficient_all_sites,
    has_peo,
    length_weighted_peo_fraction,
    n_transitions,
    parse_architecture,
    ratio_sum_lengths,
    redistribute_lengths_within_chemistry,
    remove_grafts,
    scale_peo_lengths,
    scale_ps_lengths,
    sequence_shuffle,
    serialize_architecture,
    sigma_grafting,
    signed_lengths,
    total_length,
)
from .gin_surrogate import GinSurrogate
from .ood import OODScorer


def _parse_float_list(text: str) -> List[float]:
    raw = (text or "").strip()
    if not raw:
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perturbation-based feature analysis using trained GIN (no new MD)."
    )
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=Path("data/ES.csv"),
        help="Base architecture CSV (default: data/ES.csv).",
    )
    parser.add_argument(
        "--input-col",
        type=str,
        default="Input List",
        help="Column containing architecture strings (default: 'Input List').",
    )
    parser.add_argument(
        "--n-bases",
        type=int,
        default=0,
        help="Number of base architectures to sample (0 = all; default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling/replicates (default: 0).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("PERTURB") / "perturb_long.csv",
        help="Output CSV path (default: PERTURB/perturb_long.csv).",
    )

    parser.add_argument(
        "--composition-targets",
        type=str,
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
        help="Target length-weighted PEO fractions for chemistry flips.",
    )
    parser.add_argument(
        "--sequence-shuffle-reps",
        type=int,
        default=50,
        help="Random shuffle replicates per base (default: 50).",
    )
    parser.add_argument(
        "--sigma-targets",
        type=str,
        default="1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2",
        help="Target grafting densities σ for removal study.",
    )
    parser.add_argument(
        "--peo-scale-factors",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5",
        help="Multipliers applied to Eℓ lengths (PS fixed).",
    )
    parser.add_argument(
        "--ps-scale-factors",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5",
        help="Multipliers applied to Sℓ lengths (PEO fixed).",
    )
    parser.add_argument(
        "--dispersity-alphas",
        type=str,
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
        help="Alpha in [0,1] mixing uniform→max-dispersed length redistribution.",
    )

    parser.add_argument(
        "--ood-pca-components",
        type=int,
        default=10,
        help="PCA components for OOD scoring (default: 10; 0 disables PCA).",
    )
    parser.add_argument(
        "--ood-percentile",
        type=float,
        default=99.0,
        help="Percentile threshold for OOD flag (default: 99).",
    )
    parser.add_argument(
        "--gin-batch-size",
        type=int,
        default=256,
        help="Batch size for GIN predictions (default: 256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g., 'cpu' or 'cuda').",
    )
    return parser.parse_args(argv)


def _metrics(tokens: Sequence[Token]) -> dict[str, float]:
    values = signed_lengths(tokens)
    total = total_length(tokens)
    total_e = total_length(tokens, chem="E")
    total_s = total_length(tokens, chem="S")
    return {
        "B": float(blockiness(tokens)),
        "fPEO_w": float(length_weighted_peo_fraction(tokens)),
        "sigma": float(sigma_grafting(tokens)),
        "ratio_ES": float(ratio_sum_lengths(tokens)),
        "L_tot": float(total),
        "L_E": float(total_e),
        "L_S": float(total_s),
        "n_trans": float(n_transitions(tokens)),
        "Eload_tot": float(end_loading(tokens, k=3, chem=None)),
        "Eload_E": float(end_loading(tokens, k=3, chem="E")),
        "Eload_S": float(end_loading(tokens, k=3, chem="S")),
        "cent_sep": float(centroid_separation(tokens)),
        "gini": float(gini_coefficient(values)),
        "gini_all": float(gini_coefficient_all_sites(values)),
        "cv": float(coefficient_of_variation(values)),
        "has_PEO": float(has_peo(tokens)),
    }


def _add_record(
    records: List[dict],
    *,
    base_id: int,
    knob: str,
    knob_value: float,
    pattern: str,
    replicate: int,
    tokens: Sequence[Token],
    pct_clipped: float | None = None,
    extra: dict | None = None,
) -> None:
    arch_variant = serialize_architecture(tokens)
    metrics = _metrics(tokens)
    record = {
        "base_id": int(base_id),
        "knob": str(knob),
        "knob_value": float(knob_value),
        "pattern": str(pattern),
        "replicate": int(replicate),
        "arch_variant": arch_variant,
        "pct_clipped": float(pct_clipped) if pct_clipped is not None else math.nan,
        **metrics,
    }
    if extra:
        record.update(extra)
    records.append(record)


def _sample_bases(df: pd.DataFrame, n_bases: int, seed: int) -> pd.DataFrame:
    if n_bases <= 0 or n_bases >= len(df):
        return df.copy()
    return df.sample(n=n_bases, random_state=seed).copy()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    base_df = pd.read_csv(args.base_csv)
    if args.input_col not in base_df.columns:
        raise SystemExit(f"Missing input column '{args.input_col}' in {args.base_csv}")
    base_df = base_df.reset_index(drop=False).rename(columns={"index": "base_row"})
    base_df = _sample_bases(base_df, int(args.n_bases), int(args.seed)).reset_index(drop=True)

    composition_targets = _parse_float_list(args.composition_targets)
    sigma_targets = _parse_float_list(args.sigma_targets)
    peo_scale_factors = _parse_float_list(args.peo_scale_factors)
    ps_scale_factors = _parse_float_list(args.ps_scale_factors)
    dispersity_alphas = _parse_float_list(args.dispersity_alphas)

    records: List[dict] = []
    for base_seq, row in base_df.iterrows():
        base_id = int(row["base_row"])
        base_arch = str(row[args.input_col])
        base_tokens = parse_architecture(base_arch)
        base_tokens = sorted(base_tokens, key=lambda t: t.position)

        _add_record(
            records,
            base_id=base_id,
            knob="baseline",
            knob_value=0.0,
            pattern="",
            replicate=0,
            tokens=base_tokens,
            pct_clipped=None,
            extra={"base_seq": int(base_seq)},
        )

        for target in composition_targets:
            edited, flips = flip_chemistry_greedy(base_tokens, float(target))
            _add_record(
                records,
                base_id=base_id,
                knob="composition",
                knob_value=float(target),
                pattern="greedy_flip",
                replicate=0,
                tokens=edited,
                pct_clipped=None,
                extra={"n_flips": int(flips), "base_seq": int(base_seq)},
            )

        nonzero_sites = sum(1 for tok in base_tokens if tok.length > 0)
        if nonzero_sites >= 2:
            for rep in range(int(args.sequence_shuffle_reps)):
                rng = random.Random(int(args.seed) + int(base_seq) * 100_000 + 10_000 + rep)
                shuffled = sequence_shuffle(base_tokens, rng)
                _add_record(
                    records,
                    base_id=base_id,
                    knob="sequence",
                    knob_value=float(blockiness(shuffled)),
                    pattern="shuffle",
                    replicate=int(rep),
                    tokens=shuffled,
                    pct_clipped=None,
                    extra={"base_seq": int(base_seq)},
                )

        backbone_len = len(base_tokens)
        grafted = sum(1 for tok in base_tokens if tok.length > 0)
        for sigma_target in sigma_targets:
            target_grafts = int(round(float(sigma_target) * backbone_len))
            if target_grafts > grafted:
                continue
            n_remove = grafted - target_grafts
            periodic, _ = remove_grafts(base_tokens, n_remove, pattern="periodic")
            _add_record(
                records,
                base_id=base_id,
                knob="grafting",
                knob_value=float(sigma_target),
                pattern="periodic",
                replicate=0,
                tokens=periodic,
                pct_clipped=None,
                extra={"n_remove": int(n_remove), "base_seq": int(base_seq)},
            )

        for factor in peo_scale_factors:
            scaled, pct = scale_peo_lengths(base_tokens, float(factor))
            _add_record(
                records,
                base_id=base_id,
                knob="peo_scale",
                knob_value=float(factor),
                pattern="scale_peo",
                replicate=0,
                tokens=scaled,
                pct_clipped=float(pct),
                extra={"base_seq": int(base_seq)},
            )

        for factor in ps_scale_factors:
            scaled, pct = scale_ps_lengths(base_tokens, float(factor))
            _add_record(
                records,
                base_id=base_id,
                knob="ps_scale",
                knob_value=float(factor),
                pattern="scale_ps",
                replicate=0,
                tokens=scaled,
                pct_clipped=float(pct),
                extra={"base_seq": int(base_seq)},
            )

        for alpha in dispersity_alphas:
            redistributed = redistribute_lengths_within_chemistry(base_tokens, alpha=float(alpha))
            _add_record(
                records,
                base_id=base_id,
                knob="dispersity",
                knob_value=float(alpha),
                pattern="alpha",
                replicate=0,
                tokens=redistributed,
                pct_clipped=None,
                extra={"base_seq": int(base_seq)},
            )

    df_long = pd.DataFrame.from_records(records)
    unique_arch = df_long["arch_variant"].drop_duplicates().tolist()

    gin = GinSurrogate(repo_root=repo_root, device=args.device)
    preds = gin.predict_many(unique_arch, batch_size=int(args.gin_batch_size))
    pred_map = {arch: pred for arch, pred in zip(unique_arch, preds, strict=True)}

    ood = OODScorer(
        repo_root=repo_root,
        pca_components=int(args.ood_pca_components),
        percentile_threshold=float(args.ood_percentile),
    )
    ood_results = ood.score_many(unique_arch)
    ood_map = {arch: res for arch, res in zip(unique_arch, ood_results, strict=True)}

    df_long["A_pred"] = df_long["arch_variant"].map(lambda a: pred_map[a].area)
    df_long["Rg_pred"] = df_long["arch_variant"].map(lambda a: pred_map[a].rg)
    df_long["RDF_pred"] = df_long["arch_variant"].map(lambda a: pred_map[a].rdf)

    df_long["ood_score"] = df_long["arch_variant"].map(lambda a: ood_map[a].score)
    df_long["ood_percentile"] = df_long["arch_variant"].map(lambda a: ood_map[a].percentile)
    df_long["ood_flag"] = df_long["arch_variant"].map(lambda a: ood_map[a].flag)

    base_preds = (
        df_long.loc[df_long["knob"] == "baseline", ["base_id", "A_pred", "Rg_pred", "RDF_pred"]]
        .drop_duplicates(subset=["base_id"])
        .rename(
            columns={
                "A_pred": "base_A_pred",
                "Rg_pred": "base_Rg_pred",
                "RDF_pred": "base_RDF_pred",
            }
        )
    )
    df_long = df_long.merge(base_preds, on="base_id", how="left")
    df_long["dA"] = df_long["A_pred"] - df_long["base_A_pred"]
    df_long["dRg"] = df_long["Rg_pred"] - df_long["base_Rg_pred"]
    df_long["dRDF"] = df_long["RDF_pred"] - df_long["base_RDF_pred"]
    df_long = df_long.drop(columns=["base_A_pred", "base_Rg_pred", "base_RDF_pred"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(out_path, index=False)
    print(f"Wrote {len(df_long)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
