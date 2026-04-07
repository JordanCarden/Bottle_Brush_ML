#!/usr/bin/env python3
"""Compute mean per-knob max-min effects and plot the retained overlay figure.

Definition (per the perturbation study writeup):
  - Filter to in-distribution perturbations only (ood_flag == 0) unless --include-ood.
  - For each base architecture (base_id) and each knob:
      1) Aggregate predictions at each knob_value via the median.
      2) Define a signed max–min effect Δ = pred(max knob_value) - pred(min knob_value).
  - Report the mean effect across bases for each knob and each target property.

Outputs:
  - An optional CSV summary table
  - A 3-panel centered horizontal bar chart PNG with boxplot-caps overlays
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator  # noqa: E402


DEFAULT_INPUT_CANDIDATES = [
    Path("PERTURB/perturb_long_mega.csv.gz"),
    Path("PERTURB/perturb_long_mega.csv"),
    Path("PERTURB/perturb_long.csv.gz"),
    Path("PERTURB/perturb_long.csv"),
]


REQUIRED_COLUMNS = [
    "base_id",
    "knob",
    "knob_value",
    "A_pred",
    "Rg_pred",
    "RDF_pred",
    "ood_flag",
]


KNOB_ORDER = ["composition", "grafting", "peo_scale", "ps_scale", "sequence", "dispersity"]

KNOB_LABELS = {
    "composition": r"Composition ($c_E$)",
    "grafting": r"Grafting density ($f$)",
    "peo_scale": r"PEO length scale ($\chi_E$)",
    "ps_scale": r"PS length scale ($\chi_S$)",
    "sequence": r"Sequence (blockiness $l_B$)",
    "dispersity": "Length dispersity (Ð)",
}

def _force_x_minor_ticks_inside(ax: plt.Axes) -> None:
    # Matplotlib draws tick lines (zorder~2) under spines (default zorder~2.5). With thick
    # spines and short minor ticks, the inward part can be visually lost, making minor ticks
    # look like they point outward. Force the x minor ticks to be clearly inside and on top.
    ax.tick_params(axis="x", which="minor", direction="in", bottom=True, top=False)
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_marker(2)  # tick-up for bottom axis
        tick.tick1line.set_zorder(10)
        tick.tick1line.set_clip_on(False)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_marker(2)  # tick-up for bottom axis
        tick.tick1line.set_zorder(10)
        tick.tick1line.set_clip_on(False)


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Nimbus Sans",
            "mathtext.it": "Nimbus Sans",
            "mathtext.bf": "Nimbus Sans",
            "mathtext.sf": "Nimbus Sans",
            "mathtext.tt": "Nimbus Sans",
            "mathtext.cal": "Nimbus Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean-width centered bar plot with boxplot-caps overlays from perturbation CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV or CSV.GZ (default: auto-detect in PERTURB/).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("PERTURB/figs_mean_width"),
        help="Output directory (default: PERTURB/figs_mean_width).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for PNG output (default: 600).",
    )
    parser.add_argument(
        "--include-ood",
        action="store_true",
        help="Include OOD rows (disables ood_flag==0 filter).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="mean_width_centered_bars__overlay-boxplot_caps.png",
        help=(
            "Output image filename inside --outdir "
            "(default: mean_width_centered_bars__overlay-boxplot_caps.png)."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional CSV filename inside --outdir for the summary table (default: do not write one).",
    )
    return parser.parse_args(argv)


def resolve_input_path(user_path: Optional[Path]) -> Path:
    if user_path is not None:
        if not user_path.exists():
            raise FileNotFoundError(f"Input not found: {user_path}")
        return user_path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No input file found. Tried:\n  - " + "\n  - ".join(str(p) for p in DEFAULT_INPUT_CANDIDATES)
    )


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_widths(df: pd.DataFrame, *, include_ood: bool) -> pd.DataFrame:
    work = df.copy()
    work = work.rename(columns=lambda c: c.strip())
    work["knob"] = work["knob"].astype(str).str.strip()
    work = work.loc[work["knob"].isin(KNOB_ORDER), :]

    if not include_ood:
        work = work.loc[work["ood_flag"] == 0, :]

    work["knob_value"] = pd.to_numeric(work["knob_value"], errors="coerce")
    work = work.dropna(subset=["base_id", "knob", "knob_value", "A_pred", "Rg_pred", "RDF_pred"])

    # 1) Median aggregate at each knob_value (handles repeated shuffles/replicates).
    med = (
        work.groupby(["base_id", "knob", "knob_value"], as_index=False)
        .agg(A_pred=("A_pred", "median"), Rg_pred=("Rg_pred", "median"), RDF_pred=("RDF_pred", "median"))
        .sort_values(["base_id", "knob", "knob_value"], kind="mergesort")
    )

    # 2) Per-base widths = pred(max knob_value) - pred(min knob_value).
    width_rows = []
    for (base_id, knob), g in med.groupby(["base_id", "knob"], sort=False):
        if g["knob_value"].nunique() < 2:
            continue
        g = g.sort_values("knob_value", kind="mergesort")
        min_row = g.iloc[0]
        max_row = g.iloc[-1]
        width_rows.append(
            {
                "base_id": int(base_id),
                "knob": str(knob),
                "width_A": float(max_row["A_pred"] - min_row["A_pred"]),
                "width_Rg": float(max_row["Rg_pred"] - min_row["Rg_pred"]),
                "width_RDF": float(max_row["RDF_pred"] - min_row["RDF_pred"]),
            }
        )

    widths = pd.DataFrame(width_rows)
    if widths.empty:
        raise ValueError("No widths computed (check filters: ood_flag/knob_value).")
    return widths


def compute_mean_widths(widths: pd.DataFrame) -> pd.DataFrame:
    if widths.empty:
        raise ValueError("Empty widths input.")

    summary = (
        widths.groupby("knob", as_index=False)
        .agg(
            **{
                "Mean width A": ("width_A", "mean"),
                "Mean width Rg": ("width_Rg", "mean"),
                "Mean width RDF": ("width_RDF", "mean"),
                "n_bases": ("base_id", "nunique"),
            }
        )
        .set_index("knob")
    )

    ordered = []
    for knob in KNOB_ORDER:
        if knob not in summary.index:
            raise ValueError(f"Knob missing after filtering: {knob}")
        row = summary.loc[knob].to_dict()
        row["Feature"] = KNOB_LABELS.get(knob, knob)
        row["knob"] = knob
        ordered.append(row)

    out = pd.DataFrame(ordered)[["Feature", "knob", "Mean width A", "Mean width Rg", "Mean width RDF", "n_bases"]]
    return out


def _overlay_boxplot_caps(ax: plt.Axes, data: list[np.ndarray], positions: np.ndarray) -> None:
    bp = ax.boxplot(
        data,
        positions=positions,
        vert=False,
        widths=0.32,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        showcaps=True,
    )
    for box in bp.get("boxes", []):
        # Solid black boxes without an outline match the kept perturbation figure.
        box.set(edgecolor="none", facecolor="black", linewidth=0.0, zorder=6)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp.get(key, []):
            artist.set(color="black", linewidth=1.0 if key != "medians" else 1.2, zorder=7)


def plot_centered_bars(summary: pd.DataFrame, widths: pd.DataFrame, *, outpath: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.45), sharey=True)
    targets = ["Mean width A", "Mean width Rg", "Mean width RDF"]
    width_col_map = {"Mean width A": "width_A", "Mean width Rg": "width_Rg", "Mean width RDF": "width_RDF"}
    xlabel_map = {
        "Mean width A": r"$\Delta$ Area ($\times 10^3$)",
        "Mean width Rg": r"$\Delta R_g$",
        "Mean width RDF": r"$\Delta$ RDF Peak",
    }

    y_vals = summary["Feature"].tolist()
    knobs = summary["knob"].tolist()
    y_pos = np.arange(len(y_vals), dtype=float)
    for i, target in enumerate(targets):
        ax = axes[i]
        x_vals = summary[target].astype(float)

        width_col = width_col_map[target]
        dist_vals = widths[width_col].astype(float).abs().values if not widths.empty else np.array([])
        dist_scale = float(np.nanquantile(dist_vals, 0.95)) if dist_vals.size else 0.0
        mean_max = float(np.nanmax(np.abs(x_vals.values))) if len(x_vals) else 0.0
        max_abs = max(dist_scale, mean_max)
        limit = 1.0 if not np.isfinite(max_abs) or max_abs <= 0 else max_abs * 1.15

        colors = ["#d62728" if x < 0 else "#1f77b4" for x in x_vals]
        ax.barh(y_pos, x_vals, color=colors, edgecolor="black", linewidth=0.5, height=0.72, zorder=1)

        boxplot_data: list[np.ndarray] = []
        for knob in knobs:
            vals = widths.loc[widths["knob"] == knob, width_col].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            boxplot_data.append(vals)
        _overlay_boxplot_caps(ax, boxplot_data, y_pos)

        ax.set_yticks(y_pos)
        if i == 0:
            ax.set_yticklabels(y_vals)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(-limit, limit)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel(xlabel_map.get(target, "Mean Width"))
        ax.tick_params(direction="in", length=3, width=1.0, top=False, right=False)
        ax.tick_params(axis="y", length=0, width=0)
        if target == "Mean width A":
            ax.set_xlim(-15000, 15000)
            ax.set_xticks([-10000, -5000, 0, 5000, 10000])
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis="x", which="minor", direction="in", length=2, width=0.8, bottom=True, top=False)
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, pos: "0"
                    if abs(x) < 1e-9
                    else str(int(round(x / 1000.0)))
                    if abs(x / 1000.0 - round(x / 1000.0)) < 1e-9
                    else f"{x/1000.0:g}"
                )
            )
        elif target == "Mean width Rg":
            ax.set_xlim(-7.5, 7.5)
            ax.set_xticks([-5.0, -2.5, 0.0, 2.5, 5.0])
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis="x", which="minor", direction="in", length=2, width=0.8, bottom=True, top=False)
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, pos: "0"
                    if abs(x) < 1e-9
                    else str(int(round(x)))
                    if abs(x - round(x)) < 1e-9
                    else f"{x:g}"
                )
            )
        elif target == "Mean width RDF":
            ax.set_xlim(-1.5, 1.5)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis="x", which="minor", direction="in", length=2, width=0.8, bottom=True, top=False)
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, pos: "0"
                    if abs(x) < 1e-9
                    else str(int(round(x)))
                    if abs(x - round(x)) < 1e-9
                    else f"{x:g}"
                )
            )
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5], symmetric=True))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis="x", which="minor", direction="in", length=2, width=0.8, bottom=True, top=False)
        _force_x_minor_ticks_inside(ax)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    fig.tight_layout(pad=0.115)
    # Re-introduce a small visual gutter between panels (tight_layout can collapse it).
    fig.subplots_adjust(wspace=0.08)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    input_path = resolve_input_path(args.input)
    df = pd.read_csv(input_path)
    validate_columns(df, REQUIRED_COLUMNS)

    apply_publication_style()
    widths = compute_widths(df, include_ood=bool(args.include_ood))
    summary = compute_mean_widths(widths)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = None
    if args.summary_csv:
        csv_path = outdir / str(args.summary_csv)
        summary.to_csv(csv_path, index=False)

    fig_path = outdir / str(args.outfile)
    plot_centered_bars(summary, widths, outpath=fig_path, dpi=int(args.dpi))

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    if csv_path is not None:
        print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")
    print("\nSummary:")
    print(summary[["Feature", "Mean width A", "Mean width Rg", "Mean width RDF", "n_bases"]].to_string(index=False))


if __name__ == "__main__":
    main()
