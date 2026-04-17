# BBmodels

Benchmarking suite for predicting bottlebrush polymer properties (Area, Rg, RDF Peak) from graft-chain sequences. Includes linear, MLP, CNN, and graph neural network (GCN, GAT, GIN) baselines, plus a surrogate-only perturbation study.

## Repository layout

- [CNN/](CNN/), [GAT/](GAT/), [GCN/](GCN/), [GIN/](GIN/), [LR/](LR/), [MLP/](MLP/) — per-model `preprocess.py`, `train.py`, `predict.py`, plus `models/`, `processed/`, `results/`, and `predictions.csv` outputs.
- [data/](data/) — datasets: `ES.csv` (base), `extreme_test_set_{1,2}.csv`, and `ES_plus_extreme_{1,2}.csv` variants; `test_set.csv`.
- [splits/](splits/) — precomputed 5-fold CV split indices (seed 42) for each dataset variant.
- [utils/](utils/) — shared helpers, including [splits.py](utils/splits.py) (`load_splits`).
- [scripts/](scripts/) — [analysis/](scripts/analysis/) (`cv_performance.py`, `test_performance.py`, `plot_bars.py`) and [optimization/](scripts/optimization/) (`optimization.py`, `run_extreme_optimization.sh`).
- [PERTURB/](PERTURB/) — GIN-surrogate perturbation study; see [PERTURB/README.md](PERTURB/README.md).
- [simulations/](simulations/) — NAMD 2.14 binaries, input decks, and simulation scripts used to generate ground-truth data.
- [FIGURES/](FIGURES/), [RESULTS_RANDOM_ONLY/](RESULTS_RANDOM_ONLY/), [RESULTS_EXTREME_1/](RESULTS_EXTREME_1/), [RESULTS_EXTREME_2/](RESULTS_EXTREME_2/) — generated figures and results for each experimental regime.

## Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

Core dependencies: PyTorch, PyTorch Geometric, scikit-learn, UMAP, CMA, pandas, matplotlib, seaborn.

## Typical workflow

Run from the repo root so package-relative imports resolve.

```bash
# Preprocess + train + predict for a given model (example: GIN)
./venv/bin/python -m GIN.preprocess
./venv/bin/python -m GIN.train
./venv/bin/python -m GIN.predict

# Aggregate CV / test metrics across models
./venv/bin/python scripts/analysis/cv_performance.py
./venv/bin/python scripts/analysis/test_performance.py
./venv/bin/python scripts/analysis/plot_bars.py

# CMA-ES optimization over an architecture space
./venv/bin/python scripts/optimization/optimization.py
```

For the perturbation study, see [PERTURB/README.md](PERTURB/README.md).

## Data format

Each row in [data/ES.csv](data/ES.csv) is an architecture: `Input List` is a list of `(site, label)` pairs where labels are `E{len}` (PEO graft) or `S{len}` (PS graft), `0` meaning ungrafted. Targets are `Area AVG`, `RG AVG`, and `RDF Peak`.
