# ORACLE-VARX

**OR**thogonalized **A**daptive **C**ausal **L**ag **E**stimation for **V**ector **A**uto**R**egression with e**X**ogenous variables.

A machine learning framework for financial time series forecasting that combines Vector Autoregression (VAR) with Double Machine Learning (DML) to produce orthogonalized, confounding-robust forecasts.

## Quick Start

Install dependencies:

```bash
pip install -r requirements-cpu.txt   # CPU (tree-based learners)
pip install -r requirements-gpu.txt   # GPU (adds TabPFN + CUDA)
```

## Experiments

The two main experiment scripts are:

### 1. Real-Data Experiment (`scripts/run_combined_experiment.py`)

Runs VAR-family methods on 9 SPDR sector ETFs with macro confounder presets (`vix`, `macro5`, `all10`). Supports OLS baselines, DML methods with tree-based learners, and backtesting with PnL evaluation.

```bash
# OLS baselines (no confounders)
python scripts/run_combined_experiment.py --no-confounders --no-show

# DML methods with LightGBM first-stage
python scripts/run_combined_experiment.py --confounders vix --learner lgbm --no-show

# Quick smoke test
python scripts/run_combined_experiment.py --confounders vix --learner lgbm --n-days 1500 --no-show
```

Results are saved to `results/`.

### 2. Toy Benchmark (`scripts/run_toy_benchmark.py`)

Runs a controlled synthetic benchmark with known ground-truth coefficients, 3 regime changes, and 3 confounder observability levels (`all`, `partial_2`, `partial_1`). Evaluates coefficient recovery, edge detection, and forecast accuracy.

```bash
# Phase 0: OLS baselines (~6s)
python scripts/run_toy_benchmark.py --phase 0 --no-show

# Phase 1: DML methods (tree-based)
python scripts/run_toy_benchmark.py --phase 1 --learner lgbm --obs all --no-show

# Phase 2: TabPFN methods (GPU)
python scripts/run_toy_benchmark.py --phase 2 --device cuda --no-show
```

Results are saved to `results-toy/`. Each run's `metrics.json` includes the edge FDR, power and F1 of
stage (iii) BH edge discovery (level q = 0.05). To add these metrics to runs saved before they existed,
without refitting (older VARX / ACLE-VARX runs did not keep their SEs; rerun phase 0 for those, which
takes seconds):

```bash
python scripts/eval_bh_edges.py
```

### 3. Causal-Discovery Baselines (`scripts/run_cd_baselines.py`, `scripts/time_cd_baselines.py`)

Rolling-window PCMCI (tigramite, ParCorr test) and VAR-LiNGAM (lingam) on the toy benchmark, refit
on every output day and scored with the toy benchmark's own metrics (edge FDR/power/F1 through the
same BH step, lag recovery, and for VAR-LiNGAM also coefficients and forecasts). The timing script
measures seconds per window on the toy data and on the ETF data with the `all10` confounders.

```bash
pip install -r requirements-cd.txt

# Accuracy: both methods at the 4 observability levels (a few minutes on 8 cores)
python scripts/run_cd_baselines.py

# Timing: seconds per window (CMIknn takes minutes to hours per window; cap it)
python scripts/time_cd_baselines.py --dataset toy --method parcorr --n-windows 3
python scripts/time_cd_baselines.py --dataset etf --method cmiknn --n-windows 1 --max-seconds 3600
```

Results are saved to `results-toy/cd-baselines/` (accuracy) and `results/cd-timing/timing.jsonl`
(timing).

## Further Reading

For detailed execution guides, CLI references, confounder presets, output structure, and EC2/GPU setup instructions, see `docs/`:

- [`docs/run-experiment-guide.md`](docs/run-experiment-guide.md) -- full guide for the real-data experiment
- [`docs/run-toy-experiment-guide.md`](docs/run-toy-experiment-guide.md) -- full guide for the toy benchmark
