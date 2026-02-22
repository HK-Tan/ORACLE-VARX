# Experiment Execution Guide

## Overview

This guide covers running the full method comparison suite:
- 16 runs across 3 confounder configs (vix, macro5, all10) and 4 tree learners + TabPFN
- 35 unique model outputs: VAR(1) + ACLE-VAR(1) + VARX(3) + ACLE-VARX(3) + OR-VARX(12) + ORACLE-VARX(12) + ORACLE-VARX-TabPFN(3)

## Confounder Configurations

| Config | Variables | Data starts | First DML forecast ~ | Eval years |
|--------|-----------|-------------|---------------------|------------|
| vix | VIX | ~2000 | ~2004 | ~17 yrs |
| macro5 | VIX, DFF, T5YIE, DCOILWTICO, USEPUINDXD | ~2000 | ~2004 | ~17 yrs |
| all10 | All 10 confounders | ~2000 | ~2004 | ~17 yrs |

> **Note**: Leading NaN confounders (from late-starting series like GVZCLS, mid-2008) are backfilled with 0.0 (log-return of 0 = "no change"), preserving all asset data from ~2000 onward. This means `macro5` and `all10` runs will take longer than previously estimated due to more data points.

## Scripts

### `run_combined_experiment.py` — Single (config, learner) pair

Runs up to 4 methods with amortized DML first stage:

```bash
# Run all methods for VIX / lgbm
python scripts/run_combined_experiment.py --confounders vix --learner lgbm --no-show --verbose

# VAR + ACLE-VAR baseline (no confounders)
python scripts/run_combined_experiment.py --no-confounders --no-show

# Quick smoke test with limited data
python scripts/run_combined_experiment.py --confounders vix --learner lgbm --n-days 1500 --no-show
```

Key options:
- `--confounders`: preset name (`vix`, `macro5`, `all10`) or comma-separated (`VIX,DFF`)
- `--learner`: `lgbm`, `xgboost`, `rf`, `extra_trees`
- `--n-jobs`: CPU cores per method (auto-computed if not specified)
- `--no-confounders`: run VAR + ACLE-VAR baseline

### `run_all_experiments.py` — tmux orchestrator

Automates all 16 runs across 4 phases:

```bash
# Run all phases
python scripts/run_all_experiments.py --phase all --verbose

# Run one phase
python scripts/run_all_experiments.py --phase 1 --verbose

# Dry run to see commands
python scripts/run_all_experiments.py --phase all --dry-run
```

## EC2 Setup

```bash
# 1. Launch c7i.4xlarge spot (Ubuntu 22.04, 30GB gp3)
# 2. SSH in and run setup:
curl -sSL https://raw.githubusercontent.com/HK-Tan/ORACLE-VARX/main/scripts/setup-ec2.sh | bash
cd ORACLE-VARX && source .venv/bin/activate
```

## Phase Execution

### Phase 0: Baseline (sequential)

```bash
tmux new -s phase0
python scripts/run_combined_experiment.py --no-confounders --no-show --verbose
# ~25 min
```

### Phase 1: VIX x 4 learners (parallel)

```bash
tmux new-session -d -s vix
tmux split-window -h -t vix
tmux split-window -v -t vix:0.0
tmux split-window -v -t vix:0.1
tmux send-keys -t vix:0.0 'python scripts/run_combined_experiment.py --confounders vix --learner lgbm --no-show --verbose' Enter
tmux send-keys -t vix:0.1 'python scripts/run_combined_experiment.py --confounders vix --learner xgboost --no-show --verbose' Enter
tmux send-keys -t vix:0.2 'python scripts/run_combined_experiment.py --confounders vix --learner rf --no-show --verbose' Enter
tmux send-keys -t vix:0.3 'python scripts/run_combined_experiment.py --confounders vix --learner extra_trees --no-show --verbose' Enter
tmux attach -t vix
```

Wait for all to finish, then repeat for macro5 (Phase 2) and all10 (Phase 3).

Or use the orchestrator:
```bash
python scripts/run_all_experiments.py --phase all --verbose
```

### Core Allocation

| Instance | vCPU | Physical cores | Auto n_jobs/pane | Total used |
|----------|------|---------------|------------------:|----------:|
| c7i.4xlarge | 16 | 8 | 1 | 4 of 8 |
| c7i.8xlarge | 32 | 16 | 3 | 12 of 16 |
| c7i.12xlarge | 48 | 24 | 5 | 20 of 24 |

Override with `--n-jobs N` when needed.

## Run Table

| Run | Config | Learner | Methods | Est. Time |
|-----|--------|---------|---------|-----------|
| 0 | none | OLS | VAR + ACLE-VAR | ~25 min |
| 1 | vix | lgbm | VARX + ACLE-VARX + OR-VARX + ORACLE-VARX | ~25 min |
| 2 | vix | xgboost | OR-VARX + ORACLE-VARX (VARX/ACLE skipped) | ~22 min |
| 3 | vix | rf | OR-VARX + ORACLE-VARX | ~27 min |
| 4 | vix | extra_trees | OR-VARX + ORACLE-VARX | ~22 min |
| 5 | vix | TabPFN | ORACLE-VARX TabPFN | ~35 min (GPU) |
| 6 | macro5 | lgbm | VARX + ACLE-VARX + OR-VARX + ORACLE-VARX | ~30 min |
| 7 | macro5 | xgboost | OR-VARX + ORACLE-VARX | ~28 min |
| 8 | macro5 | rf | OR-VARX + ORACLE-VARX | ~32 min |
| 9 | macro5 | extra_trees | OR-VARX + ORACLE-VARX | ~28 min |
| 10 | macro5 | TabPFN | ORACLE-VARX TabPFN | ~40 min (GPU) |
| 11 | all10 | lgbm | VARX + ACLE-VARX + OR-VARX + ORACLE-VARX | ~35 min |
| 12 | all10 | xgboost | OR-VARX + ORACLE-VARX | ~32 min |
| 13 | all10 | rf | OR-VARX + ORACLE-VARX | ~37 min |
| 14 | all10 | extra_trees | OR-VARX + ORACLE-VARX | ~32 min |
| 15 | all10 | TabPFN | ORACLE-VARX TabPFN | ~45 min (GPU) |

**VARX/ACLE-VARX** are OLS-only (no learner dependency) — runs 2-4, 7-9, 12-14 skip them since run 1/6/11 already computed them.

## Wall-Clock Timeline (c7i.4xlarge)

| Phase | What runs | Wall time |
|-------|-----------|-----------|
| Phase 0 | VAR + ACLE-VAR (sequential) | ~25 min |
| Phase 1 | VIX x 4 learners (4 tmux panes) | ~27 min |
| Phase 2 | macro5 x 4 learners (4 tmux panes) | ~32 min |
| Phase 3 | all10 x 4 learners (4 tmux panes) | ~37 min |
| **Total CPU** | | **~2 hours** |

GPU (A100): Runs 5, 10, 15 sequential ~2 hours.

## Copying Results

From EC2 to local:
```bash
scp -i /home/hktan/AWS_Keys/EC2-Keys.pem -r \
    ubuntu@<EC2-PUBLIC-IP>:~/ORACLE-VARX/results/* \
    /home/hktan/ORACLE-VARX/results/
```

Or with rsync (resume-capable):
```bash
rsync -avz -e "ssh -i /home/hktan/AWS_Keys/EC2-Keys.pem" \
    ubuntu@<EC2-PUBLIC-IP>:~/ORACLE-VARX/results/ \
    /home/hktan/ORACLE-VARX/results/
```

## Output Structure

Each method produces results in `results/<method>/<experiment_name>/`:
- `results.pt` — model results (forecasts, coefficients, etc.)
- `pnl_results.pkl` — PnL data for all strategies
- `performance.csv` — market-adjusted performance metrics
- `performance_raw.csv` — raw performance with SPY benchmark
- `lag_analysis.png` — optimal lag (p) over time
- `strategy_comparison.png` — market-adjusted strategy comparison
- `strategy_comparison_raw.png` — raw strategy comparison vs SPY

## Estimated Cost

| Resource | Duration | Rate | Cost |
|----------|----------|------|------|
| 1x c7i.4xlarge spot | ~2.5 hr | ~$0.20/hr | ~$0.50 |
| GPU (user's A100) | ~2 hr | $0 | $0 |
| **Total** | | | **~$0.50** |
