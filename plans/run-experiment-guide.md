# Experiment Execution Guide

## 1. Methods Explained

| Method | Description | Requires GPU |
|--------|-------------|:------------:|
| **VAR** x 1 | Standard Vector Autoregression — forecasts ETF returns using only their own lagged values | No |
| **ACLE-VAR** x 1 | VAR with Augmented Covariance-Learning Estimation — applies cross-validated elastic net regularization to VAR coefficients | No |
| **VARX** x 3 | VAR with exogenous confounders — adds macro variables as extra predictors (OLS-fitted, no learner needed) | No |
| **ACLE-VARX** x 3 | Regularized VARX via elastic net (OLS-fitted, no learner needed) | No |
| **OR-VARX** x 12 | Orthogonal VARX — uses Double Machine Learning (DML) to partial out confounder effects with a tree-based first-stage learner, 4 learners x 3 confounder presets | No |
| **ORACLE-VARX** x 12 | OR-VARX plus ACLE regularization on the second-stage coefficients, 4 learners x 3 confounder presets | No |
| **ORACLE-VARX-TabPFN** x 3 | ORACLE-VARX using TabPFN (a GPU-based tabular transformer) as the first-stage learner, 3 confounder presets | **Yes** |

The four tree-based learners used in OR-VARX and ORACLE-VARX are: `lgbm`, `xgboost`, `rf`, `extra_trees`.

## 2. Confounder Presets

Each preset selects a different set of macro/financial variables as confounders (exogenous regressors):

| Preset | Variables | Description |
|--------|-----------|-------------|
| `vix` | VIX | CBOE Volatility Index |
| `macro5` | VIX, DFF, T5YIE, DCOILWTICO, USEPUINDXD | 5 key macro variables |
| `all10` | All 10 below | Full confounder set |

**Variable definitions:**

| Variable | Description |
|----------|-------------|
| VIX | CBOE Volatility Index (market fear gauge) |
| DFF | Federal Funds Effective Rate |
| T5YIE | 5-Year Breakeven Inflation Rate |
| DCOILWTICO | WTI Crude Oil Price |
| USEPUINDXD | US Economic Policy Uncertainty Index |
| BAMLC0A4CBBB | ICE BofA BBB Corporate Bond Spread |
| DFII10 | 10-Year Real Interest Rate (TIPS) |
| DTWEXBGS | Trade-Weighted US Dollar Index (Broad Goods) |
| DTWEXEMEGS | Trade-Weighted US Dollar Index (Emerging Markets) |
| GVZCLS | CBOE Gold Volatility Index |

## 3. Experiment Phases

The orchestrator (`run_all_experiments.py`) organizes work into 4 CPU phases plus optional GPU runs:

| Phase | What it runs | Methods produced | CPU/GPU |
|-------|-------------|-----------------|---------|
| **Phase 0** | All OLS baselines (sequential) | VAR x1, ACLE-VAR x1, VARX x3, ACLE-VARX x3 | CPU |
| **Phase 1** | VIX x 4 learners (parallel tmux panes) | OR-VARX x4, ORACLE-VARX x4 | CPU |
| **Phase 2** | macro5 x 4 learners (parallel tmux panes) | OR-VARX x4, ORACLE-VARX x4 | CPU |
| **Phase 3** | all10 x 4 learners (parallel tmux panes) | OR-VARX x4, ORACLE-VARX x4 | CPU |
| **TabPFN** | 3 confounder presets (run manually) | ORACLE-VARX-TabPFN x3 | **GPU** |

Phase 0 runs all OLS methods sequentially with full CPU access (`--ols-only` for confounder presets). Phases 1-3 run DML methods only — `run_combined_experiment.py` without `--ols-only` goes directly to the DML branch.

## 4. Prerequisites

### Python

Python 3.10+.

### Dependencies (CPU — Phases 0-3)

```bash
pip install -r requirements-cpu.txt
```

This installs PyTorch (CPU), scikit-learn, LightGBM, XGBoost, and other core packages. No GPU or CUDA required.

### Dependencies (GPU — TabPFN)

```bash
pip install -r requirements-gpu.txt
```

This is a superset of `requirements-cpu.txt` that adds TabPFN and NVIDIA CUDA libraries. Requires an NVIDIA GPU with CUDA support.

### HuggingFace Token (TabPFN only)

TabPFN downloads model weights from HuggingFace. You need a free access token:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to Settings > Access Tokens > New Token
3. Export the token before running TabPFN experiments:

```bash
export HF_TOKEN=<your-token>
```

## 5. Setting Up on EC2

### One-line setup

```bash
# CPU-only setup (Phases 0-3):
curl -sSL https://raw.githubusercontent.com/HK-Tan/ORACLE-VARX/main/scripts/setup-ec2.sh | bash

# GPU setup (TabPFN):
curl -sSL https://raw.githubusercontent.com/HK-Tan/ORACLE-VARX/main/scripts/setup-ec2.sh | bash -s -- --gpu
```

### Manual setup

```bash
git clone https://github.com/HK-Tan/ORACLE-VARX.git
cd ORACLE-VARX
bash scripts/setup-ec2.sh          # CPU
bash scripts/setup-ec2.sh --gpu    # GPU
```

### After setup

```bash
cd ORACLE-VARX
source .venv/bin/activate
```

## 6. Running Experiments

### Smoke test

Run a quick test with limited data to verify everything works:

```bash
python scripts/run_combined_experiment.py --confounders vix --learner lgbm --n-days 1500 --no-show
```

### Single experiment

Run one (confounder preset, learner) combination:

```bash
# VIX with LightGBM (produces VARX, ACLE-VARX, OR-VARX, ORACLE-VARX)
python scripts/run_combined_experiment.py --confounders vix --learner lgbm --no-show --verbose

# VAR + ACLE-VAR baseline (no confounders)
python scripts/run_combined_experiment.py --no-confounders --no-show --verbose
```

### Orchestrator (all CPU phases)

The orchestrator launches all 4 phases sequentially, with each phase's learner runs in parallel tmux panes:

```bash
# Run everything
python scripts/run_all_experiments.py --phase all --verbose

# Run a single phase
python scripts/run_all_experiments.py --phase 1 --verbose

# Preview commands without executing
python scripts/run_all_experiments.py --phase all --dry-run
```

#### tmux pane layout

Each phase (1-3) launches a 2x2 tmux grid. The panes map to learners as follows:

```
┌──────────────────┬─────────────────-----─┐
│  Pane 0: lgbm    │  Pane 1: xgboost      │
├──────────────────┼──────────────────-----┤
│  Pane 2: rf      │  Pane 3: extra_trees  │
└──────────────────┴──────────────────-----┘
```

Navigate between panes with `Ctrl-b` then arrow keys. Scroll up in a pane with `Ctrl-b [` (press `q` to exit scroll mode). Detach with `Ctrl-b d`.

BLAS thread limits (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) and `--n-jobs` are automatically set per pane to avoid CPU over-subscription.

### TabPFN (GPU)

Run on a machine with an NVIDIA GPU:

```bash
export HF_TOKEN=<your-token>

# All three presets sequentially (vix → macro5 → all10)
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders all --no-show --verbose

# Or one preset at a time
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders vix --no-show --verbose
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders macro5 --no-show --verbose
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders all10 --no-show --verbose

# Debug run example: p=1..3 with CPU Phase 4 OLS
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders vix --p-max 3 --phase4-ols-device cpu --no-show --verbose

# Just probing VRAM behaviors
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders vix --probe

# Skip coefficient heatmap refit (faster, no extra TabPFN cold-start)
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders vix --no-show --verbose --skip-heatmap
```

Note: There is no `--p-list` option in this script. Lag orders are always run as the inclusive range `p=1..p_max`.

## 7. CLI Reference

### `run_combined_experiment.py`

Runs up to 4 methods for a single (confounder preset, learner) pair with amortized DML first stage.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--confounders` | str | `vix` | Preset name (`vix`, `macro5`, `all10`) or comma-separated variable names |
| `--no-confounders` | flag | — | Run VAR + ACLE-VAR baseline (no confounders) |
| `--learner` | str | `lgbm` | First-stage learner: `lgbm`, `xgboost`, `rf`, `extra_trees` |
| `--n-days` | int | all | Number of days to load (default: full dataset) |
| `--validation-days` | int | `21` | Validation period in days |
| `--p-max` | int | `10` | Maximum lag order |
| `--alpha-grid` | str | `0.01,...,0.30` | Comma-separated alpha values for elastic net |
| `--n-jobs` | int | auto | CPU cores per method |
| `--device` | str | `cpu` | Device (`cpu` or `cuda`) |
| `--output-dir` | str | `results` | Base results directory |
| `--no-show` | flag | — | Don't display plots (use on headless servers) |
| `--verbose` | flag | — | Print detailed progress |
| `--ols-only` | flag | — | In confounders mode, run only VARX + ACLE-VARX (skip DML). No effect with `--no-confounders`. |

### `run_all_experiments.py`

Orchestrates all experiment phases using tmux.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--phase` | str | *required* | Phase to run: `0`, `1`, `2`, `3`, or `all` |
| `--n-jobs` | int | auto | CPU cores per tmux pane |
| `--dry-run` | flag | — | Print commands without executing |
| `--verbose` | flag | — | Pass `--verbose` to experiment scripts |

### `run_oraclevarx_tabpfn_experiment.py`

Runs ORACLE-VARX with TabPFN as the first-stage learner (GPU required).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--confounders` | str | `vix` | Preset name (`vix`/`macro5`/`all10`), `all` to run all three sequentially, or comma-separated variable names |
| `--n-days` | int | all | Number of days to load |
| `--validation-days` | int | `21` | Validation period in days |
| `--p-max` | int | `10` | Maximum lag order |
| `--alpha-grid` | str | `0.01,...,0.30` | Comma-separated alpha values |
| `--output-dir` | str | `results/oraclevarx_tabpfn` | Output directory |
| `--name` | str | auto | Experiment name |
| `--no-show` | flag | — | Don't display plots |
| `--verbose` | flag | — | Print detailed progress |
| `--probe` | flag | — | Probe mode: run 1 fold per p to test VRAM usage (no results saved) |
| `--phase4-ols-device` | str | `cpu` | Device for Phase 4 batched OLS (`cpu` or `cuda`) |
| `--skip-heatmap` | flag | — | Skip coefficient heatmap refit (avoids slow TabPFN refit) |

## 8. Output Structure

Each method saves results under `results/<method>/<experiment_name>/`:

| File | Contents |
|------|----------|
| `results.pt` | Model results (forecasts, coefficients, etc.) |
| `pnl_results.pkl` | PnL data for all strategies |
| `performance.csv` | Market-adjusted performance metrics |
| `performance_raw.csv` | Raw performance with SPY benchmark |
| `lag_analysis.png` | Optimal lag order (p) over time |
| `strategy_comparison.png` | Market-adjusted strategy comparison chart |
| `strategy_comparison_raw.png` | Raw strategy comparison vs SPY |

## 9. Copying Results from EC2

Using `scp`:

```bash
scp -i <your-key.pem> -r ubuntu@<EC2-IP>:~/ORACLE-VARX/results/ ./results/
```

Using `rsync` (resume-capable, recommended for large transfers):

```bash
rsync -avz -e "ssh -i <your-key.pem>" ubuntu@<EC2-IP>:~/ORACLE-VARX/results/ ./results/
```


> **Note on c8a vCPU counts:** The c8a instance family (AMD EPYC 5th gen) ships with SMT disabled by default (1 thread per core), so **1 vCPU = 1 physical core**. For example, `c8a.4xlarge` has 16 vCPUs = 16 physical cores. This differs from older families (c5, m5, etc.) where 2 vCPUs = 1 core due to hyperthreading.
