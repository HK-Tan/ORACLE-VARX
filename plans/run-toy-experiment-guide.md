# Toy Experiment Execution Guide

## 1. Methods Explained

| Method | Description | Phase |
|--------|-------------|:-----:|
| **VAR** x 1 | Standard Vector Autoregression on endogenous variables only | 0 |
| **ACLE-VAR** x 1 | VAR with significance-based lag selection (FDR + alpha tuning) | 0 |
| **VARX** x 3 | VAR with confounders as extra predictors (OLS, no learner) | 0 |
| **ACLE-VARX** x 3 | VARX with significance-based lag selection | 0 |
| **OR-VARX** x 3 | Orthogonal VARX — DML first stage partials out confounders | 1 |
| **ORACLE-VARX** x 3 | OR-VARX plus significance-based lag selection | 1 |
| **OR-VARX-TabPFN** x 3 | OR-VARX with TabPFN transformer for first-stage nuisance estimation | 2 |
| **ORACLE-VARX-TabPFN** x 3 | TabPFN OR-VARX plus significance-based lag selection | 2 |

Total: **14 CPU runs** (8 OLS + 6 DML) + **6 GPU runs** (TabPFN). Multi-learner Phase 1 adds up to 24 more.

## 2. Observability Levels

Each confounder-aware method runs under three observability conditions:

| Level | Variables Observed | Label |
|-------|--------------------|-------|
| All | W1, W2, W3 | `all` |
| Partial-2 | W1, W2 | `partial_2` |
| Partial-1 | W1 | `partial_1` |

VAR and ACLE-VAR run once with no confounders (`none`).

## 3. Experiment Phases

| Phase | What it runs | Methods produced | Notes |
|-------|-------------|-----------------|-------|
| **Phase 0** | All OLS baselines (sequential) | VAR x1, ACLE-VAR x1, VARX x3, ACLE-VARX x3 | Fast (~6s) |
| **Phase 1** | DML methods per obs level | OR-VARX x3, ORACLE-VARX x3 (per learner) | Slow (tree fitting); parallelizable per obs level |
| **Phase 2** | TabPFN methods per obs level (GPU) | OR-VARX-TabPFN x3, ORACLE-VARX-TabPFN x3 | Requires CUDA GPU |

Phase 0 runs all 8 OLS methods sequentially. Phase 1 runs DML for each observability level — the expensive `fit_orvarx_core()` is computed once per obs level and shared between OR-VARX and ORACLE-VARX. Phase 2 uses TabPFN transformer for GPU-accelerated first-stage estimation.

## 4. Prerequisites

### Python

Python 3.10+.

### Dependencies

```bash
pip install -r requirements-cpu.txt   # Phase 0 + 1
pip install -r requirements-gpu.txt   # Phase 2 (adds TabPFN, CUDA)
```

Phase 2 requires a CUDA-capable GPU. Phases 0 and 1 run on CPU only.

## 5. Setting Up on EC2

### One-line setup

```bash
# CPU-only setup (Phases 0-1):
curl -sSL https://raw.githubusercontent.com/HK-Tan/ORACLE-VARX/main/scripts/setup-ec2.sh | bash

# GPU setup (Phase 2 TabPFN):
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

### Step 1: Generate data (once)

```bash
python scripts/generate_toy_data.py
```

If you are re-running/reproducing the results, **you do not have to run this!**.

This creates `dataset/toy/` with:
- `Y.csv` — endogenous variables (3000 × 3)
- `W.csv` — confounders (3000 × 3)
- `A_true.pt` — ground truth coefficients (3000 × 3 × 3 × 3)
- `dgp_config.json` — all DGP parameters

Verify the output shows:
```
Regime [0, 1000): 6 non-zero entries at t=500
Regime [1000, 2000): 5 non-zero entries at t=1500
Regime [2000, 3000): 3 non-zero entries at t=2500
```

### Step 2: Run Phase 0 (OLS baselines)

```bash
python scripts/run_toy_benchmark.py --phase 0 --no-show
```

Runs all 8 OLS methods sequentially. Takes ~6 seconds on a laptop.

### Step 3: Run Phase 1 (DML methods)

Phase 1 is the slow part — it fits tree-based models for every fold/day. Each (learner, obs level) pair produces 2 methods: OR-VARX + ORACLE-VARX.

#### Orchestrator (recommended) — 4 learners in parallel

The orchestrator launches all 4 learners in parallel tmux panes, each running all 3 obs levels sequentially within its pane:

```
Pane 1 (lgbm):        obs=all → obs=partial_2 → obs=partial_1  (6 method-runs, sequential)
Pane 2 (xgboost):     obs=all → obs=partial_2 → obs=partial_1  (6 method-runs, sequential)
Pane 3 (rf):          obs=all → obs=partial_2 → obs=partial_1  (6 method-runs, sequential)
Pane 4 (extra_trees): obs=all → obs=partial_2 → obs=partial_1  (6 method-runs, sequential)
                                                                 ─────────────────────────
                                                                 24 total, 4 panes parallel
```

Wall-clock time is ~1 pane's time (not 4×), since all 4 run concurrently.

```bash
# Run everything: Phase 0 (sequential, ~6s) then Phase 1 (4 parallel tmux panes)
python scripts/run_all_toy_experiments.py --phase all --verbose

# Phase 1 only (skip OLS baselines)
python scripts/run_all_toy_experiments.py --phase 1

# Preview commands without executing
python scripts/run_all_toy_experiments.py --phase all --dry-run

# Override per-pane thread count
python scripts/run_all_toy_experiments.py --phase 1 --n-jobs 3
```

BLAS thread limits (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) are auto-set per pane to avoid oversubscription: `threads_per_pane = (physical_cores - 1) / 4`.

#### Single experiment (debugging / one-off)

`run_toy_benchmark.py --phase 1` runs a single process, looping over learners (outer) then obs levels (inner), all **sequentially**. Useful for debugging or running a specific (learner, obs) combo.

**Defaults:** `--learner extra_trees`, `--obs` = all 3 levels.

```bash
# 1 learner × 3 obs levels = 6 method-runs, sequential
#   Produces: OR-VARX_extra_trees_{all,partial_2,partial_1}/,
#             ORACLE-VARX_extra_trees_{all,partial_2,partial_1}/
python scripts/run_toy_benchmark.py --phase 1 --no-show

# 1 learner × 1 obs level = 2 method-runs, sequential
#   Produces: OR-VARX_extra_trees_all/, ORACLE-VARX_extra_trees_all/
python scripts/run_toy_benchmark.py --phase 1 --obs all --no-show

# Specific learner + specific obs = 2 method-runs, sequential
#   Produces: OR-VARX_lgbm_partial_2/, ORACLE-VARX_lgbm_partial_2/
python scripts/run_toy_benchmark.py --phase 1 --learner lgbm --obs partial_2 --no-show

# Custom thread count
python scripts/run_toy_benchmark.py --phase 1 --learner rf --n-jobs 4 --no-show

# Phase 0 + Phase 1 together (OLS baselines then DML, all sequential)
python scripts/run_toy_benchmark.py --phase all --no-show
```

> **Avoid `--learner all`** — it runs 4 learners × 3 obs = 24 method-runs **sequentially** (very slow). Use the orchestrator instead.

#### Quick reference

| Command | Learners | Obs levels | Method-runs | Execution |
|---------|----------|-----------|-------------|-----------|
| `run_all_toy_experiments.py --phase 1` | **4 (parallel tmux)** | **3 per pane** | **24** | **4 panes parallel** |
| `run_all_toy_experiments.py --phase all` | **4 (parallel tmux)** | **3 per pane** | **8 OLS + 24 DML** | **Phase 0 seq, Phase 1 parallel** |
| `run_toy_benchmark.py --phase 1` | 1 (extra_trees) | 3 | 6 | Sequential |
| `run_toy_benchmark.py --phase 1 --obs all` | 1 (extra_trees) | 1 | 2 | Sequential |
| `run_toy_benchmark.py --phase 1 --learner lgbm` | 1 (lgbm) | 3 | 6 | Sequential |
| `run_toy_benchmark.py --phase 1 --learner lgbm --obs partial_2` | 1 (lgbm) | 1 | 2 | Sequential |
| `run_toy_benchmark.py --phase 1 --learner all` | 4 | 3 | 24 | Sequential (slow!) |
| `run_toy_benchmark.py --phase all` | 1 (extra_trees) | 3 | 8 OLS + 6 DML | Sequential |

### Step 4: Run Phase 2 (TabPFN methods, GPU)

```bash
# All obs levels (requires CUDA)
python scripts/run_toy_benchmark.py --phase 2 --device cuda --no-show

# Single obs level with custom ensemble size
python scripts/run_toy_benchmark.py --phase 2 --obs all --device cuda --n-estimators 8 --no-show
```

Phase 2 requires a CUDA-capable GPU. The batch size heuristic auto-scales for the toy problem's smaller n_assets=3.

## 7. CLI Reference

### `scripts/generate_toy_data.py`

Generates and saves toy benchmark data.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--seed` | int | `42` | Random seed |
| `--noise-scale` | float | `0.05` | Innovation noise scaling ν |
| `--confounder-strength` | float | `1.0` | Confounder nuisance scaling λ |
| `--T` | int | `3000` | Number of time steps |

### `scripts/run_toy_benchmark.py`

Runs the toy benchmark experiments.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--phase` | str | `all` | Phase to run: `0`, `1`, `2`, or `all` (=0+1) |
| `--obs` | str | all levels | Obs level for Phase 1/2: `all`, `partial_2`, `partial_1` |
| `--learner` | str | `extra_trees` | First-stage DML learner, or `all` for all 4 learners |
| `--n-jobs` | int | auto | CPU cores for DML (`-1` = `physical_cores - 1`) |
| `--device` | str | `cpu` | Device for Phase 2 TabPFN: `cpu` or `cuda` |
| `--n-estimators` | int | `8` | TabPFN ensemble size for Phase 2 |
| `--noise-scale` | float | — | Override innovation noise scale (regenerates data) |
| `--confounder-strength` | float | — | Override confounder scaling λ (regenerates data) |
| `--no-show` | flag | — | Don't display plots (use on headless servers) |
| `--verbose` | flag | — | Print detailed progress |

> **`--no-show` is required on headless servers** (EC2, etc.). Without it, matplotlib tries to open an interactive display window.

### `scripts/run_all_toy_experiments.py`

Tmux orchestrator for parallel Phase 1 runs.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--phase` | str | required | Phase to run: `0`, `1`, or `all` (=0 then 1) |
| `--n-jobs` | int | auto | CPU cores per tmux pane (default: auto-computed) |
| `--dry-run` | flag | — | Print commands without executing |
| `--verbose` | flag | — | Pass `--verbose` to experiment scripts |
| `--no-show` | flag | `True` | Pass `--no-show` to experiment scripts |

## 8. Output Structure

All results are saved under `results-toy/`:

### Per-run outputs (`results-toy/{method}_{obs_level}/`)

| File | Contents |
|------|----------|
| `result.pt` | Saved result object (forecasts, coefficients, etc.) |
| `metrics.json` | Edge MAE/MSE, zero-edge MAE, forecast MAE/MSE (by regime) |
| `edge_trajectories.png` | True vs estimated coefficient trajectories (6 edges) |
| `lag_analysis.png` | Optimal lag order (p) over time |
| `forecast_error_mse.png` | Rolling MSE vs time with regime boundaries |
| `forecast_error_mae.png` | Rolling MAE vs time with regime boundaries |
| `heatmap_max_p.png` | Coefficient heatmap at the day with highest p_optimal |
| `heatmap_last_day.png` | Coefficient heatmap at the last output day |
| `coef_evolution_max_p.png` | Per-p refit grid at max_p day |
| `coef_evolution_last_day.png` | Per-p refit grid at last day |

### Top-level outputs (`results-toy/`)

| File | Contents |
|------|----------|
| `metrics_summary.csv` | Incrementally-appended consolidated table of all runs |

Each experiment appends its row to `metrics_summary.csv` (replacing if the same method+obs already exists). This means tmux panes can write concurrently without coordination.

### Example directory tree after a full run

DML methods always include the learner name in the directory (e.g. `OR-VARX_extra_trees_all/`).

```
results-toy/
├── metrics_summary.csv          # Incremental, all methods
├── VAR_none/
│   ├── result.pt
│   ├── metrics.json
│   ├── edge_trajectories.png
│   ├── lag_analysis.png
│   ├── forecast_error_mse.png
│   ├── forecast_error_mae.png
│   ├── heatmap_max_p.png
│   ├── heatmap_last_day.png
│   ├── coef_evolution_max_p.png
│   └── coef_evolution_last_day.png
├── ACLE-VAR_none/
│   └── ...
├── VARX_all/
│   └── ...
├── ACLE-VARX_all/
│   └── ...
├── OR-VARX_extra_trees_all/     # Phase 1 (learner always in name)
│   └── ...
├── ORACLE-VARX_extra_trees_all/
│   └── ...
├── OR-VARX_lgbm_all/
│   └── ...
├── ORACLE-VARX_lgbm_all/
│   └── ...
├── OR-VARX-TabPFN_all/          # Phase 2
│   └── ...
├── ORACLE-VARX-TabPFN_all/
│   └── ...
└── ... (more method × obs_level directories)
```

## 9. What to Look For

### Regime transitions
The lag analysis plots should show `p_optimal` shifting:
- **Regime 1** [0, 1000): p ≈ 3 (all 6 edges active, max lag = 3)
- **Regime 2** [1000, 2000): p ≈ 2 (Z_{t-3}→Z_t gone)
- **Regime 3** [2000, 3000]: p ≈ 1 (only lag-1 cycle remains)

### DML advantage
Under "all" confounders, OR-VARX/ORACLE-VARX `overall_nonzero_mae` in `metrics.json` should be lower than VAR's, since DML removes confounder bias.

### Edge trajectories
Each experiment's `edge_trajectories.png` shows true coefficient paths (black) vs the method's estimated values for all 6 edges. The time-varying edges (X→X lag 2, Y→Y lag 2, Z→Z lag 3) should show estimated paths tracking the linear decay toward zero.

### Degradation under partial observability
As confounders are hidden (all → partial_2 → partial_1), DML methods should gradually lose their advantage over OLS baselines.

### TabPFN vs tree-based DML
Phase 2 TabPFN results can be compared directly against Phase 1 tree-based results at each observability level. TabPFN may offer different bias-variance tradeoffs, especially with fewer training samples per fold.

## 10. Customizing the DGP

### Option A: Regenerate data separately, then run benchmark

```bash
# Higher noise
python scripts/generate_toy_data.py --noise-scale 0.2

# Stronger confounding (stress-test DML)
python scripts/generate_toy_data.py --confounder-strength 3.0

# Both at once
python scripts/generate_toy_data.py --noise-scale 0.1 --confounder-strength 2.0

# Shorter time series / different seed
python scripts/generate_toy_data.py --T 1500 --seed 123
```

After regenerating, re-run the benchmark to get updated results.

### Option B: Override noise-scale and λ directly in the benchmark command

The `--noise-scale` and `--confounder-strength` flags regenerate data inline before running:

```bash
# Run Phase 0 with higher noise (noise_scale=0.1)
python scripts/run_toy_benchmark.py --phase 0 --noise-scale 0.1 --no-show

# Run Phase 0 with stronger confounding (λ=3.0)
python scripts/run_toy_benchmark.py --phase 0 --confounder-strength 3.0 --no-show

# Sweep: low noise + strong confounding
python scripts/run_toy_benchmark.py --phase 0 --noise-scale 0.02 --confounder-strength 5.0 --no-show

# Full Phase 0+1 with custom DGP
python scripts/run_toy_benchmark.py --phase all --noise-scale 0.1 --confounder-strength 2.0 --no-show
```

The `dgp_config.json` records all parameters for reproducibility.

## 11. Copying Results

### From EC2

```bash
scp -i <your-key.pem> -r ubuntu@<EC2-IP>:~/ORACLE-VARX/results-toy/ ./results-toy/
```

### Using rsync (resume-capable)

```bash
rsync -avz -e "ssh -i <your-key.pem>" ubuntu@<EC2-IP>:~/ORACLE-VARX/results-toy/ ./results-toy/
```

> **Note on c8a vCPU counts:** The c8a instance family (AMD EPYC 5th gen) ships with SMT disabled by default (1 thread per core), so **1 vCPU = 1 physical core**. For example, `c8a.4xlarge` has 16 vCPUs = 16 physical cores. This differs from older families (c5, m5, etc.) where 2 vCPUs = 1 core due to hyperthreading.

### From ThunderCompute

Using `tnr scp` (recommended — handles keys and ports automatically):

```bash
tnr scp 0:~/ORACLE-VARX/results-toy/ ./
```

> **ThunderCompute notes:**
> - Install the CLI with `pip install tnr` and authenticate with `tnr login`.
> - SSH ports can change between sessions — always re-check with `tnr status`.
> - There is no stop/restart — only delete or snapshot. **Download results before deleting the instance.**
