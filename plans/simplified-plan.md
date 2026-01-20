# ORACLE-VARX Project Roadmap

## Current Project Structure

```text
ORACLE-VARX/
├── scripts/
│   ├── example_var_usage.py      # VAR benchmarking (WORKING)
│   ├── example_data_usage.py     # Data loading demo (WORKING)
│   ├── cpu_vs_gpu_analysis_varx.md
│   └── explain_var_and_orvarx.md
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── constants.py          # ETFs, file paths, parameters
│   │   └── loader.py             # Data loading functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base class
│   │   ├── var_pytorch.py        # VAR (WORKING)
│   │   ├── dml_pytorch.py        # OR-VARX (NEEDS VALIDATION)
│   │   ├── oracle_var.py         # ORACLE-VARX (NEEDS VALIDATION)
│   │   └── results.py            # VARXResult, ORACLEVARXResult
│   └── modules/
│       ├── __init__.py
│       ├── batch_utils.py        # Shared batched OLS
│       ├── config.py             # RollingWindowConfig
│       ├── grid_config.py        # Grid configuration (separate lookbacks)
│       ├── validation.py         # RMSE, optimal p/alpha selection
│       ├── rolling_split.py      # CV splitter for DML cross-fitting
│       ├── model_cache.py        # Model caching for fold reuse
│       └── factory.py            # Learner factory (NEEDS VALIDATION)
├── plans/                        # Keep all old plans for reference
├── docs/
├── old-code/                     # Legacy code (untouched)
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Implementation Status

| Phase | Model | Status | Notes |
| ----- | ----- | ------ | ----- |
| 1 | VAR | DONE | Batched PyTorch, GPU-accelerated |
| 2 | OR-VARX | DONE | Grid-based memoization + vectorized batched OLS |
| 3 | ORACLE-VARX | NEXT | DML working, needs validation |

---

## Phase 1: VAR (COMPLETED)

### What's Working

- `src/models/var_pytorch.py` - Batched VAR with GPU acceleration
- `scripts/example_var_usage.py` - Benchmarking script

### Key Functions

| Function | Purpose |
| -------- | ------- |
| `build_var_design_batch()` | Constructs design matrices for all days in batch |
| `batch_var_all_days()` | Estimates VAR(1) through VAR(p_max) for all test days |
| `select_optimal_p()` | Picks best lag via validation RMSE |
| `fit_var()` | Main API - returns VARXResult |

### Algorithm

```text
For each test day d (from lookback to n_days):
    For each lag order p (from 1 to p_max):
        1. Extract training window: Y[d-lookback : d]
        2. Build design matrix X with p lags + intercept
        3. Solve OLS: β = (X'X)^{-1} X'Y
        4. Generate forecast: Y_hat[d] = X_pred @ β
    Select p* that minimizes validation RMSE
```

### Performance

- 16x speedup with chunked CUDA processing
- Cold start: ~0.35s (CUDA) vs ~0.8s (CPU)

---

## Phase 2: OR-VARX (COMPLETED)

**Goal:** Validate and test the DML (Double Machine Learning) implementation that removes confounding effects from VAR coefficients.

### What's Working

- `src/models/dml_pytorch.py` - Grid-based memoization with vectorized inference
- `src/modules/batch_utils.py` - Shared batched OLS utility
- `src/modules/grid_config.py` - Separate lookbacks for VAR (514) vs OR-VARX (1018)
- `src/modules/model_cache.py` - Model caching for fold reuse

**Files:**

| File | Key Functions |
| ---- | ------------- |
| `src/models/dml_pytorch.py` | `build_dml_data()`, `compute_residuals()`, `estimate_theta()`, `fit_orvarx()` |
| `src/modules/factory.py` | `get_regressor()`, `get_multi_output_regressor()` |
| `src/modules/grid_config.py` | `GridConfig` - Grid configuration dataclass |
| `src/modules/model_cache.py` | `ModelCache`, `FoldModels` - Model caching for fold reuse |

### Vectorized OR-VARX Functions

| Function | Purpose |
| -------- | ------- |
| `fit_orvarx_batched()` | Vectorized OR-VARX using pre-computed residuals and batched OLS |
| `get_all_required_folds()` | Determine all grid folds needed for entire test period |
| `precompute_all_residuals()` | Pre-compute residuals for all test windows at a given lag |
| `batched_ols()` | Shared batched OLS: beta = (X'X)^{-1} X'Y |

### First-Stage Learners (Phase 2 Requirement)

The factory supports CPU-only learners with native parallelization (`n_jobs=-1`):

| Learner | Library | Parallelization | Status |
| ------- | ------- | --------------- | ------ |
| XGBoost | xgboost | `n_jobs=-1`, `tree_method='hist'` | DONE |
| LightGBM | lightgbm | `n_jobs=-1`, `device='cpu'` | DONE |
| RandomForest | sklearn | `n_jobs=-1` | DONE |
| ExtraTrees | sklearn | `n_jobs=-1` | DONE |

**Note:** All learners are flexible/nonlinear tree-based methods, suitable for nuisance function estimation in DML. GPU learners (cuML, TabPFN) were considered but CPU with grid-based memoization proved more practical.

**Factory Implementation:**

```python
def get_regressor(name: str = 'xgboost', n_jobs: int = -1, **kwargs):
    """CPU-only learners with native parallelization."""
    if name == 'xgboost':
        return xgb.XGBRegressor(n_jobs=n_jobs, tree_method='hist', **kwargs)
    elif name == 'lgbm':
        return lgb.LGBMRegressor(n_jobs=n_jobs, device='cpu', **kwargs)
    elif name == 'rf':
        return RandomForestRegressor(n_jobs=n_jobs, **kwargs)
    elif name == 'extra_trees':
        return ExtraTreesRegressor(n_jobs=n_jobs, **kwargs)
```

`list_available_regressors()` returns `['xgboost', 'lgbm', 'rf', 'extra_trees']`

### DML Algorithm (Grid-Based Cross-Fitting)

The implementation uses **grid-based memoization** with fixed 21-day intervals. Models are pre-trained on a grid and reused across consecutive days, reducing complexity by ~250x.

```text
Grid Configuration:
  - Lookback: 1018 days (504 + 504 + 10 for p_max offset)
  - Training window per fold: 504 days (~2 years)
  - Test window per fold: 21 days (~1 month)

For each test day d:
    1. Build DML data from lookback window:
       - Outcome Y: current returns Y_t
       - Treatment T: lagged returns [Y_{t-1}, ..., Y_{t-p}]
       - Controls W: lagged confounders only [W_{t-1}, ..., W_{t-p}]
         (no current W_t to avoid lookahead bias)

    2. Grid-based cross-fitting (cached folds):
       - Identify active folds that cover the lookback window
       - For each active fold (from model cache):
         - Train or retrieve cached model: W -> Y (first stage for outcome)
         - Train or retrieve cached model: W -> T (first stage for treatment)
         - Compute residuals on held-out test rows:
           Y_res = Y - E[Y|W]
           T_res = T - E[T|W]

    3. Estimate deconfounded coefficients (second stage OLS):
       θ = (T_res' T_res)^{-1} T_res' Y_res

    4. Compute standard errors for ORACLE:
       SE(θ) = sqrt(diag((T_res'T_res)^{-1}) * σ²)

    5. Generate forecast using last fold's model:
       Y_hat = T_res @ θ (causal effect only)
```

### What θ Represents

After DML, the coefficients θ are **deconfounded VAR coefficients**:

- θ[i,j] = causal effect of asset j's lag on asset i
- Removes spurious correlation from confounders (e.g., both assets responding to VIX)
- These are what you want for lead-lag analysis

### Validation Tasks for OR-VARX

1. Run `scripts/example_orvarx_usage.py` - verify shapes and "PASS" output
2. Test `build_dml_data()` produces correct shapes:
   - outcome: (T-p, n_assets)
   - treatment: (T-p, n_assets * p)
   - controls: (T-p, n_confounders * p) — no current W_t
3. Verify grid-based cross-fitting produces valid residuals
4. Test `estimate_theta()` with known coefficients
5. Verify `fit_orvarx()` returns VARXResult with `is_orthogonalized=True`
6. Test each learner: XGBoost, LightGBM, RandomForest, ExtraTrees
7. Verify model cache reuse across consecutive days

### Cross-Fitting Parameters (Grid-Based)

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| `ols_window` | 504 | Rows for final OLS (both methods) |
| `tree_train_window` | 504 | First-stage tree training (OR-VARX only) |
| `test_size` | 21 | Test window per fold (~1 month) |
| `p_max_offset` | 10 | Extra days for lag offset |
| `lookback_var` | 514 | VAR lookback (504 + 10) |
| `lookback_orvarx` | 1018 | OR-VARX lookback (504 + 504 + 10) |

**Note:** Old parameters `lookback_base` (756) and `lookback` (766) are DEPRECATED.

Folds are cached and reused across consecutive days. New models trained only every 21 days.

---

## Phase 3: ORACLE-VARX (AFTER OR-VARX)

**Goal:** Validate the significance-based lag selection that adaptively chooses p based on statistical significance of coefficients.

**Files:** `src/models/oracle_var.py` - `oracle_significance_test()`, `fit_oracle_varx()`

### ORACLE Algorithm

```text
For each significance level α in alpha_grid:
    For each test day d:
        1. Fit OR-VARX to get θ and SE(θ)

        2. Run significance test:
           z_crit = Φ^{-1}(1 - α/2)  # two-tailed
           For each lag k from 1 to p_max:
               z_stat = |θ_k| / SE(θ_k)
               if any(z_stat > z_crit):
                   p_selected = k

        3. Generate forecast using p_selected

Select α* that minimizes validation RMSE
Final forecast uses p determined by α*
```

### Alpha Grid (Default)

```python
alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
```

- Lower α = stricter test = fewer lags kept
- Higher α = more permissive = more lags kept

### Result Structure

ORACLEVARXResult contains:

| Field | Shape | Description |
| ----- | ----- | ----------- |
| `forecasts` | (n_assets, n_days) | Final forecasts at optimal α |
| `forecasts_all` | (n_assets, n_days, n_alphas) | Forecasts for all α values |
| `alpha_optimal` | (n_days,) | Selected α per day |
| `p_optimal` | (n_days,) | Selected p per day |
| `p_optimal_all` | (n_days, n_alphas) | p values for each α |
| `coefficients_all` | (n_days, n_alphas, p_max, n_assets, n_assets) | 5D tensor |

### Validation Tasks for ORACLE-VARX

1. Create `scripts/example_oracle_usage.py` with working example
2. Test `oracle_significance_test()`:
   - With strong signal → should keep more lags
   - With noise → should fall back to p=1
   - Higher α → should keep more lags
3. Test full `fit_oracle_varx()` pipeline
4. Verify 5D coefficient tensor shape
5. Test `get_leadlag_matrix()` with and without alpha_idx

---

## Data Requirements

### Dataset Location

```text
dataset/
├── OPCL_20000103_20201231.csv     # ETF returns (required)
├── VIX_20000103_20201231.csv      # VIX index (required for OR-VARX)
├── DFF_20000103_20201231.csv      # Fed funds rate (optional)
├── T5YIE_20030102_20201231.csv    # Inflation expectations (optional)
├── DCOILWTICO_20000103_20201231.csv  # WTI crude oil (optional)
└── USEPUINDXD_20000103_20201231.csv  # Policy uncertainty (optional)
```

### Test Universe (9 ETFs)

| Ticker | Sector |
| ------ | ------ |
| XLY | Consumer Discretionary |
| XLP | Consumer Staples |
| XLE | Energy |
| XLF | Financials |
| XLV | Healthcare |
| XLI | Industrials |
| XLB | Materials |
| XLK | Technology |
| XLU | Utilities |

### Default Parameters

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| `lookback_days` | 756 | ~3 years of trading days |
| `validation_days` | 20 | ~1 month for hyperparameter selection |
| `p_max` | 10 | Maximum VAR lag order |
| `test_days` | 252+ | Minimum 1 year test period (need n_days >= 1008) |

---

## Tensor Shapes Reference

### During Computation

| Model | forecasts_all | coefficients |
| ----- | ------------- | ------------ |
| VAR | (n_assets, n_days, p_max) | (n_days, p_max, n_assets, n_assets) |
| OR-VARX | (n_assets, n_days, p_max) | (n_days, p_max, n_assets, n_assets) |
| ORACLE | (n_assets, n_days, n_alphas) | (n_days, n_alphas, p_max, n_assets, n_assets) |

### Example (9 ETFs, 120 test days)

| Model | forecasts_all | coefficients | Memory |
| ----- | ------------- | ------------ | ------ |
| VAR/OR-VARX | (9, 120, 10) | (120, 10, 9, 9) | ~1 MB |
| ORACLE | (9, 120, 7) | (120, 7, 10, 9, 9) | ~5 MB |

---

## Immediate Next Steps

### Step 1: Validate OR-VARX

```python
# scripts/example_orvarx_usage.py - Simple verification script
import torch
from src.models import fit_orvarx
from src.modules.grid_config import GridConfig

# Minimal test data (1050 days = 1018 lookback + 32 output days)
n_days = 1050
n_assets = 10
n_confounders = 3
p_max = 10

Y = torch.randn(n_days, n_assets)
W = torch.randn(n_days, n_confounders)

config = GridConfig()  # defaults: lookback_orvarx=1018

# Fit OR-VARX
result = fit_orvarx(
    Y, W,
    p_max=p_max,
    config=config,
    validation_days=20,
    learner_name='extra_trees',
)

# Verify output
expected_output_days = n_days - config.lookback_orvarx
print(f"Output shape: {result.forecasts.shape}")
print(f"Expected: ({n_assets}, {expected_output_days})")
print(f"p_optimal: {result.p_optimal[0].item()}")
print("PASS" if result.forecasts.shape[1] == expected_output_days else "FAIL")
```

### Step 2: Validate ORACLE-VARX

```python
# Create scripts/example_oracle_usage.py
from src.models.oracle_var import fit_oracle_varx

result = fit_oracle_varx(
    Y, W,
    alpha_grid=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    p_max=10,
    lookback=756,
    validation_days=20,
    asset_names=tickers,
    confounder_names=["VIX"],
    dates=dates[756:],
    learner_name='xgboost',
    use_gpu=True,
    include_confounder_baseline=False,
)

print(f"Method: {result.method}")  # Should be 'ORACLE-VARX'
print(f"Alpha grid: {result.alpha_grid}")
print(f"Coefficients shape: {result.coefficients_all.shape}")  # (252, 7, 10, 9, 9)
```

### Step 3: Compare Models

After validation, compare forecasting performance:

1. Run VAR, OR-VARX, ORACLE-VARX on same test period
2. Compare RMSE on out-of-sample forecasts
3. Visualize lead-lag matrices to see deconfounding effect

---

## Future Work (After Validation)

### Cloud/Dockerization

- Build Docker container for RunPod
- S3 upload/download utilities
- Worker script for batch processing
- Full 5000-day backfill

### Additional Enhancements

- Additional confounders (DFF, T5YIE, oil, policy uncertainty)
- Streaming results for long runs
- Learner comparison analysis (XGB vs LGBM vs RF vs TabPFN)
