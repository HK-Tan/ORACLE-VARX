# How VAR and OR-VARX Work: A Complete Code Walkthrough

## Overview

The codebase implements two related time-series models:

1. **VAR (Vector Autoregression)**: Standard multivariate time-series model
2. **OR-VARX (Orthogonalized VAR with Exogenous Variables)**: VAR enhanced with Double Machine Learning (DML) to remove confounding effects

---

## Part 1: Plain VAR Model

### Step 1: Data Loading

From the example at `scripts/example_var_usage.py`:

```python
Y = torch.randn(n_days, n_assets)  # Generate or load your data
```

For real data, you can use `load_test_data()` in `src/data/loader.py:350-392`, which:

1. Loads **9 sector ETFs** (XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU) from OPCL data
2. Loads **VIX** as the confounder variable
3. Aligns dates and converts to PyTorch tensors

**Output shapes:**
- `Y`: `(624, 9)` - 624 days of returns for 9 assets
- `W`: `(624, 1)` - 624 days of VIX values

---

### Step 2: Calling `fit_var()`

From `scripts/example_var_usage.py`:

```python
result = fit_var(
    Y,
    p_max=10,
    config=GridConfig(),  # Uses lookback=766 by default
    validation_days=20,
)
```

This is the main entry point in `src/models/var_pytorch.py:354-469`.

**The VAR Model:**

```
Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + ε_t
```

Where:
- `Y_t` = returns at time t for all assets (vector of size n_assets)
- `A_k` = coefficient matrix for lag k (n_assets x n_assets)
- `A_k[i, j]` = effect of asset j at lag k on asset i

---

### Step 3: Batched VAR Estimation

The core computation happens in `batch_var_all_days()` at `src/models/var_pytorch.py:194-292`.

**Algorithm:**

```
1. Create sliding windows for all test days using torch.unfold()
   - windows shape: (n_test_days, lookback, n_assets)

2. For each p in 1..p_max:
   a. Build design matrices via build_var_design_batch()
   b. Solve batched OLS: β = (X'X)^{-1} X'Y
   c. Extract coefficient matrices A_1, ..., A_p
   d. Generate predictions
```

#### 3a. Building Design Matrices

`build_var_design_batch()` at `src/models/var_pytorch.py:35-132`:

For VAR(p), the design matrix X contains:
```
X row at time t: [1, Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
                  ^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               intercept        lagged values
```

**Example for p=2, n_assets=3:**
```
X[t] = [1, y1_{t-1}, y2_{t-1}, y3_{t-1}, y1_{t-2}, y2_{t-2}, y3_{t-2}]
```

The function uses vectorized indexing (`src/models/var_pytorch.py:97-131`):
```python
# Create time indices for all lags
base_indices = torch.arange(T, device=device)
lag_offsets = torch.arange(p - 1, -1, -1, device=device)
time_indices = lag_offsets.unsqueeze(1) + base_indices.unsqueeze(0)

# Gather lagged values in one operation
lagged_all = torch.gather(windows_expanded, 2, time_indices_for_gather)
```

#### 3b. Batched OLS Solution

At `src/models/var_pytorch.py:261-271`:

```python
# Compute X'X and X'Y for all windows in batch
XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)  # (batch, n_features, n_features)
XtY = torch.bmm(X_batch.transpose(1, 2), Y_batch)  # (batch, n_features, n_assets)

# Solve the normal equations: β = (X'X)^{-1} X'Y
beta = torch.linalg.solve(XtX, XtY)  # (batch, n_features, n_assets)
```

This solves all days simultaneously using batched matrix multiplication (`torch.bmm`).

#### 3c. Extracting Coefficients

At `src/models/var_pytorch.py:277-286`:

```python
# Remove intercept term
beta_no_intercept = beta[:, 1:, :]  # (batch, n_assets*p, n_assets)

# Reshape to (batch, p, n_assets, n_assets)
A_matrices = beta_no_intercept.view(n_test_days, p, n_assets, n_assets)

# Transpose so coefficients[d, k, i, j] = effect of j on i
A_matrices = A_matrices.transpose(-2, -1)
```

**Coefficient convention:**
- `coefficients[day, lag-1, i, j]` = effect of asset j at that lag on asset i

---

### Step 4: Optimal Lag Selection

`select_optimal_p()` at `src/models/var_pytorch.py:295-351`:

Uses the first `validation_days` of the test period to select optimal p:

```python
# Compute RMSE for each p
squared_errors = (forecasts_val - actuals_val.unsqueeze(2)) ** 2
rmse_per_p = torch.sqrt(squared_errors.mean(dim=(0, 1)))

# Select p with minimum RMSE
p_optimal_idx = torch.argmin(rmse_per_p).item()
```

---

### Step 5: Return VARXResult

At `src/models/var_pytorch.py:460-469`:

```python
return VARXResult(
    forecasts=forecasts,              # (n_assets, n_output_days)
    forecasts_all=forecasts_all,      # (n_assets, n_output_days, p_max)
    p_optimal=p_optimal,              # (n_output_days,)
    coefficients=coefficients_output, # (n_output_days, p_max, n_assets, n_assets)
    ...
)
```

The `VARXResult` class in `src/results.py:7-131` provides:
- `method` property: Returns `"VAR"` (line 32-34)
- `get_leadlag_matrix(day_idx, lag)`: Extracts coefficient matrix as DataFrame (line 58-82)

---

## Part 2: OR-VARX with Double Machine Learning

### The Problem: Confounding

Plain VAR assumes asset returns only depend on their own lagged values. But in reality, **common factors** (like VIX) affect all assets simultaneously, creating spurious correlations.

**OR-VARX solves this** by using Double Machine Learning to "orthogonalize" (remove) the confounding effect.

---

### Step 1: Calling `fit_orvarx()`

From `scripts/example_orvarx_usage.py`:

```python
result = fit_orvarx(
    Y, W,
    p_max=10,
    config=GridConfig(),  # Uses lookback=766 by default
    validation_days=20,
    learner_name='extra_trees',  # Tree-based learner for nuisance estimation
)
```

---

### Step 2: The DML Algorithm

The core DML algorithm in `src/models/dml_pytorch.py:625-787`:

```
For each test day (day_idx = lookback, lookback+1, ..., n_days-1):
  For each lag order p in 1..p_max:
    1. Build DML data (outcome, treatment, controls)
    2. Cross-fit to get residuals
    3. Estimate θ from residuals (deconfounded coefficients)
    4. Generate forecast
```

---

### Step 3: Building DML Data

`build_dml_data()` at `src/models/dml_pytorch.py:37-127`:

**Key insight:** We reframe VAR as a causal inference problem:
- **Outcome (Y)**: Current returns `Y_t`
- **Treatment (T)**: Lagged returns `[Y_{t-1}, ..., Y_{t-p}]` (what we want causal effect of)
- **Controls (W)**: Confounders `[W_{t-1}, ..., W_{t-p}]` (lagged only, no W_t to avoid lookahead bias)

```python
# Outcome: Y_t for t = p, p+1, ..., lookback-1
outcome = Y_window[p:, :]  # Shape: (T, n_assets)

# Treatment: lagged returns [Y_{t-1}, ..., Y_{t-p}]
lags = torch.stack([Y_window[p - lag:lookback - lag, :] for lag in range(1, p + 1)], dim=2)
treatment = lags.permute(0, 2, 1).reshape(T, n_assets * p)  # Shape: (T, n_assets * p)

# Controls: [W_{t-1}, ..., W_{t-p}] (no W_t to avoid lookahead bias)
controls = lagged_controls.permute(0, 2, 1).reshape(T, n_confounders * p)  # Shape: (T, n_confounders*p)
```

---

### Step 4: Grid-Based Cross-Fitting (Memoized)

**Why cross-fitting?** If we used the same data to estimate nuisance functions (Y~W, T~W) and then estimate θ, we'd have overfitting bias. Cross-fitting prevents this.

**The Performance Challenge:** Traditional per-day cross-fitting would require ~33M model trainings for a year of daily forecasts.

**The Solution: Grid-Based Memoization** (`src/models/dml_pytorch.py:243-537`):

Instead of retraining models for every day, the implementation uses a fixed grid:
- Pre-train models on grid points every 21 days
- Cache trained models in `ModelCache`
- Reuse cached models for residual computation
- This reduces trainings from ~33M to ~131K (~250x speedup)

**Grid Configuration** (`src/modules/grid_config.py:19-42`):
```python
@dataclass
class GridConfig:
    train_size: int = 504   # 2 years training window
    test_size: int = 21     # ~1 month test window
    lookback_base: int = 756  # 3 years
    p_max_offset: int = 10    # Buffer for VAR lags

    @property
    def lookback(self) -> int:
        return self.lookback_base + self.p_max_offset  # 766
```

**Key Functions:**

1. **`get_active_folds_for_day()`** at `src/models/dml_pytorch.py:284-330`:
   Determines which grid folds provide residuals for a given day's lookback window.

2. **`ensure_fold_trained()`** at `src/models/dml_pytorch.py:381-449`:
   Checks cache for a fold; trains and caches if not found.

3. **`compute_residuals()`** at `src/models/dml_pytorch.py:452-537`:
   Computes residuals for a day using cached fold models.

**Algorithm:**
```
For each test day:
  1. Determine active grid folds via get_active_folds_for_day()
  2. For each active fold:
     - If not cached: train models (Y~W, T~W) and cache in ModelCache
     - Predict on fold's test window
     - Compute residuals: Y_res = Y - E[Y|W], T_res = T - E[T|W]
  3. Concatenate residuals from all active folds
```

**Model Cache** (`src/modules/model_cache.py`):
```python
class ModelCache:
    """Cache for storing trained fold models by (grid_idx, p)."""
    def get_fold(self, grid_idx: int, p: int) -> Optional[FoldModels]
    def add_fold(self, grid_idx: int, p: int, fold: FoldModels)
```

**Nuisance model training** uses `MultiOutputRegressor` for parallel prediction:
```python
# Train models for all assets/treatments at once
model_y = get_multi_output_regressor(learner_name)  # XGBoost, LightGBM, etc.
model_t = get_multi_output_regressor(learner_name)
model_y.fit(controls_train, outcome_train)
model_t.fit(controls_train, treatment_train)
```

The `RollingWindowSplit` class in `src/modules/rolling_split.py:6-99` provides the underlying CV split logic.

---

### Step 5: Estimating Deconfounded Coefficients θ

`estimate_theta()` at `src/models/dml_pytorch.py:130-174`:

After cross-fitting, we have:
- `Y_residuals`: Variation in Y not explained by confounders
- `T_residuals`: Variation in T not explained by confounders

The **DML estimator** is simply OLS on the residuals:

```
θ = (T_res' T_res)^{-1} T_res' Y_res
```

In code (`src/models/dml_pytorch.py:162-166`):
```python
TtT = torch.mm(T_residuals.T, T_residuals)  # (n_treatments, n_treatments)
TtY = torch.mm(T_residuals.T, Y_residuals)  # (n_treatments, n_assets)
theta = torch.linalg.solve(TtT, TtY)        # (n_treatments, n_assets)
```

**θ gives the CAUSAL effect** of lagged returns on current returns, with confounding removed.

---

### Step 6: Computing Standard Errors

`compute_se_oracle()` at `src/models/dml_pytorch.py:177-240`:

For significance testing (used by ORACLE-VARX), computes standard errors:

```python
# Residual variance
RSS = sum((Y_res - T_res * θ)²)
σ² = RSS / (n - k)

# Standard error formula
SE[i, j] = sqrt(diag((T'T)^{-1})[i] * σ²[j])
```

---

### Step 7: Generating Forecasts

At `src/models/dml_pytorch.py:596-620` (within `fit_orvarx_single_day()`):

For the forecast day, we need to residualize the prediction features too:

```python
# Build treatment features: [Y_{day_idx-1}, ..., Y_{day_idx-p}]
indices = torch.arange(day_idx - 1, day_idx - p - 1, -1, device=device)
treatment_pred = Y[indices, :].reshape(1, n_assets * p)

# Residualize treatment using the last active fold's model
T_hat = last_fold.model_t.predict(controls_pred_np)
T_pred_residual = treatment_pred_np - T_hat

# Forecast: Y_hat = T_residualized * θ
forecast = torch.mm(T_pred_residual_torch, theta).squeeze(0)
```

The full single-day pipeline is in `fit_orvarx_single_day()` at `src/models/dml_pytorch.py:540-622`.

---

### Step 8: Return VARXResult

At `src/models/dml_pytorch.py:778-787`:

```python
return VARXResult(
    forecasts=forecasts,
    forecasts_all=forecasts_all,
    p_optimal=p_optimal,
    p_max=p_max,
    coefficients=coefficients,
    confounder_names=confounder_names,  # ["VIX"] - marks this as OR-VARX
    ...
)
```

**Note:** The optimal lag `p_optimal` is selected globally based on validation RMSE (lines 755-769), not per-day.

The `VARXResult.method` property (`src/results.py:32-34`) returns `"OR-VARX"` because `confounder_names` is non-empty.

---

## Summary: Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VAR FLOW                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Y (returns)  ──► build_var_design_batch() ──► Batched OLS ──► β        │
│                         │                          │                     │
│                    Design matrix X              (X'X)^{-1}X'Y            │
│                    [1, Y_{t-1}, ...]                                     │
│                                                                          │
│  β ──► Extract A matrices ──► Forecast: Y_t = c + A_1*Y_{t-1} + ...     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    OR-VARX FLOW (DML with Grid Memoization)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Y (returns)    ──┐                                                      │
│  W (confounders) ─┴──► fit_orvarx() ──► For each day and p:             │
│                              │                                           │
│                              ▼                                           │
│                    ┌─────────────────────┐                               │
│                    │   build_dml_data()   │                              │
│                    │   outcome, treatment │                              │
│                    │   controls           │                              │
│                    └──────────┬───────────┘                              │
│                               │                                          │
│                               ▼                                          │
│                    ┌─────────────────────┐                               │
│                    │  compute_residuals() │                              │
│                    │  (Grid-based caching)│                              │
│                    │                      │                              │
│                    │  1. get_active_folds │                              │
│                    │  2. For each fold:   │                              │
│                    │    - Check ModelCache│                              │
│                    │    - If miss: train  │◄── XGBoost/LightGBM/RF       │
│                    │      (ensure_fold_   │                              │
│                    │       trained)       │                              │
│                    │    - Get residuals   │                              │
│                    └──────────┬───────────┘                              │
│                               │                                          │
│                               ▼                                          │
│                    ┌─────────────────────┐                               │
│                    │  estimate_theta()    │                              │
│                    │  θ = (T'T)^{-1} T'Y  │                              │
│                    │  (Deconfounded coef) │                              │
│                    └──────────┬───────────┘                              │
│                               │                                          │
│                               ▼                                          │
│                    Forecast: Y_hat = T_residualized * θ                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key File Reference Summary

| Component | File | Key Functions/Lines |
|-----------|------|---------------------|
| VAR fitting | `src/models/var_pytorch.py` | `fit_var()`:354-469, `batch_var_all_days()`:194-292 |
| Design matrices | `src/models/var_pytorch.py` | `build_var_design_batch()`:35-132 |
| Lag selection | `src/models/var_pytorch.py` | `select_optimal_p()`:295-351 |
| OR-VARX fitting | `src/models/dml_pytorch.py` | `fit_orvarx()`:625-787 |
| Single-day OR-VARX | `src/models/dml_pytorch.py` | `fit_orvarx_single_day()`:540-622 |
| DML data building | `src/models/dml_pytorch.py` | `build_dml_data()`:37-127 |
| Grid-based cross-fitting | `src/models/dml_pytorch.py` | `compute_residuals()`:452-537, `get_active_folds_for_day()`:284-330, `ensure_fold_trained()`:381-449 |
| θ estimation | `src/models/dml_pytorch.py` | `estimate_theta()`:130-174 |
| Standard errors | `src/models/dml_pytorch.py` | `compute_se_oracle()`:177-240 |
| Grid config | `src/modules/grid_config.py` | `GridConfig`:19-42 |
| Model cache | `src/modules/model_cache.py` | `ModelCache`:40-113, `FoldModels`:14-37 |
| Rolling CV split | `src/modules/rolling_split.py` | `RollingWindowSplit`:6-99 |
| Learner factory | `src/modules/factory.py` | `get_regressor()`:16-56, `get_multi_output_regressor()`:59-77 |
| Results container | `src/results.py` | `VARXResult`:7-131 |
| Data loading | `src/data/loader.py` | `load_test_data()`:350-392 |
