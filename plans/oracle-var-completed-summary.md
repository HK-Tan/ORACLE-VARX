# ORACLE-VARX Cached Implementation - Completed Summary

## Overview

ORACLE-VARX is a **post-processing layer** on cached OR-VARX results. The key insight is that `fit_orvarx_batched()` already computes θ for each p ∈ [1, p_max]. We extended it to also compute SE, then run significance tests as post-processing.

## Implementation Status

| Step | File | Status |
|------|------|--------|
| 1 | `src/models/dml_pytorch.py` | ✓ Added `return_se` param + batched SE computation |
| 2 | `src/modules/batch_utils.py` | ✓ Added `batched_benjamini_hochberg()` |
| 3 | `src/models/oracle_var.py` | ✓ Created `fit_oracle_varx_cached()` |
| 4 | `src/results.py` | ✓ Added optional `SE_all` field to `ORACLEVARXResult` |

## Key Changes

### Step 1: Batched SE Computation (`src/models/dml_pytorch.py`)

- Added `return_se: bool = False` parameter to `fit_orvarx_batched()`
- When `return_se=True`, computes standard errors using:
  ```
  SE[i,j,k] = sqrt(diag((T'T)^{-1})[j] * σ²[i,k])
  ```
- Returns tuple `(VARXResult, SE_tensor)` when enabled
- Fully backward compatible (default returns same as before)

### Step 2: Batched Benjamini-Hochberg (`src/modules/batch_utils.py`)

- Added `batched_benjamini_hochberg(p_values, alpha)` function
- Vectorized FDR correction for multiple hypothesis testing
- Accepts 1D or 2D tensors, handles batching automatically
- Returns boolean tensor indicating which hypotheses to reject

### Step 3: Main Function (`src/models/oracle_var.py`)

Created `fit_oracle_varx_cached()` implementing:

**Phase 1: Batched DML Estimation**
- Single call to `fit_orvarx_batched(..., return_se=True)`
- Gets θ_all, SE_all, forecasts_all for all lags p ∈ [1, p_max]

**Phase 2: Batched Significance Testing**
```
For each α in alpha_grid:
  For each day d (batched):
    p_selected = 1
    For p in [2, p_max]:
      θ_new = θ_all[p][d, (p-1)*m : p*m, :]
      SE_new = SE_all[p][d, (p-1)*m : p*m, :]
      z_new = |θ_new / SE_new|
      p_vals = 2 * (1 - norm.cdf(z_new))
      reject = batched_benjamini_hochberg(p_vals, α)
      if reject.any(): p_selected = p
      else: break
    Store p_α[d, α_idx] = p_selected
```

**Phase 3: α Selection via Validation**
```
For each α in alpha_grid:
  forecast_α = forecasts_all[validation_days, :, p_α]
  RMSE[α] = sqrt(mean((forecast_α - actuals)²))
α_optimal = argmin(RMSE)
```

**Phase 4: Forecast Retrieval**
```
For each output day d:
  p* = p_α[d, α_optimal]
  forecast[d] = forecasts_all[d, :, p*]
```

### Step 4: Result Class (`src/results.py`)

- Added `SE_all: Optional[torch.Tensor] = None` field to `ORACLEVARXResult`
- Updated `save()` and `load()` methods for persistence
- Backward compatible with existing saved results

## Function Signature

```python
def fit_oracle_varx_cached(
    Y: torch.Tensor,
    W: torch.Tensor,
    alpha_grid: List[float] = None,  # Default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    validation_days: int = 21,
    asset_names: Optional[List[str]] = None,
    confounder_names: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
) -> ORACLEVARXResult:
```

## Design Decisions

1. **No treatment → confounder migration** - Empirically rare, not DAG-consistent
2. **No drift test** - Conceptually flawed (different residualizations make comparison invalid)
3. **Significance test only** - Test whether new lag-p coefficients are significant
4. **Benjamini-Hochberg FDR correction** - Required for multiple hypothesis testing
5. **Batched with PyTorch** - Vectorized, no for loops
6. **α tuned on validation** - Same approach as p tuning in OR-VARX

## Performance Expectations

- **Previous approach**: O(n_days × n_alphas) calls to `fit_orvarx_single_day()` + re-fits
- **Cached approach**: O(p_max) calls to pre-compute residuals, then O(1) post-processing
- **Expected speedup**: ~100x or more
