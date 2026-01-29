# ORACLE-VARX Implementation Summary

## Overview

ORACLE-VARX shares a common DML core with OR-VARX via `_fit_orvarx_core()`.
The key difference is p-selection:
- **OR-VARX**: p-selection via validation RMSE
- **ORACLE-VARX**: p-selection via significance tests + α-selection via validation RMSE

## Architecture

```
_fit_orvarx_core()           <- Core DML: fold training, residuals, batched OLS
    |
    +---> fit_orvarx_batched()       <- p-selection via validation RMSE
    |
    +---> fit_oraclevarx_batched()   <- p-selection via significance + α-selection via validation RMSE
```

## Implementation Status

| Step | File | Status |
|------|------|--------|
| 1 | `src/models/dml_pytorch.py` | ✓ `_fit_orvarx_core()` + refactored `fit_orvarx_batched()` |
| 2 | `src/modules/batch_utils.py` | ✓ `batched_benjamini_hochberg()` |
| 3 | `src/models/oracle_var.py` | ✓ `fit_oraclevarx_batched()` |
| 4 | `src/results.py` | ✓ `ORACLEVARXResult` with SE_all field |
| 5 | `src/models/__init__.py` | ✓ Export `fit_oraclevarx_batched` |

## Key Functions

### `_fit_orvarx_core()` (dml_pytorch.py)

Core DML computation shared by both models. Performs:
1. Pre-training all required grid folds
2. Pre-computing residuals for all test windows
3. Batched OLS for all days and all lags

**Returns:**
- `forecasts_all`: shape `(n_total_test_days, n_assets, p_max)`
- `coefficients`: shape `(n_total_test_days, p_max, n_assets, n_assets)`
- `standard_errors`: shape `(n_total_test_days, p_max, n_assets, n_assets)`
- `actuals`: shape `(n_total_test_days, n_assets)` - for caller's validation

**Key properties:**
- No validation trimming
- No p-selection
- Always computes SE (needed by both models)
- Returns ALL test days: `n_total_test_days = n_days - lookback`

### `fit_orvarx_batched()` (dml_pytorch.py)

Calls `_fit_orvarx_core()` + p-selection via validation RMSE.

**Parameters:**
- `validation_days`: for p-selection via validation RMSE

**Output shape:**
- `n_output_days = (n_days - lookback) - validation_days`

### `fit_oraclevarx_batched()` (oracle_var.py)

Calls `_fit_orvarx_core()` + significance-based p-selection + α-selection via validation RMSE.

**Parameters:**
- `validation_days`: for α-selection (the ONLY validation - p is automatic via significance tests)

**Output shape:**
- `n_output_days = (n_days - lookback) - validation_days` (SAME formula as OR-VARX!)

## Output Shape Formula

Both models now use the SAME intuitive formula:

```
n_output_days = (n_days - lookback) - validation_days
```

| Model | Formula | Example (n_days=1061, lookback=1018, validation=21) |
|-------|---------|-----------------------------------------------------|
| OR-VARX | `(n_days - lookback) - validation_days` | `43 - 21 = 22` |
| ORACLE-VARX | `(n_days - lookback) - validation_days` | `43 - 21 = 22` |

## Function Signature

```python
def fit_oraclevarx_batched(
    Y: torch.Tensor,
    W: torch.Tensor,
    alpha_grid: List[float] = None,  # Default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    validation_days: int = 21,  # For α-selection (the ONLY validation)
    asset_names: Optional[List[str]] = None,
    confounder_names: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
) -> ORACLEVARXResult:
```

## Algorithm (fit_oraclevarx_batched)

**Phase 1: Core DML Estimation**
- Call `_fit_orvarx_core()` to get coefficients and SEs for all p ∈ [1, p_max]
- Returns ALL test days (no trimming)

**Phase 2: Batched Significance Testing**
```
For each α in alpha_grid:
  For each day d (batched):
    p_selected = 1
    For p in [2, p_max]:
      θ_new = θ_all[d, p-1, :, :]  # New lag-p coefficients
      SE_new = SE_all[d, p-1, :, :]
      z_new = |θ_new / SE_new|
      p_vals = 2 * (1 - norm.cdf(z_new))
      reject = batched_benjamini_hochberg(p_vals, α)
      if reject.any(): p_selected = p
      else: break
    Store p_α[d, α_idx] = p_selected
```

**Phase 3: α Selection via Validation (the ONLY validation)**
```
For each α in alpha_grid:
  forecast_α = forecasts_all[validation_days, :, p_α]
  RMSE[α] = sqrt(mean((forecast_α - actuals)²))
α_optimal = argmin(RMSE)
```

**Phase 4: Forecast Retrieval and Trimming**
```
For each output day d:
  p* = p_α[d, α_optimal]
  forecast[d] = forecasts_all[d, :, p*]
n_output_days = n_total_test_days - validation_days
```

## Design Decisions

1. **No treatment → confounder migration** - Empirically rare, not DAG-consistent
2. **No drift test** - Conceptually flawed (different residualizations make comparison invalid)
3. **Significance test only** - Test whether new lag-p coefficients are significant
4. **Benjamini-Hochberg FDR correction** - Required for multiple hypothesis testing
5. **Batched with PyTorch** - Vectorized, no for loops
6. **α tuned on validation** - Same approach as p tuning in OR-VARX
7. **Shared core function** - `_fit_orvarx_core()` eliminates code duplication
8. **Clean output shape formula** - Same formula for both models

## Benefits of Refactoring

1. **Clean API**: No more `validation_days=1` hack
2. **Intuitive shapes**: Same formula for both models
3. **Single validation**: Each model has ONE validation purpose (OR-VARX for p, ORACLE-VARX for α)
4. **Minimal duplication**: Core DML computation shared
5. **Clear separation**: Core computation vs. model-specific selection logic

## Example Usage

### Script Location
`scripts/example_oraclevarx_usage.py`

### How to Run
```bash
python scripts/example_oraclevarx_usage.py
```

### What It Demonstrates
- Generates synthetic Y and W tensors
- Calls `fit_oraclevarx_batched()` with configurable parameters
- Verifies output shapes for all result fields:
  - `forecasts`: Final forecasts `(n_assets, n_output_days)`
  - `forecasts_all`: Forecasts for all alpha values `(n_assets, n_output_days, n_alphas)`
  - `p_optimal`: Selected lag order per day `(n_output_days,)`
  - `alpha_optimal`: Selected alpha per day `(n_output_days,)`
  - `alpha_grid`: List of alpha values tested
  - `method`: Should be `"ORACLE-VARX"`
- Reports timing and pass/fail status for each check

### Quick Test
```python
import torch
from src.models.oracle_var import fit_oraclevarx_batched
from src.modules.grid_config import GridConfig

# Generate synthetic data
n_days, n_assets, n_confounders = 1061, 5, 1
Y = torch.randn(n_days, n_assets)
W = torch.randn(n_days, n_confounders)

# Fit model
result = fit_oraclevarx_batched(
    Y, W,
    p_max=3,
    validation_days=21,
    learner_name='lgbm',
)

# Check output
print(f"Method: {result.method}")  # 'ORACLE-VARX'
print(f"Forecasts shape: {result.forecasts.shape}")  # (5, 22)
print(f"Optimal α: {result.alpha_grid[result.alpha_optimal[0].item()]}")
```
