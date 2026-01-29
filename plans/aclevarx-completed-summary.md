# ACLE-VARX Implementation Summary

## Overview

ACLE-VARX applies the same significance-testing methodology as ORACLE-VARX but to **plain VAR models** (without DML/orthogonalization).

| Model | p-selection | α-selection | Confounders |
|-------|-------------|-------------|-------------|
| **VAR** | Validation RMSE | N/A | No |
| **ACLE-VARX** | Significance tests | Rolling validation RMSE | No |
| **OR-VARX** | Validation RMSE | N/A | Yes (DML) |
| **ORACLE-VARX** | Significance tests | Rolling validation RMSE | Yes (DML) |

## Architecture

```
batch_var_all_days_with_se()  <- Batched VAR with standard errors
    |
    +---> fit_aclevarx()      <- p-selection via significance + α-selection via rolling validation
```

## Implementation Status

| Step | File | Status |
|------|------|--------|
| 1 | `src/models/acle_var.py` | ✓ `batch_var_all_days_with_se()` + `fit_aclevarx()` |
| 2 | `src/modules/batch_utils.py` | ✓ `batched_benjamini_hochberg()` |
| 3 | `src/results.py` | ✓ `ACLEVARXResult` with SE_all field |
| 4 | `src/models/__init__.py` | ✓ Export `fit_aclevarx` |

## Data Structures and Memoization

### Key Data Structures

| Variable | Shape | Description |
|----------|-------|-------------|
| `Y` | `(n_days, n_assets)` | Raw asset returns (input) |
| `forecasts_all_batched` | `(n_assets, n_total_test_days, p_max)` | **MEMOIZED**: Forecasts for every (day, lag) combination |
| `theta_all` | `(n_total_test_days, p_max, n_assets, n_assets)` | **MEMOIZED**: VAR coefficients |
| `SE_all` | `(n_total_test_days, p_max, n_assets, n_assets)` | **MEMOIZED**: Standard errors for coefficients |
| `actuals` | `(n_total_test_days, n_assets)` | Realized returns for test days |
| `p_alpha_all` | `(n_total_test_days, n_alphas)` | **MEMOIZED**: Selected p for each (day, α) |
| `forecasts_at_p_alpha` | `(n_total_test_days, n_alphas, n_assets)` | **DERIVED**: Forecasts indexed by p_alpha |
| `alpha_optimal` | `(n_output_days,)` | Per-day optimal α index |
| `p_optimal_all_days` | `(n_output_days,)` | Per-day optimal p |

### What is Memoized (Computed Once, Reused)

**Phase 1 (Expensive - O(n_days × p_max × OLS_cost)):**
- `forecasts_all_batched[:, d, p-1]` = forecast for day `d` using VAR(p)
- `theta_all[d, p-1, :, :]` = coefficient matrix for day `d`, lag `p`
- `SE_all[d, p-1, :, :]` = standard errors for day `d`, lag `p`

**Phase 2 (Moderate - O(n_alphas × p_max × n_days)):**
- `p_alpha_all[d, α_idx]` = selected p for day `d` at significance level `α`

**Phase 3 uses memoized data with O(1) lookups:**
- `forecasts_at_p_alpha[d, α, :]` = lookup into `forecasts_all_batched` using `p_alpha_all[d, α]`
- Rolling MSE computed via cumsum trick (O(1) per output day)

## Full Training & Inference Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Batched VAR Estimation with SE (MEMOIZED)                     │
│ ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│ For each day d ∈ [0, n_total_test_days):                               │
│   training_window = Y[d : d + lookback]  # Sliding window              │
│                                                                         │
│   For each lag p ∈ [1, p_max]:                                         │
│     • Build VAR design matrix: X = [1, Y_{t-1}, ..., Y_{t-p}]          │
│     • Solve OLS: β = (X'X)^{-1} X'Y                                    │
│     • Compute residuals: ε = Y - Xβ                                    │
│     • Estimate σ² = Σε² / (T - k)                                      │
│     • Compute SE = sqrt(diag((X'X)^{-1}) × σ²)                         │
│     • Store: forecasts_all_batched[:, d, p-1] = prediction             │
│     • Store: theta_all[d, p-1, :, :] = coefficients                    │
│     • Store: SE_all[d, p-1, :, :] = standard errors                    │
│                                                                         │
│ Output: forecasts_all_batched, theta_all, SE_all, actuals              │
│ Complexity: O(n_total_test_days × p_max × OLS_cost)                    │
│ This is the EXPENSIVE phase - but computed ONCE                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Batched Significance Testing (MEMOIZED)                       │
│ ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│ For each α ∈ alpha_grid:                                               │
│   p_selected = tensor of 1s (shape: n_total_test_days)                 │
│                                                                         │
│   For p = 2 to p_max:                                                  │
│     # Vectorized across all days                                       │
│     θ_new = theta_all[:, p-1, :, :]      # New lag-p coefficients      │
│     SE_new = SE_all[:, p-1, :, :]                                      │
│     z = |θ_new / SE_new|                 # z-statistics                │
│     p_vals = 2 × (1 - Φ(z))              # Two-tailed p-values         │
│                                                                         │
│     # Benjamini-Hochberg FDR correction (batched)                      │
│     reject = batched_benjamini_hochberg(p_vals.flatten(), α)           │
│     is_significant = reject.any(dim=coefficients)                      │
│                                                                         │
│     # Update p_selected where significant AND still active             │
│     still_active = (p_selected == p - 1)                               │
│     p_selected = where(is_significant & still_active, p, p_selected)   │
│                                                                         │
│   Store: p_alpha_all[:, α_idx] = p_selected                            │
│                                                                         │
│ Output: p_alpha_all (n_total_test_days, n_alphas)                      │
│ Complexity: O(n_alphas × p_max × n_total_test_days)                    │
│ This is MODERATE cost - vectorized, no OLS                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Rolling α-Selection (uses memoized data)                      │
│ ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│ Step 3a: Precompute forecasts at p_alpha (one-time lookup)             │
│   For α_idx in range(n_alphas):                                        │
│     For d in range(n_total_test_days):                                 │
│       p = p_alpha_all[d, α_idx]                                        │
│       forecasts_at_p_alpha[d, α_idx, :] = forecasts_all_batched[:, d, p-1] │
│                                                                         │
│ Step 3b: Compute squared errors                                        │
│   squared_errors = (forecasts_at_p_alpha - actuals.unsqueeze(1))²      │
│   mse_per_day_alpha = squared_errors.mean(dim=assets)                  │
│                                                                         │
│ Step 3c: Cumsum for O(1) rolling window                                │
│   cumsum = cumsum(mse_per_day_alpha, dim=0)                            │
│   cumsum_padded = concat([zeros(1, n_alphas), cumsum], dim=0)          │
│                                                                         │
│ Step 3d: Rolling α-selection for each output day                       │
│   For d in range(n_output_days):                                       │
│     day_idx = validation_days + d        # Full test index             │
│     val_start = d                        # = day_idx - validation_days │
│     val_end = day_idx                    # Exclusive                   │
│                                                                         │
│     # O(1) rolling MSE via cumsum difference                           │
│     rolling_mse = (cumsum_padded[val_end] - cumsum_padded[val_start])  │
│                   / validation_days                                    │
│                                                                         │
│     alpha_optimal[d] = argmin(rolling_mse)                             │
│                                                                         │
│ Output: alpha_optimal (n_output_days,)                                 │
│ Complexity: O(n_total_test_days × n_alphas) for Step 3a                │
│           + O(n_output_days) for Step 3d (cumsum trick!)               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Forecast Retrieval (simple lookups)                           │
│ ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│ For d in range(n_output_days):                                         │
│   day_idx_full = validation_days + d                                   │
│   α* = alpha_optimal[d]                  # Per-day optimal α           │
│   p* = p_alpha_all[day_idx_full, α*]     # p at optimal α              │
│   forecasts_output[d] = forecasts_all_batched[:, day_idx_full, p*-1]   │
│                                                                         │
│ Output: forecasts_final (n_assets, n_output_days)                      │
│ Complexity: O(n_output_days) - just index lookups                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Rolling Window Visualization

```
Calendar days:        0    ...   513   514   515   ...   534   535   536   ...
                      |<-- lookback -->|
                                       |<-------- test period -------->|

Full test index:                        0     1    ...    20    21    22   ...
                                       |<-- validation -->|
                                                          ^ Output day 0 (d=0)
                                                            forecasts day 21
                                                            validated on [0,21)

                                             |<-- validation -->|
                                                                ^ Output day 1 (d=1)
                                                                  forecasts day 22
                                                                  validated on [1,22)
```

**Key insight**: The validation window "rolls" forward with each output day, so α is re-selected based on the most recent `validation_days` of performance.

## Complexity Analysis

| Phase | Operation | Complexity | Notes |
|-------|-----------|------------|-------|
| 1 | Batched VAR + SE | O(n_days × p_max × n_assets³) | Expensive but memoized |
| 2 | Significance tests | O(n_alphas × p_max × n_days) | Vectorized |
| 3a | Forecast lookup | O(n_days × n_alphas) | Index operations |
| 3b-c | Squared errors + cumsum | O(n_days × n_alphas) | Tensor ops |
| 3d | Rolling selection | O(n_output_days × n_alphas) | Cumsum trick: O(1) per day |
| 4 | Final retrieval | O(n_output_days) | Index lookups |

**Total**: Dominated by Phase 1 (OLS). Phases 3-4 add negligible overhead.

## Why the For Loops Are Not a Problem

| Loop Location | Iterations | Cost per Iteration | Bottleneck? |
|---------------|------------|-------------------|-------------|
| Phase 1: p loop | p_max (~10) | Full OLS | Yes (memoized) |
| Phase 2: α loop | n_alphas (~7) | Vectorized BH | No |
| Phase 2: p loop | p_max (~10) | Tensor ops | No |
| Phase 3a: α×d loops | ~10,000 | `.item()` + index | No (microseconds) |
| Phase 3d: d loop | ~1,500 | Cumsum slice | No (nanoseconds) |
| Phase 4: d loop | ~1,500 | Index lookup | No (nanoseconds) |

The expensive OLS computation in Phase 1 is **batched across all days simultaneously** using `torch.bmm` and `torch.linalg.solve`. The loops in Phases 3-4 are pure Python index operations on already-computed tensors.

## Output Shape Formula

```
n_total_test_days = n_days - lookback_var
n_output_days = n_total_test_days - validation_days
```

| Parameter | Default | Example |
|-----------|---------|---------|
| `lookback_var` | 514 | 514 |
| `validation_days` | 21 | 21 |
| `n_days` | - | 2000 |
| `n_total_test_days` | - | 1486 |
| `n_output_days` | - | 1465 |

## Function Signature

```python
def fit_aclevarx(
    Y: torch.Tensor,
    alpha_grid: List[float] = None,  # Default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    validation_days: int = 21,  # For rolling α-selection
    asset_names: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    verbose: bool = False,
) -> ACLEVARXResult:
```

## Design Decisions

1. **No confounders** - ACLE-VARX is plain VAR, not DML
2. **Significance test only** - Test whether new lag-p coefficients are significant
3. **Benjamini-Hochberg FDR correction** - Required for multiple hypothesis testing
4. **Rolling α-selection** - Per-day optimal α using trailing validation window
5. **Efficient memoization** - Phase 1 precomputes all forecasts; Phase 3-4 are O(1) lookups
6. **Cumsum trick** - Rolling window MSE computed in O(1) per day

## Example Usage

### Script Location
`scripts/example_aclevarx_usage.py`

### How to Run
```bash
python scripts/example_aclevarx_usage.py
```

### Quick Test
```python
import torch
from src.models.acle_var import fit_aclevarx
from src.modules.grid_config import GridConfig

# Generate synthetic data (no confounders needed)
n_days, n_assets = 2000, 10
Y = torch.randn(n_days, n_assets)

# Fit model
result = fit_aclevarx(
    Y,
    p_max=10,
    validation_days=21,
)

# Check output
print(f"Method: {result.method}")  # 'ACLE-VARX'
print(f"Forecasts shape: {result.forecasts.shape}")  # (10, 1465)
print(f"alpha_optimal shape: {result.alpha_optimal.shape}")  # (1465,) - per-day!

# α varies across days (rolling selection)
alpha_counts = torch.bincount(result.alpha_optimal, minlength=len(result.alpha_grid))
for i, alpha in enumerate(result.alpha_grid):
    print(f"  α={alpha}: selected {alpha_counts[i].item()} days")
```

### Expected Output
```
Method: ACLE-VARX
Forecasts shape: torch.Size([10, 1465])
alpha_optimal shape: torch.Size([1465])
  α=0.01: selected 829 days
  α=0.05: selected 69 days
  α=0.1: selected 168 days
  α=0.15: selected 116 days
  α=0.2: selected 100 days
  α=0.25: selected 70 days
  α=0.3: selected 113 days
```

## Comparison: ACLE-VARX vs ORACLE-VARX

| Aspect | ACLE-VARX | ORACLE-VARX |
|--------|-----------|-------------|
| Base model | Plain VAR | DML (Double ML) |
| Confounders | No | Yes |
| p-selection | Significance tests | Significance tests |
| α-selection | Rolling validation | Rolling validation |
| Core function | `batch_var_all_days_with_se()` | `_fit_orvarx_core()` |
| Lookback default | 514 | 1018 |
| Use case | No confounding | Confounding present |
