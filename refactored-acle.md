# ACLE-VARX Refactoring Summary

## Overview

Extracted duplicate rolling α-selection logic into a shared utility function to reduce code duplication between `oracle_var.py` and `acle_var.py`.

## Changes Made

### 1. New Shared Function: `rolling_alpha_selection()`

**Location:** `src/models/var_pytorch.py`

Added a new utility function that implements rolling validation window α-selection:

```python
def rolling_alpha_selection(
    forecasts_all_batched: torch.Tensor,  # (n_assets, n_total_test_days, p_max)
    p_alpha_all: torch.Tensor,            # (n_total_test_days, n_alphas)
    actuals: torch.Tensor,                # (n_total_test_days, n_assets)
    validation_days: int,
    alpha_grid: List[float],
    verbose: bool = True,
    use_greek_symbol: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], torch.Tensor]:
```

**Returns:**
- `alpha_optimal`: Optimal α index for each output day
- `alpha_counts`: Count of days each α was selected
- `alpha_percentages`: Percentage of days each α was selected
- `p_optimal_all_days`: The p value to use for each output day

**Algorithm:**
- For each output day d, uses trailing validation window of `validation_days`
- Computes MSE for each α over the window using cumsum trick (O(1) per day)
- Selects α with minimum MSE

### 2. Updated `oracle_var.py`

- Added import: `from src.models.var_pytorch import rolling_alpha_selection`
- Replaced ~75 lines of Phase 3 code with single function call:
  ```python
  alpha_optimal, alpha_counts, alpha_percentages, p_optimal_all_days = rolling_alpha_selection(
      forecasts_all_batched=forecasts_all_batched,
      p_alpha_all=p_alpha_all,
      actuals=actuals,
      validation_days=validation_days,
      alpha_grid=alpha_grid,
      verbose=True,
      use_greek_symbol=True,  # Uses "α" in output
  )
  ```

### 3. Updated `acle_var.py`

- Added import: `from src.models.var_pytorch import ..., rolling_alpha_selection`
- Replaced ~75 lines of Phase 3 code with single function call (same as above, but `use_greek_symbol=False`)
- Fixed confusing transpose comments in coefficient matrix handling:
  ```python
  # Before (confusing):
  # First reshape: (batch, p, n_assets, n_assets) where [:, k, j, i] = effect of j on i
  # Transpose to get [:, k, i, j] = effect of j on i  # <-- contradictory!

  # After (clear):
  # beta_no_intercept[:, :, i] contains coeffs predicting asset i
  # After reshape: (batch, p, n_assets, n_assets) where [:, k, :, i] = coeffs for asset i
  # After transpose: [:, k, i, j] = effect of asset j at lag k+1 on asset i
  ```

### 4. Dependencies Added to `var_pytorch.py`

```python
import numpy as np
from scipy import stats
```

## Benefits

1. **DRY Principle:** Eliminated ~150 lines of duplicated code
2. **Maintainability:** Single source of truth for the rolling α-selection algorithm
3. **Testability:** Shared function can be unit tested independently
4. **Consistency:** Both models guaranteed to use identical α-selection logic
5. **Documentation:** Clear docstring explaining the algorithm and parameters

## Testing

- Import test: All imports successful
- Unit test: `rolling_alpha_selection()` produces correct output shapes
- Integration test: ACLE-VARX runs successfully with refactored code
