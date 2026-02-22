# OR-VARX Grid-Based Memoization Implementation Plan (COMPLETED)

## Problem Statement

Current `fit_orvarx` has O(n_days × p_max × n_folds × n_models) complexity:
- 5000 days × 10 p values × 6 folds × ~110 models ≈ **33 million model trainings**
- Consecutive days have 99.9% overlapping data but retrain from scratch

## Implementation Status: COMPLETED ✅

All phases of this plan have been implemented:

| Phase | Description | Status |
|-------|-------------|--------|
| Grid Config | Separate lookbacks for VAR (514) vs OR-VARX (1018) | ✅ Done |
| Model Cache | FoldModels + ModelCache with get_all_trained_folds() | ✅ Done |
| DML Functions | Grid-based cross-fitting, cached folds | ✅ Done |
| Batch Utils | Shared batched_ols() for VAR and OR-VARX | ✅ Done |
| Vectorized Inference | fit_orvarx_batched(), precompute_all_residuals() | ✅ Done |

## Key Insight

Pre-train models on a **fixed grid** of 21-day intervals and reuse them across consecutive days. New models trained only every 21 days.

## Proposed Grid Structure

```
Lookback (OR-VARX): 1018 days (504 tree training + 504 OLS + 10 p_max offset)
Lookback (VAR): 514 days (504 OLS + 10 p_max offset)
Training window per fold: 504 days (~2 years)
Test window per fold: 21 days (~1 month)
Folds within lookback: 12 folds × 21 days = 252 days coverage

For day_idx = 1018:
  Fold 0:  Train [0:504]    -> Test [504:525]
  Fold 1:  Train [21:525]   -> Test [525:546]
  Fold 2:  Train [42:546]   -> Test [546:567]
  ...
  Fold 11: Train [231:735]  -> Test [735:756]

For day_idx = 1019:
  - Add Fold 12: Train [252:756] -> Test [756:777]
  - Reuse Folds 0-11 for overlapping residuals
  - Only first day of Fold 12 is used
```

**Performance improvement: ~250x reduction** (33M → 131K model trainings)

---

## Implementation Steps

### Step 1: Create Grid Configuration ✅

**File:** `src/modules/grid_config.py` (NEW)

```python
@dataclass
class GridConfig:
    # Core window sizes
    ols_window: int = 504          # Rows for final OLS (both methods)
    tree_train_window: int = 504   # First-stage tree training (OR-VARX only)

    # Grid parameters
    test_size: int = 21            # Test window per fold (~1 month)

    # Lag parameters
    p_max_offset: int = 10         # Extra days for VAR lag computation

    @property
    def lookback_var(self) -> int:
        return self.ols_window + self.p_max_offset  # 514

    @property
    def lookback_orvarx(self) -> int:
        return self.tree_train_window + self.ols_window + self.p_max_offset  # 1018

    @property
    def train_size(self) -> int:
        return self.tree_train_window  # Backward compat alias
```

### Step 2: Create Model Cache ✅

**File:** `src/modules/model_cache.py` (NEW)

```python
@dataclass
class FoldModels:
    model_y: Any              # MultiOutputRegressor for Y ~ W
    model_t: Any              # MultiOutputRegressor for T ~ W
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    p: int

class ModelCache:
    def __init__(self, n_assets: int, n_confounders: int, p_max: int):
        # Dict: grid_idx -> Dict[p -> FoldModels]
        self.folds: Dict[int, Dict[int, FoldModels]] = {}

    def get_fold(self, grid_idx: int, p: int) -> Optional[FoldModels]
    def add_fold(self, grid_idx: int, p: int, fold: FoldModels)
    def clear_old_folds(self, current_grid_idx: int, keep_n: int = 12)
```

### Step 3: Replace DML Implementation with Grid-Based Version ✅

**File:** `src/models/dml_pytorch.py` (REPLACE entirely)

The new implementation replaces the old one completely.

Key functions:

1. **`build_dml_data(Y, W, p, day_idx, lookback)`** - Keep existing, unchanged

2. **`compute_fold_boundaries(grid_idx, config)`** - Returns (train_start, train_end, test_start, test_end)

3. **`ensure_fold_trained(cache, grid_idx, p, W, Y, T, config, model_factory)`**
   - Check cache, train if missing
   - Use `MultiOutputRegressor` with native parallelization

4. **`get_active_folds_for_day(day_idx, p, config)`**
   - Returns list of grid indices that provide residuals for this day

5. **`compute_residuals(cache, day_idx, p, outcome, treatment, controls, config, model_factory)`**
   - Loop through active folds
   - For each fold: predict on test rows, compute residuals
   - Return Y_residuals, T_residuals, last_fold

6. **`estimate_theta(Y_residuals, T_residuals)`** - Keep existing, unchanged

7. **`compute_se_oracle(theta, Y_residuals, T_residuals)`** - Keep existing, unchanged

8. **`fit_orvarx_single_day(Y, W, p, day_idx, cache, config, learner_name)`**
   - Uses cache internally

9. **`fit_orvarx(Y, W, p_max, config, ...)`**
   - Main entry point
   - Pre-computes all needed folds across test period
   - Returns `VARXResult`

10. **`get_all_required_folds(n_days, p_max, config)`** (NEW) - Returns all grid indices needed for test period

11. **`precompute_all_residuals(cache, p, Y, W, config, ...)`** (NEW) - Pre-compute residuals for all test windows

12. **`fit_orvarx_batched(Y, W, p_max, config, ...)`** (NEW) - Main vectorized entry point using batched OLS

### Step 4: Simplify Factory for CPU-Only Mode ✅

**File:** `src/modules/factory.py` (MODIFY)

Remove `use_gpu` option entirely - we use CPU for tree-based learners. TabPFN (GPU-only) would be a separate code path if needed later.

```python
def get_regressor(name: str = 'xgboost', n_jobs: int = -1, **kwargs):
    """CPU-only learners with native parallelization."""

    if name == 'xgboost':
        return xgb.XGBRegressor(
            n_jobs=n_jobs,
            tree_method='hist',
            **kwargs
        )

    elif name == 'lgbm':
        return lgb.LGBMRegressor(
            n_jobs=n_jobs,
            device='cpu',
            **kwargs
        )

    elif name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_jobs=n_jobs, **kwargs)

    elif name == 'extra_trees':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_jobs=n_jobs, **kwargs)
```

Add helper for MultiOutputRegressor wrapping:

```python
def get_multi_output_regressor(name: str, n_jobs: int = -1, **kwargs):
    """Wrap base regressor with MultiOutputRegressor for Y/T prediction.

    Parallelization strategy:
    - Base regressor uses n_jobs for internal tree parallelization
    - MultiOutputRegressor outer loop is sequential (n_jobs=1)
      since base already saturates CPU cores
    """
    from sklearn.multioutput import MultiOutputRegressor
    base = get_regressor(name, n_jobs=n_jobs, **kwargs)
    return MultiOutputRegressor(base, n_jobs=1)
```

**Note:** If you want MultiOutputRegressor to parallelize across outputs (and base to be single-threaded), use:
```python
base = get_regressor(name, n_jobs=1, **kwargs)
return MultiOutputRegressor(base, n_jobs=-1)
```
The first approach is usually better for tree-based methods since tree parallelization is more efficient.

### Step 4.5: Shared Batch Utilities ✅

**File:** `src/modules/batch_utils.py` (NEW)

Shared batched OLS utility used by both VAR and OR-VARX:

```python
def batched_ols(
    X_batch: torch.Tensor,  # (batch, n_samples, n_features)
    Y_batch: torch.Tensor,  # (batch, n_samples, n_targets)
    rcond: Optional[float] = None,
) -> torch.Tensor:
    """Batched OLS regression: beta = (X'X)^{-1} X'Y"""
    XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)
    XtY = torch.bmm(X_batch.transpose(1, 2), Y_batch)
    return torch.linalg.solve(XtX, XtY)
```

### Step 5: ORACLE-VARX (Deferred)

`src/models/oracle_var.py` - Will update later to use new DML functions.

---

## Critical File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/modules/grid_config.py` | CREATE | Grid configuration dataclass |
| `src/modules/model_cache.py` | CREATE | Model cache for fold reuse |
| `src/models/dml_pytorch.py` | REPLACE | Grid-based DML (replaces old implementation entirely) |
| `src/modules/factory.py` | MODIFY | CPU-only, add MultiOutputRegressor helper |
| `src/modules/batch_utils.py` | CREATE | Shared batched OLS for VAR and OR-VARX |
| `test_orvarx.py` | CREATE | Simple tests in root folder |
| `scripts/example_orvarx_usage.py` | MODIFY | Benchmark with n_days=1000, extratrees |
| `plans/orvarx-cached-plan.md` | CREATE | New plan doc with full discussion |
| `plans/orvarx-batched-plan.md` | DELETE | Remove old plan |

---

## Algorithm: Active Fold Selection

For `day_idx`, we need residuals for rows in the DML window `[day_idx - lookback + p : day_idx)` (exclusive end).

```python
def get_active_folds_for_day(day_idx: int, p: int, config: GridConfig) -> List[int]:
    """Get grid indices that provide residuals for this day."""

    # Row range in absolute indices
    row_start_abs = day_idx - config.lookback + p
    row_end_abs = day_idx - 1

    # First grid_idx that could cover any row in this range
    first_grid_idx = max(0, (row_start_abs - config.train_size) // config.test_size)

    # Last grid_idx needed (covers up to day_idx - 1)
    last_grid_idx = (row_end_abs - config.train_size) // config.test_size

    return list(range(first_grid_idx, last_grid_idx + 1))
```

---

## Verification Plan

1. **Simple Test File (root folder):**
   - `test_orvarx.py` - Single file with basic tests
   - `test_fold_boundaries()` - Verify correct train/test splits
   - `test_active_fold_selection()` - Verify correct folds selected for each day
   - `test_cache_reuse()` - Verify models are reused across days

2. **Performance Benchmark:**
   - `scripts/example_orvarx_usage.py` - Run with n_days=1000, extratrees
   - Simple end-to-end test to verify everything works

---

## VAR p_max Offset

Use `lookback=766` explicitly in the VARX script for both VAR and OR-VARX to ensure consistent comparison. The VAR code handles the p reduction internally, but using the same lookback value ensures both models use the same effective training window.

```python
# In scripts/example_orvarx_cached_usage.py
LOOKBACK = 766  # 756 base + 10 for p_max offset

# VAR
var_result = fit_var(Y, p_max=10, lookback=LOOKBACK)

# OR-VARX
orvarx_result = fit_orvarx_cached(Y, W, p_max=10, config=GridConfig(lookback_base=756, p_max_offset=10))
```

---

## Implementation Order

**Step 0 (First):** ✅ Create `plans/orvarx-cached-plan.md` with full discussion, delete `plans/orvarx-batched-plan.md`

**Step 1-4:** ✅ Implement in parallel using Sonnet subagents:
- Agent 1: ✅ `src/modules/grid_config.py` + `src/modules/model_cache.py`
- Agent 2: ✅ `src/modules/factory.py` modifications
- Agent 3: ✅ `src/models/dml_pytorch.py` replacement
- Agent 4: ✅ `src/modules/batch_utils.py` creation

**Step 5:** ✅ Create `test_orvarx.py` and `scripts/example_orvarx_usage.py`

**Step 6 (Final):** ✅ Run Opus subagent to verify code works correctly

---

## Design Decisions

1. **Memory:** Keep all folds in memory (no eviction)

2. **Parallelization:** Use native learner parallelization:
   - XGBoost/LightGBM: `n_jobs=-1` or `num_threads`
   - sklearn RF/ExtraTrees: `n_jobs=-1`
   - No outer joblib parallelization needed

3. **Staleness:** 21-day fold window is acceptable for the ~250x speedup
