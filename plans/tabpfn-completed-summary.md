# TabPFN Batching Implementation Summary

**Status:** Completed

## Updates

- Fixed forecast computation to use exact `E[Y|W]` and `E[T|W]` predictions
- Forecast rows included in Phase 3 batch (no re-running TabPFN)
- Experiment script now uses `fit_oraclevarx_tabpfn()` directly
- Added adaptive batch size that scales with p (batch_size // p, min 5)
- Added VRAM monitoring with verbose output showing usage per p value

## Files

### `src/modules/batched_tabpfn.py`

Batched TabPFN inference using transformer's batch dimension.

**Class:** `BatchedFoldTabPFN`
- Stacks fold × output combinations into batch dimension
- Single forward pass for all folds with same feature count
- Configurable batch size for memory management
- `clear_cache()` to free GPU memory

**Helper functions:**
- `_stack_problems_for_batch()` - Prepare tensors for batched inference
- `_batched_forward()` - Run transformer forward pass
- `_decode_predictions()` - Convert logits to predictions
- `_disable_tabpfn_telemetry()` - Disable posthog for lower latency

### `src/models/oracle_var_tabpfn.py`

ORACLE-VARX implementation using BatchedFoldTabPFN.

**Function:** `fit_oraclevarx_tabpfn(Y, W, ...)`

**Helper functions:**
- `_get_vram_usage()` - Get GPU VRAM usage (used, total, percent)
- `_compute_adaptive_batch_size()` - Scale batch size inversely with p

### `src/modules/factory.py`

Regressor factory with options: `xgboost`, `lgbm`, `rf`, `extra_trees`, `tabpfn`

### `scripts/example_benchmark_batched_tabpfn.py`

Benchmark comparing sequential vs batched inference.

### `scripts/sequential_vs_batchedfold_tabpfn_analysis.md`

Detailed performance analysis.

## Performance

| Configuration | Sequential | BatchedFoldTabPFN | Speedup |
|---------------|------------|-------------------|---------|
| 1 fold, 9 outputs | 13.42s | 0.94s | 14.3x |
| 23 folds, 9 outputs | 308.73s | 3.75s | 82.4x |

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU

## Usage

```python
from src.modules.batched_tabpfn import BatchedFoldTabPFN

tabpfn = BatchedFoldTabPFN(n_estimators=8, device='cuda')
predictions = tabpfn.fit_predict_batch(X_trains, Y_trains, X_tests)
# Returns: (n_folds, n_test, n_outputs)
tabpfn.clear_cache()
```

## Verification

```bash
python -m py_compile src/modules/batched_tabpfn.py
python -c "from src.modules.batched_tabpfn import BatchedFoldTabPFN; print('OK')"
python scripts/example_benchmark_batched_tabpfn.py
```
