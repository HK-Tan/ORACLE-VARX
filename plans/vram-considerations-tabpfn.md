# VRAM Considerations for Batched TabPFN

## Batch Size Heuristic

After extensive testing, we use an **empirical linear VRAM model** rather than runtime VRAM probing. The probing approach proved unreliable due to PyTorch memory caching and the difficulty of capturing TabPFN's full memory footprint (CUDA kernels, cuBLAS workspaces, attention buffers).

**Formula:**
```
batch_size = floor(target_vram_pct * VRAM_GB / per_fold_gb)
per_fold_gb = 0.0171 * f + 0.1292
f = n_confounders * p
```

Where:
- `VRAM_GB`: Total GPU memory in GB
- `f`: Total features = n_confounders x p
- `per_fold_gb`: Predicted VRAM cost per fold (empirically fitted from 37 probe data points)
- `target_vram_pct`: Fraction of total VRAM to use (default: 0.65)

## Why This Formula?

### Linear Feature Scaling
Per-fold VRAM cost scales linearly with `f = n_confounders * p`. Coefficients (0.0171, 0.1292) fitted from 37 probe data points across A6000/A100 GPUs with n_confounders in {1, 5, 10}. The fit is a conservative upper bound that never underestimates actual cost.

### Unified Confounder Handling
Unlike the old formula which had a separate `10/(9+n_confounders)` scaling factor, confounders are now naturally incorporated via `f = n_confounders * p`. No separate scaling needed.

### Default Target: 65% VRAM
`target_vram_pct=0.65` targets ~60% actual torch utilization, leaving headroom for CUDA workspace overhead that PyTorch doesn't track.

## Expected Batch Sizes by GPU (VIX, n_confounders=1)

### 8GB GPU (e.g., RTX 3070)
Budget = 0.65 x 8 = 5.2 GB

| p | f | per_fold_gb | Batch Size |
|---|---|-------------|------------|
| 1 | 1 | 0.1463 | 35 |
| 2 | 2 | 0.1634 | 31 |
| 3 | 3 | 0.1805 | 28 |
| 5 | 5 | 0.2147 | 24 |
| 10 | 10 | 0.3002 | 17 |

### 24GB GPU (e.g., RTX 4090)
Budget = 0.65 x 24 = 15.6 GB

| p | f | per_fold_gb | Batch Size |
|---|---|-------------|------------|
| 1 | 1 | 0.1463 | 106 |
| 2 | 2 | 0.1634 | 95 |
| 3 | 3 | 0.1805 | 86 |
| 5 | 5 | 0.2147 | 72 |
| 10 | 10 | 0.3002 | 51 |

### 48GB GPU (e.g., A40, RTX 6000 Ada)
Budget = 0.65 x 48 = 31.2 GB

| p | f | per_fold_gb | Batch Size |
|---|---|-------------|------------|
| 1 | 1 | 0.1463 | 213 |
| 2 | 2 | 0.1634 | 190 |
| 3 | 3 | 0.1805 | 172 |
| 5 | 5 | 0.2147 | 145 |
| 10 | 10 | 0.3002 | 103 |

### 80GB GPU (e.g., A100 80GB, H100)
Budget = 0.65 x 80 = 52.0 GB

| p | f | per_fold_gb | Batch Size |
|---|---|-------------|------------|
| 1 | 1 | 0.1463 | 355 |
| 2 | 2 | 0.1634 | 318 |
| 3 | 3 | 0.1805 | 288 |
| 5 | 5 | 0.2147 | 242 |
| 10 | 10 | 0.3002 | 173 |

### Effect of Confounders on 80GB GPU

| p | n_c=1 (f=p) | n_c=5 (f=5p) | n_c=10 (f=10p) |
|---|-------------|--------------|----------------|
| 1 | 355 | 242 | 173 |
| 2 | 318 | 173 | 110 |
| 5 | 242 | 93 | 52 |
| 10 | 173 | 52 | 28 |

## Implementation

```python
def _get_batch_size_for_p(
    p: int, n_folds: int, n_confounders: int = 1,
    target_vram_pct: float = 0.65, verbose: bool = False,
) -> int:
    """Get batch size using empirical linear VRAM model."""
    _, total_vram_gb, _ = _get_vram_usage()
    f = n_confounders * p
    per_fold_gb = 0.0171 * f + 0.1292
    vram_budget = target_vram_pct * total_vram_gb
    batch_size = int(vram_budget / per_fold_gb)
    batch_size = max(1, min(batch_size, n_folds))
    return batch_size
```

## Why Not Automatic VRAM Probing?

We tried several *automatic* probing approaches (measuring VRAM delta to compute
batch size at runtime) that all failed. For *manual* probing before a full run,
use the `--probe` flag (see below).

### 1. `max_memory_allocated()` Approach
Only tracks PyTorch tensor allocations, missing:
- CUDA kernel memory
- cuBLAS workspaces
- Flash attention buffers
- Other non-PyTorch allocations

Result: Reported ~0.1 GB when actual usage was ~20 GB.

### 2. `mem_get_info()` Before/After Approach
Measures driver-level memory but suffers from:
- PyTorch memory caching causing inconsistent baselines
- Memory reuse patterns between batch=1 and batch=2
- Could show batch=2 using LESS memory than batch=1

Result: Negative per_batch costs, triggering fallback.

### 3. Memory Increase Measurement with Warmup
Better in theory, but still inconsistent due to:
- `empty_cache()` not fully resetting state
- Model weight loading affecting first measurement
- Fragmentation effects

## Verification

Run with `--verbose` to see batch sizes:
```bash
python scripts/run_oraclevarx_tabpfn_experiment.py --n-days 1500 --verbose
```

Expected output on 80GB GPU (VIX):
```
    p=1: f=1, per_fold=0.1463 GB, budget=52.0 GB (65% of 80 GB) -> batch=355 (n_folds=228)
    p=5: f=5, per_fold=0.2147 GB, budget=52.0 GB (65% of 80 GB) -> batch=242 (n_folds=228)
    p=10: f=10, per_fold=0.3002 GB, budget=52.0 GB (65% of 80 GB) -> batch=173 (n_folds=228)
```

## Tuning

Adjust `target_vram_pct` to control VRAM usage:
```python
# More conservative (less VRAM usage)
result = fit_oraclevarx_tabpfn(..., target_vram_pct=0.50)

# More aggressive (faster, uses more VRAM)
result = fit_oraclevarx_tabpfn(..., target_vram_pct=0.80)
```

## Probe Mode (`--probe`)

When using more confounders, the heuristic formula may not be accurate enough for
your GPU. The `--probe` flag runs 1 iteration per p with the real heuristic batch
size to empirically test VRAM usage before committing to a full run. No results
are saved. Probe shares the exact same grouped code path as non-probe (including
`folds_by_test_size` grouping and `fit_predict_batch` sub-batching).

```bash
# Probe macro5 on your GPU
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders macro5 --probe

# Probe all10 with less data for faster testing
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders all10 --probe --n-days 1500
```

Probe output includes per-p VRAM usage and suggested batch sizes:

```
PROBE COMPLETE
    p  features  status                  VRAM    time  suggested_batch
  ------------------------------------------------------------------
    1         5    PASS  12.3/80.0 GB (15%)    2.3s               96
   10        50    PASS  28.9/80.0 GB (36%)    8.7s                3
```

If any p value OOMs at the heuristic batch size, the probe reports it and
continues to the next p.
