# VRAM Considerations for Batched TabPFN

## Batch Size Heuristic

After extensive testing, we use a **simple size-based heuristic** rather than runtime VRAM probing. The probing approach proved unreliable due to PyTorch memory caching and the difficulty of capturing TabPFN's full memory footprint (CUDA kernels, cuBLAS workspaces, attention buffers).

**Formula:**
```
batch_size = (10 / (9 + n_confounders)) × (6 × VRAM_GB) / p^1.5
```

Where:
- `VRAM_GB`: Total GPU memory in GB
- `p`: Lag order (1 to p_max)
- `p^1.5`: Super-linear scaling to account for attention memory growth
- `n_confounders`: Number of confounders (1 for VIX, 5 for macro5, 10 for all10)
- `10 / (9 + n_confounders)`: Confounder scaling factor (9 assets always present)

### Confounder Scaling

The `(9 + n_confounders)` term captures total variable load: 9 assets are always
present in the batch dimension, confounders add features on top. The scaling factor
`10 / (9 + n_confounders)` ensures backward compatibility with VIX (1 confounder):

| Preset | n_confounders | Scale Factor | Effect |
|--------|---------------|-------------|--------|
| vix | 1 | 10/10 = 1.00x | Exact backward compat |
| macro5 | 5 | 10/14 = 0.71x | ~30% smaller batches |
| all10 | 10 | 10/19 = 0.53x | ~47% smaller batches |

## Why This Formula?

### Linear VRAM Scaling
Calibrated so that 80GB GPU gets numerator of 480:
- `6 × 80 = 480`

The coefficient of 6 was empirically tuned on A100 80GB with ~228 folds.

### Super-linear p Scaling
Higher p means more treatment features (n_assets × p), which increases memory super-linearly due to:
1. **Attention matrices**: O(features²) in the attention mechanism
2. **Intermediate activations**: Scale with feature count
3. **cuBLAS workspaces**: Grow with matrix dimensions

The `p^1.5` exponent provides extra safety margin for larger p values.

## Expected Batch Sizes by GPU (VIX, n_confounders=1)

### 8GB GPU (e.g., RTX 3070)
Numerator = 6 × 8 = 48

| p | Batch Size |
|---|------------|
| 1 | 48 |
| 2 | 16 |
| 3 | 9 |
| 5 | 4 |
| 10 | 1 |

### 24GB GPU (e.g., RTX 4090)
Numerator = 6 × 24 = 144

| p | Batch Size |
|---|------------|
| 1 | 144 |
| 2 | 50 |
| 3 | 27 |
| 5 | 12 |
| 10 | 4 |

### 48GB GPU (e.g., A40, RTX 6000 Ada)
Numerator = 6 × 48 = 288

| p | Batch Size |
|---|------------|
| 1 | 288 |
| 2 | 101 |
| 3 | 55 |
| 5 | 25 |
| 10 | 9 |

### 80GB GPU (e.g., A100 80GB, H100)
Numerator = 6 × 80 = 480

| p | Batch Size |
|---|------------|
| 1 | 480 |
| 2 | 169 |
| 3 | 92 |
| 5 | 42 |
| 10 | 15 |

### Effect of Confounders on 80GB GPU

| p | VIX (1.0x) | macro5 (0.71x) | all10 (0.53x) |
|---|-----------|----------------|---------------|
| 1 | 480 | 342 | 252 |
| 2 | 169 | 120 | 89 |
| 5 | 42 | 30 | 22 |
| 10 | 15 | 10 | 7 |

## Implementation

```python
def _get_batch_size_for_p(p: int, n_folds: int, n_confounders: int = 1, verbose: bool = False) -> int:
    """Get batch size for a given lag order p using size-based heuristic."""
    _, total_vram_gb, _ = _get_vram_usage()

    # Scale factor: 6 * VRAM (calibrated: 480 for 80GB)
    numerator = 6 * total_vram_gb

    # Scale down for more confounders: (9 + n_confounders) captures total variable load
    confounder_scale = 10.0 / (9 + n_confounders)
    batch_size = int(confounder_scale * numerator / (p ** 1.5))
    batch_size = min(batch_size, n_folds)
    batch_size = max(1, batch_size)

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
  Phase 3: Running batched TabPFN predictions...
    p=1: batch_size = 1.00x * 480/1^1.5 = 480 → capped to 228 (n_confounders=1)
    p=2: batch_size = 1.00x * 480/2^1.5 = 169 → capped to 169 (n_confounders=1)
    p=3: batch_size = 1.00x * 480/3^1.5 = 92 → capped to 92 (n_confounders=1)
    ...
    p=10: batch_size = 1.00x * 480/10^1.5 = 15 → capped to 15 (n_confounders=1)
```

## Tuning

If you experience OOM errors, reduce the coefficient:
```python
numerator = 4.5 * total_vram_gb  # More conservative
```

If you want to be more aggressive:
```python
numerator = 7.5 * total_vram_gb  # Use more VRAM
```

## Probe Mode (`--probe`)

When using more confounders, the heuristic formula may not be accurate enough for
your GPU. The `--probe` flag runs 1 fold per p value with `batch_size=1` to
empirically test VRAM usage before committing to a full run. No results are saved.

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

If any p value OOMs even with `batch_size=1`, the probe reports it and continues
to the next p.
