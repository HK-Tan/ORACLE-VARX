# VRAM Considerations for Batched TabPFN

## Batch Size Heuristic

After extensive testing, we use a **simple size-based heuristic** rather than runtime VRAM probing. The probing approach proved unreliable due to PyTorch memory caching and the difficulty of capturing TabPFN's full memory footprint (CUDA kernels, cuBLAS workspaces, attention buffers).

**Formula:**
```
batch_size = (4.5 × VRAM_GB) / p^1.5
```

Where:
- `VRAM_GB`: Total GPU memory in GB
- `p`: Lag order (1 to p_max)
- `p^1.5`: Super-linear scaling to account for attention memory growth

## Why This Formula?

### Linear VRAM Scaling
Calibrated so that 80GB GPU gets numerator of 360:
- `4.5 × 80 = 360`

The 4.5 coefficient was empirically tuned on A100 80GB with ~228 folds.

### Super-linear p Scaling
Higher p means more treatment features (n_assets × p), which increases memory super-linearly due to:
1. **Attention matrices**: O(features²) in the attention mechanism
2. **Intermediate activations**: Scale with feature count
3. **cuBLAS workspaces**: Grow with matrix dimensions

The `p^1.5` exponent provides extra safety margin for larger p values.

## Expected Batch Sizes by GPU

### 8GB GPU (e.g., RTX 3070)
Numerator = 4.5 × 8 = 36

| p | Batch Size |
|---|------------|
| 1 | 36 |
| 2 | 12 |
| 3 | 6 |
| 5 | 3 |
| 10 | 1 |

### 24GB GPU (e.g., RTX 4090)
Numerator = 4.5 × 24 = 108

| p | Batch Size |
|---|------------|
| 1 | 108 |
| 2 | 38 |
| 3 | 20 |
| 5 | 9 |
| 10 | 3 |

### 48GB GPU (e.g., A40, RTX 6000 Ada)
Numerator = 4.5 × 48 = 216

| p | Batch Size |
|---|------------|
| 1 | 216 |
| 2 | 76 |
| 3 | 41 |
| 5 | 19 |
| 10 | 6 |

### 80GB GPU (e.g., A100 80GB, H100)
Numerator = 4.5 × 80 = 360

| p | Batch Size |
|---|------------|
| 1 | 360 |
| 2 | 127 |
| 3 | 69 |
| 5 | 32 |
| 10 | 11 |

## Implementation

```python
def _get_batch_size_for_p(p: int, n_folds: int, verbose: bool = False) -> int:
    """Get batch size for a given lag order p using size-based heuristic."""
    _, total_vram_gb, _ = _get_vram_usage()

    # Scale factor: 4.5 * VRAM (calibrated: 360 for 80GB)
    numerator = 4.5 * total_vram_gb

    batch_size = int(numerator / (p ** 1.5))
    batch_size = min(batch_size, n_folds)
    batch_size = max(1, batch_size)

    return batch_size
```

## Why Not VRAM Probing?

We tried several probing approaches that all failed:

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

Expected output on 80GB GPU:
```
  Phase 3: Running batched TabPFN predictions...
    p=1: batch_size = 360/1^1.5 = 360 → capped to 228
    p=2: batch_size = 360/2^1.5 = 127 → capped to 127
    p=3: batch_size = 360/3^1.5 = 69 → capped to 69
    ...
    p=10: batch_size = 360/10^1.5 = 11 → capped to 11
```

## Tuning

If you experience OOM errors, reduce the coefficient:
```python
numerator = 3.5 * total_vram_gb  # More conservative
```

If you want to be more aggressive:
```python
numerator = 5.5 * total_vram_gb  # Use more VRAM
```
