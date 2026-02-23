# VRAM Considerations for Batched TabPFN

Date: 2026-02-23

## Batch Size Formula

```
batch = floor(11 * VRAM_GB / (n_c^0.75 * p^1.5))
```

Where:
- `VRAM_GB`: Total GPU memory in GB
- `n_c`: Number of confounders (1 for VIX, 5 for macro5, 10 for all10)
- `p`: Lag order (1 to p_max)

Batch is capped at n_folds (228) and floored at 1.

## Why This Formula?

### Sub-linear confounder scaling (n_c^0.75)

T-prediction probe data on A100 shows per-fold VRAM cost scales as n_c^0.75,
not linearly. Going from n_c=1 to n_c=5, per-fold cost increases by ~3.4×
(not 5×). The exponent 0.75 fits this: 5^0.75 = 3.34.

### Power-law lag scaling (p^1.5)

Per-fold cost grows super-linearly with p. The exponent 1.5 is a conservative
fit across the VIX probe data (actual log-log slope is ~1.4).

### Coefficient (11)

Chosen so the worst-case peak utilization across all (n_c, p) combinations
stays under 75% of total VRAM. The binding constraint is macro5 p=10 on A100
at ~72%.

### CUDA allocator

Requires `expandable_segments:True` (set automatically by the experiment
script). Without this, PyTorch's default block allocator fragments CUDA
memory across p values with different tensor shapes.

## Expected Batch Sizes by GPU

### 80 GB GPU (A100, H100)

| p | n_c=1 (VIX) | n_c=5 (macro5) | n_c=10 (all10) |
|--:|------------:|---------------:|---------------:|
| 1 | 228\* | 228\* | 156 |
| 2 | 228\* | 93 | 55 |
| 3 | 169 | 50 | 30 |
| 4 | 110 | 32 | 19 |
| 5 | 78 | 23 | 14 |
| 6 | 59 | 17 | 10 |
| 7 | 47 | 14 | 8 |
| 8 | 38 | 11 | 6 |
| 9 | 32 | 9 | 5 |
| 10 | 27 | 8 | 4 |

\* capped at n_folds=228

### 48 GB GPU (A6000, RTX 6000 Ada)

| p | n_c=1 (VIX) | n_c=5 (macro5) | n_c=10 (all10) |
|--:|------------:|---------------:|---------------:|
| 1 | 228\* | 157 | 93 |
| 2 | 186 | 55 | 33 |
| 3 | 101 | 30 | 18 |
| 4 | 66 | 19 | 11 |
| 5 | 47 | 14 | 8 |
| 6 | 35 | 10 | 6 |
| 7 | 28 | 8 | 5 |
| 8 | 23 | 6 | 4 |
| 9 | 19 | 5 | 3 |
| 10 | 16 | 4 | 2 |

\* capped at n_folds=228

### 24 GB GPU (RTX 4090)

| p | n_c=1 (VIX) | n_c=5 (macro5) | n_c=10 (all10) |
|--:|------------:|---------------:|---------------:|
| 1 | 228\* | 78 | 46 |
| 2 | 93 | 27 | 16 |
| 3 | 50 | 15 | 9 |
| 4 | 33 | 9 | 5 |
| 5 | 23 | 7 | 4 |
| 6 | 17 | 5 | 3 |
| 7 | 14 | 4 | 2 |
| 8 | 11 | 3 | 2 |
| 9 | 9 | 2 | 1 |
| 10 | 8 | 2 | 1 |

\* capped at n_folds=228

## Implementation

```python
def _get_batch_size_for_p(
    p: int, n_folds: int, n_confounders: int = 1,
    target_vram_pct: float = 0.65, verbose: bool = False,
) -> int:
    """Get batch size using empirical power-law VRAM model."""
    _, total_vram_gb, _ = _get_vram_usage()
    batch_size = int(11 * total_vram_gb / (n_confounders ** 0.75 * p ** 1.5))
    batch_size = max(1, min(batch_size, n_folds))
    return batch_size
```

## Probe Mode (`--probe`)

The `--probe` flag runs 1 iteration per p to empirically measure VRAM usage
before committing to a full run. The probe measures **T prediction** (the
binding VRAM constraint, with 9×p outputs per fold).

```bash
# Probe all presets
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders all --probe

# Probe a single preset
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders macro5 --probe
```

Probe output includes per-p VRAM usage:

```
PROBE COMPLETE
    p  features  status                  VRAM    time  suggested_batch
  ------------------------------------------------------------------
    1         5    PASS  12.8/80.0 GB (16%)  156.0s               96
   10        50    PASS  23.5/80.0 GB (29%)  900.0s                3
```

If any p value OOMs, the probe reports it and continues to the next p.

## Why Not Automatic VRAM Probing?

Automatic probing at runtime proved unreliable:
- `max_memory_allocated()` misses CUDA kernels, cuBLAS, and flash attention
- `mem_get_info()` before/after suffers from PyTorch caching artifacts
- Memory deltas can be negative due to reuse patterns

The empirical formula with optional `--probe` verification is more robust.

## Verification

Run with `--verbose` to see batch sizes:
```bash
python scripts/run_oraclevarx_tabpfn_experiment.py --n-days 1500 --verbose
```

Expected output on 80 GB GPU (VIX):
```
    p=1: n_c=1, 11*80/(1^0.75*1^1.5) -> batch=228 (n_folds=228)
    p=5: n_c=1, 11*80/(1^0.75*5^1.5) -> batch=78 (n_folds=228)
    p=10: n_c=1, 11*80/(1^0.75*10^1.5) -> batch=27 (n_folds=228)
```
