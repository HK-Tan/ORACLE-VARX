# VRAM Considerations for Batched TabPFN

## Key Finding: Linear VRAM Scaling

VRAM usage scales **perfectly linearly** with batch size (R²=1.000 verified empirically). This enables a simple 2-point probe to determine optimal batch size for any GPU.

**Linear Model:**
```
VRAM = fixed_overhead + per_batch_cost × batch_size
```

Where:
- `fixed_overhead`: Base VRAM for TabPFN model and framework (~0.05 GB)
- `per_batch_cost`: Incremental VRAM per fold in batch (varies with p)

## VRAM Scaling by Lag Order (p)

Higher p means more treatment features (n_assets × p), which increases per-batch cost:

| p | Features (9 assets) | per_batch_cost (GB) | Notes |
|---|---------------------|---------------------|-------|
| 1 | 9 treatments | ~0.16 | Smallest batches possible |
| 3 | 27 treatments | ~0.41 | |
| 5 | 45 treatments | ~0.65 | |
| 7 | 63 treatments | ~0.90 | |
| 10 | 90 treatments | ~1.27 | Largest feature set |

*Values measured on 8GB GPU with n_estimators=8*

## Automatic Batch Size Selection

The implementation uses a **2-point VRAM probe** for each p value:

1. Run batch=1, measure peak VRAM → `vram_1`
2. Run batch=2, measure peak VRAM → `vram_2`
3. Compute linear model parameters:
   - `per_batch_cost = vram_2 - vram_1`
   - `fixed_overhead = vram_1 - per_batch_cost`
4. Extrapolate optimal batch size:
   - `optimal_batch = (target_vram - fixed_overhead) / per_batch_cost`

**Default target: 75% of total VRAM** (leaves headroom for system stability)

## Expected Batch Sizes by GPU

### 8GB GPU (e.g., RTX 3070)
Target = 0.75 × 8 = 6 GB

| p | Optimal batch |
|---|---------------|
| 1 | 36 |
| 3 | 14 |
| 5 | 9 |
| 7 | 6 |
| 10 | 4 |

### 24GB GPU (e.g., RTX 4090, A100 40GB)
Target = 0.75 × 24 = 18 GB

| p | Optimal batch |
|---|---------------|
| 1 | 110 |
| 3 | 43 |
| 5 | 27 |
| 7 | 20 |
| 10 | 14 |

### 80GB GPU (e.g., A100 80GB, H100)
Target = 0.75 × 80 = 60 GB

| p | Optimal batch |
|---|---------------|
| 1 | 370 |
| 3 | 145 |
| 5 | 92 |
| 7 | 66 |
| 10 | 47 |

## Why Linear Scaling?

TabPFN processes batches as `(seq_len, batch_size, features)` tensors. The dominant memory consumers are:

1. **Attention matrices**: O(seq_len² × batch_size) - linear in batch
2. **Intermediate activations**: O(seq_len × batch_size × hidden_dim) - linear in batch
3. **Model weights**: O(1) - constant (shared across batch)

Since model weights are constant and all other terms scale linearly with batch_size, total VRAM scales linearly.

## Implementation Details

### Probe Function
```python
def _probe_vram_for_batch_size(
    tabpfn: BatchedFoldTabPFN,
    X_trains: List[np.ndarray],
    Y_trains: List[np.ndarray],
    X_tests: List[np.ndarray],
    target_vram_pct: float = 0.75,
) -> int:
```

### Measurement Method
- Uses `torch.cuda.max_memory_allocated()` for accurate peak measurement
- Calls `torch.cuda.empty_cache()` and `reset_peak_memory_stats()` before each probe
- Runs actual TabPFN inference (not estimates) for accurate measurement

### Edge Cases Handled
- `len(X_trains) < 2`: Returns 1 (can't probe)
- `per_batch_cost <= 0`: Returns all folds (no scaling detected)
- No CUDA: Returns all folds (CPU has different constraints)
- Result capped at available folds and minimum of 1

## Tuning the Target VRAM

The `target_vram_pct` parameter (default 0.75) can be adjusted:

| Setting | Use Case |
|---------|----------|
| 0.60 | Conservative; running other GPU processes |
| 0.75 | Default; good balance of speed and stability |
| 0.85 | Aggressive; maximizes batch size |
| 0.90+ | Not recommended; risk of OOM |

## Comparison to Previous Heuristic

**Old approach** (heuristic):
```python
base = 32 * (total_gb / 8)  # Linear scaling from 8GB baseline
batch_size = base // p      # Inverse scaling with p
```

Problems:
- Assumed fixed relationship between GPU size and optimal batch
- Didn't account for actual VRAM usage patterns
- Could under- or over-estimate significantly

**New approach** (2-point probe):
- Measures actual VRAM on the specific hardware
- Accounts for TabPFN version, CUDA version, driver differences
- Adapts to actual data shapes (train_size, test_size, n_features)
- Self-calibrating for any GPU size

## Verification

Run with `--verbose` to see probed batch sizes:
```bash
python scripts/run_oraclevarx_tabpfn_experiment.py --n-days 1500 --verbose
```

Expected output:
```
  Phase 3: Running batched TabPFN predictions...
    p=1: 23 folds, probed batch_size=23...
    p=2: 23 folds, probed batch_size=18...
    ...
    p=7: 23 folds, probed batch_size=6...
    p=7: completed in 45.2s, VRAM: 5.8/8.0 GB (72%)
    ...
```

VRAM should stay around the target (75%) across all p values.
