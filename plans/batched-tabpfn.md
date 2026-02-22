# Batched TabPFN for ORACLE-VARX

## Overview

`BatchedFoldTabPFN` provides massively parallel TabPFN inference by exploiting the transformer's batch dimension. Instead of sequential fold-by-fold processing, it stacks all fold-output combinations into a single forward pass.

## Performance

| Metric | Sequential | BatchedFoldTabPFN |
|--------|------------|-------------------|
| Single fold (9 outputs) | 13.42s | 0.94s |
| 23 folds | 308.73s | 3.75s |
| **Speedup** | - | **82.4x** |

Tested on NVIDIA GeForce RTX 4060 Laptop GPU.

## How It Works

TabPFN's transformer accepts `(seq_len, batch_size, n_features)` inputs. We stack multiple fold-output problems into the batch dimension:

```
For 23 folds × 9 outputs:
    batch_size = 23 × 9 = 207 items

Sequential: 23 × 9 = 207 separate forward passes
Batched:    1 forward pass with batch_size=207
```

## Architecture

```
src/modules/batched_tabpfn.py
├── Helper functions
│   ├── _stack_problems_for_batch()   # Stack fold-outputs into batch format
│   ├── _batched_forward()            # Single forward pass through transformer
│   ├── _decode_predictions()         # Decode logits and reshape
│   └── _disable_tabpfn_telemetry()   # Reduce latency by disabling telemetry
│
└── BatchedFoldTabPFN                 # Main class for batched inference
```

## Usage

```python
from src.modules.batched_tabpfn import BatchedFoldTabPFN

# Prepare fold data
X_trains = [X_train_fold1, X_train_fold2, ...]  # List of (n_train, n_features)
Y_trains = [Y_train_fold1, Y_train_fold2, ...]  # List of (n_train, n_outputs)
X_tests = [X_test_fold1, X_test_fold2, ...]     # List of (n_test, n_features)

# Batched inference
tabpfn = BatchedFoldTabPFN(n_estimators=8, device='cuda')
predictions = tabpfn.fit_predict_batch(X_trains, Y_trains, X_tests, batch_size=50)
# Returns: (n_folds, n_test, n_outputs)

# Free GPU memory
tabpfn.clear_cache()
```

## Memory Management

BatchedFoldTabPFN trades memory for speed. Use `batch_size` to control memory usage:

```python
# Process 25 folds at a time instead of all at once
predictions = tabpfn.fit_predict_batch(X_trains, Y_trains, X_tests, batch_size=25)
```

Typical memory usage:
- Sequential: ~500MB GPU
- Batched (207 items): ~1-2GB GPU

## Adaptive Batch Size

For higher lag orders (p), the number of features increases, consuming more VRAM. Batch size scales with a size-based heuristic that also accounts for the number of confounders:

```python
confounder_scale = 10.0 / (9 + n_confounders)
batch_size = int(confounder_scale * 6 * VRAM_GB / (p ** 1.5))
```

See `plans/vram-considerations-tabpfn.md` for detailed tables and tuning guidance.

## VRAM Monitoring

With `verbose=True`, the model logs VRAM usage before and after each p value:

```
  Phase 3: Running batched TabPFN predictions...
    p=1: 23 folds, batch_size=50, VRAM: 2.1/8.0 GB (26%)
    p=1: completed in 4.2s, VRAM: 5.8/8.0 GB (73%), ...
    p=2: 23 folds, batch_size=25, VRAM: 2.3/8.0 GB (29%)
    p=2: completed in 5.1s, VRAM: 5.9/8.0 GB (74%), ...
```

Helper functions in `src/models/oracle_var_tabpfn.py`:
- `_get_vram_usage()` - Returns (used_gb, total_gb, percent_used)
- `_get_batch_size_for_p(p, n_folds, n_confounders)` - Scales batch size for VRAM control

## Integration with ORACLE-VARX

```python
from src.models.oracle_var_tabpfn import fit_oraclevarx_tabpfn

result = fit_oraclevarx_tabpfn(
    Y=Y_tensor,
    W=W_tensor,
    p_max=10,
    n_estimators=8,
    batch_size=50,
    device='cuda',
)
```

## Benchmark

Run the benchmark script to verify performance on your hardware:

```bash
python scripts/example_benchmark_batched_tabpfn.py
```

See `scripts/sequential_vs_batchedfold_tabpfn_analysis.md` for detailed analysis.

## Integration Status

The experiment script `scripts/run_oraclevarx_tabpfn_experiment.py` now uses
`fit_oraclevarx_tabpfn()` for full batched TabPFN inference with exact
forecasting (no residual proxy approximation).

### Exact DML Forecasting

The forecast computation uses the exact DML formula:
```
forecast = E[Y|W] + (T_actual - E[T|W]) × θ
```

Where:
- `E[Y|W]` and `E[T|W]` are computed by TabPFN in Phase 3 (included in the same batch)
- `T_actual` are the actual lagged returns
- `θ` are the deconfounded coefficients from second-stage OLS

This eliminates the residual proxy approximation previously used in Phase 5.

## Probe Mode (`--probe`)

When using more confounders (macro5: 5, all10: 10), the feature count per p grows
as `(9 + n_confounders) * p`, which can cause OOM on GPUs with limited VRAM.

The `--probe` flag runs 1 iteration per p with the real heuristic batch size to
empirically test VRAM usage before committing to a full run. No results are saved.
Probe shares the exact same grouped code path as non-probe (including
`folds_by_test_size` grouping and `fit_predict_batch` sub-batching).

### Batch Size Scaling for Confounders

`_get_batch_size_for_p()` accounts for confounder count with:

```
batch_size = int(10 / (9 + n_confounders) * 6 * VRAM_GB / p^1.5)
```

Scaling factors: VIX (1 confounder) = 1.0x, macro5 (5) = 0.71x, all10 (10) = 0.53x.

### Usage

```bash
# Probe macro5
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders macro5 --probe

# Probe all10 with less data for faster testing
python scripts/run_oraclevarx_tabpfn_experiment.py --confounders all10 --probe --n-days 1500
```

### Expected Output

```
*** PROBE MODE: Testing 1 iteration per p (p=1..10) ***
*** n_confounders=5, total features per p: (9+5)*p ***

  p=1: PROBE mode, testing 1 iteration with batch_size=342
    PROBE p=1: PASS (batch_size=342)
      Controls features: 5 (5 confounders x 1 lags)
      VRAM after inference: 12.3/80.0 GB (15.4%)
      Time: 2.3s
  ...

PROBE COMPLETE
    p  features  status                  VRAM    time  suggested_batch
  ------------------------------------------------------------------
    1         5    PASS  12.3/80.0 GB (15%)    2.3s               96
   10        50    PASS  28.9/80.0 GB (36%)    8.7s                3
```
