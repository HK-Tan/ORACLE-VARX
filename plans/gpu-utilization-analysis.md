# GPU Utilization Analysis: TabPFN Batch Size Formula

## Probe Data Summary

Four probe runs across 2 GPUs and 3 confounder configs:

| Run | GPU | VRAM | n_confounders | n_features = n_c * p |
|-----|-----|------|---------------|---------------------|
| VIX/A6000 | RTX A6000 | 48 GB | 1 | 1–10 |
| VIX/A100 | A100 80GB | 80 GB | 1 | 1–10 |
| macro5/A100 | A100 80GB | 80 GB | 5 | 5–50 |
| all10/A6000 | RTX A6000 | 48 GB | 10 | 10–70 |

## Per-Fold VRAM Cost (torch peak reserved / batch_size)

| f (features) | VIX/A6000 | VIX/A100 | macro5/A100 | all10/A6000 |
|-------------|-----------|----------|-------------|-------------|
| 1 | 0.093 | 0.093 | — | — |
| 2 | 0.093 | 0.093 | — | — |
| 3 | 0.093 | 0.094 | — | — |
| 4 | 0.180 | 0.178 | — | — |
| 5 | 0.180 | 0.155 | 0.154 | — |
| 6 | 0.158 | 0.178 | — | — |
| 7 | 0.233 | 0.232 | — | — |
| 8 | 0.233 | 0.233 | — | — |
| 9 | 0.240 | 0.229 | — | — |
| 10 | 0.300 | 0.300 | 0.298 | 0.299 |
| 15 | — | — | 0.343 | — |
| 20 | — | — | 0.405 | 0.406 |
| 25 | — | — | 0.513 | — |
| 30 | — | — | 0.570 | 0.572 |
| 35 | — | — | 0.678 | — |
| 40 | — | — | 0.787 | 0.789 |
| 45 | — | — | 0.833 | — |
| 50 | — | — | 0.940 | 0.954 |
| 60 | — | — | — | 1.120 |
| 70 | — | — | — | 1.325 |

**Key observation**: Per-fold cost is consistent across GPUs for the same feature count. This means per-fold cost is a GPU-independent quantity — only `f = n_confounders * p` matters.

## Fitted Model

Power-law fit yields exponent ~1.03, confirming **linear scaling**:

```
per_fold_gb = 0.0171 * f + 0.1292
```

where `f = n_confounders * p`.

This is a **conservative upper bound** that covers 100% of all 37 data points (never underestimates). The two binding constraints are at f=10 (per_fold=0.300) and f=70 (per_fold=1.325).

### Staircase at low f

For f=1,2,3 the actual cost is ~0.093 GB/fold but the formula predicts ~0.146. This ~57% overestimate at low f is acceptable because:
1. Low-f cases have abundant batch capacity anyway (often capped to n_folds)
2. Being conservative at low f prevents OOM from CUDA workspace ratcheting

## Proposed Formula

```python
def _get_batch_size_for_p(p, n_folds, n_confounders=1, target_vram_pct=0.65, verbose=False):
    _, total_vram_gb, _ = _get_vram_usage()

    # Per-fold VRAM cost: linear in total features (empirically fitted)
    f = n_confounders * p
    per_fold_gb = 0.0171 * f + 0.1292

    # VRAM budget: target fraction of total VRAM
    vram_budget = target_vram_pct * total_vram_gb

    batch_size = int(vram_budget / per_fold_gb)
    batch_size = max(1, min(batch_size, n_folds))
    return batch_size
```

## Predicted Utilization

### VIX (1 confounder) at 65% target

| p | f | per_fold | A6000 batch | A6000 util% | A100 batch | A100 util% |
|---|---|----------|-------------|-------------|------------|------------|
| 1 | 1 | 0.146 | 213 | 65% | 228 (cap) | 42% |
| 2 | 2 | 0.163 | 190 | 65% | 228 (cap) | 47% |
| 3 | 3 | 0.181 | 172 | 65% | 228 (cap) | 51% |
| 5 | 5 | 0.215 | 145 | 65% | 228 (cap) | 61% |
| 7 | 7 | 0.249 | 125 | 65% | 208 | 65% |
| 10 | 10 | 0.300 | 103 | 65% | 173 | 65% |

Note: VIX with 228 folds hits the cap for low p on A100. Utilization undershoots vs target when capped.

### macro5 (5 confounders) at 65% target

| p | f | per_fold | A100 batch | A100 util% |
|---|---|----------|------------|------------|
| 1 | 5 | 0.215 | 228 (cap) | 61% |
| 3 | 15 | 0.386 | 134 | 65% |
| 5 | 25 | 0.557 | 93 | 65% |
| 7 | 35 | 0.728 | 71 | 65% |
| 10 | 50 | 0.984 | 52 | 64% |

### all10 (10 confounders) at 65% target

| p | f | per_fold | A6000 batch | A6000 util% | A100 batch | A100 util% |
|---|---|----------|-------------|-------------|------------|------------|
| 1 | 10 | 0.300 | 103 | 65% | 173 | 65% |
| 3 | 30 | 0.642 | 48 | 64% | 80 | 64% |
| 5 | 50 | 0.984 | 31 | 64% | 52 | 64% |
| 7 | 70 | 1.326 | 23 | 64% | 39 | 65% |
| 10 | 100 | 1.839 | 16 | 61% | 28 | 64% |

## CUDA Workspace Consideration

Driver VRAM (nvidia-smi) includes a CUDA library workspace that ratchets up:
- A6000: ~22 GB (set by first large forward pass)
- A100: ~22 GB for VIX, ~54 GB for macro5

This workspace is NOT tracked by PyTorch and cannot be freed without process restart. Our formula uses **torch peak reserved** (PyTorch-tracked) as the cost metric, and targets 65% of **total VRAM**. Since the workspace is roughly proportional to the PyTorch allocation peak, this provides an implicit safety margin.

## Old vs New Formula Comparison

| Aspect | Old formula | New formula |
|--------|------------|-------------|
| Core | `480 / p^1.5` | `0.65 * VRAM / (0.0171*f + 0.1292)` |
| Confounder scaling | Linear: `10/(9+n_c)` | Built into `f = n_c * p` |
| GPU scaling | `6 * VRAM_GB` (linear) | `0.65 * VRAM_GB` (linear) |
| p scaling | `p^-1.5` (too aggressive) | `~p^-1` (matches data) |
| all10 p=7 A6000 | 8 folds → 96% VRAM, crawling | 23 folds → 64% VRAM, fast |
| VIX p=10 A100 | 15 folds → 6% util | 173 folds → 65% util |
