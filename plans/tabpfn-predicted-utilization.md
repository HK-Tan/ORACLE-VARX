# TabPFN Predicted Utilization (A6000 / A100)

Date: 2026-02-22

This note records predicted utilization from the new batch size formula.

## Assumptions

- **Per-fold memory model**: `per_fold_gb = 0.0171 * f + 0.1292` where `f = n_confounders * p`
- **Target VRAM percentage**: `target_vram_pct = 0.65`
- **Batch size formula**:
  `batch_size = floor(target_vram_pct * total_vram_gb / per_fold_gb)`, clipped to `[1, 228]`
- **Predicted utilization metric**:
  `pred_torch_util_pct = 100 * (batch_size * per_fold_gb) / total_vram_gb`

The formula sizes batches so that predicted torch allocation stays near 65% of total VRAM,
leaving headroom for framework overhead and non-batch allocations. The batch size cap of 228
corresponds to the total number of folds (n_folds = 228).

## Summary (Predicted Torch Utilization)

| GPU | n_confounders | min % | max % | mean % |
| --- | ---: | ---: | ---: | ---: |
| A6000 (48 GB) | 1 | 64.42 | 64.92 | 64.74 |
| A6000 (48 GB) | 5 | 63.56 | 64.95 | 64.28 |
| A6000 (48 GB) | 10 | 61.31 | 64.98 | 63.61 |
| A100 (80 GB) | 1 | 41.70 | 64.92 | 58.14 |
| A100 (80 GB) | 5 | 61.19 | 64.92 | 64.11 |
| A100 (80 GB) | 10 | 63.63 | 64.98 | 64.42 |

Note: A100 + `n_confounders=1` at low `p` undershoots because `batch_size` hits the fold cap (228)
before the VRAM budget is fully used. For the A6000 with `n_confounders=10` at high `p`, the large
per-fold cost means each additional fold adds ~1.8 GB, so quantization waste becomes more significant.

## A6000 (48 GB)

### n_confounders = 1

| p | f = n_c * p | per_fold_gb | batch_size | pred_util % |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.1463 | 213 | 64.92 |
| 2 | 2 | 0.1634 | 190 | 64.68 |
| 3 | 3 | 0.1805 | 172 | 64.68 |
| 4 | 4 | 0.1976 | 157 | 64.63 |
| 5 | 5 | 0.2147 | 145 | 64.86 |
| 6 | 6 | 0.2318 | 134 | 64.71 |
| 7 | 7 | 0.2489 | 125 | 64.82 |
| 8 | 8 | 0.2660 | 117 | 64.84 |
| 9 | 9 | 0.2831 | 110 | 64.88 |
| 10 | 10 | 0.3002 | 103 | 64.42 |

### n_confounders = 5

| p | f = n_c * p | per_fold_gb | batch_size | pred_util % |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0.2147 | 145 | 64.86 |
| 2 | 10 | 0.3002 | 103 | 64.42 |
| 3 | 15 | 0.3857 | 80 | 64.28 |
| 4 | 20 | 0.4712 | 66 | 64.79 |
| 5 | 25 | 0.5567 | 56 | 64.95 |
| 6 | 30 | 0.6422 | 48 | 64.22 |
| 7 | 35 | 0.7277 | 42 | 63.67 |
| 8 | 40 | 0.8132 | 38 | 64.38 |
| 9 | 45 | 0.8987 | 34 | 63.66 |
| 10 | 50 | 0.9842 | 31 | 63.56 |

### n_confounders = 10

| p | f = n_c * p | per_fold_gb | batch_size | pred_util % |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 0.3002 | 103 | 64.42 |
| 2 | 20 | 0.4712 | 66 | 64.79 |
| 3 | 30 | 0.6422 | 48 | 64.22 |
| 4 | 40 | 0.8132 | 38 | 64.38 |
| 5 | 50 | 0.9842 | 31 | 63.56 |
| 6 | 60 | 1.1552 | 27 | 64.98 |
| 7 | 70 | 1.3262 | 23 | 63.55 |
| 8 | 80 | 1.4972 | 20 | 62.38 |
| 9 | 90 | 1.6682 | 18 | 62.56 |
| 10 | 100 | 1.8392 | 16 | 61.31 |

## A100 (80 GB)

### n_confounders = 1

| p | f = n_c * p | per_fold_gb | batch_size | pred_util % |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.1463 | 228 | 41.70 |
| 2 | 2 | 0.1634 | 228 | 46.57 |
| 3 | 3 | 0.1805 | 228 | 51.44 |
| 4 | 4 | 0.1976 | 228 | 56.32 |
| 5 | 5 | 0.2147 | 228 | 61.19 |
| 6 | 6 | 0.2318 | 224 | 64.90 |
| 7 | 7 | 0.2489 | 208 | 64.71 |
| 8 | 8 | 0.2660 | 195 | 64.84 |
| 9 | 9 | 0.2831 | 183 | 64.76 |
| 10 | 10 | 0.3002 | 173 | 64.92 |

### n_confounders = 5

| p | f = n_c * p | per_fold_gb | batch_size | pred_util % |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0.2147 | 228 | 61.19 |
| 2 | 10 | 0.3002 | 173 | 64.92 |
| 3 | 15 | 0.3857 | 134 | 64.60 |
| 4 | 20 | 0.4712 | 110 | 64.79 |
| 5 | 25 | 0.5567 | 93 | 64.72 |
| 6 | 30 | 0.6422 | 80 | 64.22 |
| 7 | 35 | 0.7277 | 71 | 64.58 |
| 8 | 40 | 0.8132 | 63 | 64.04 |
| 9 | 45 | 0.8987 | 57 | 64.03 |
| 10 | 50 | 0.9842 | 52 | 63.97 |

### n_confounders = 10

| p | f = n_c * p | per_fold_gb | batch_size | pred_util % |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 0.3002 | 173 | 64.92 |
| 2 | 20 | 0.4712 | 110 | 64.79 |
| 3 | 30 | 0.6422 | 80 | 64.22 |
| 4 | 40 | 0.8132 | 63 | 64.04 |
| 5 | 50 | 0.9842 | 52 | 63.97 |
| 6 | 60 | 1.1552 | 45 | 64.98 |
| 7 | 70 | 1.3262 | 39 | 64.65 |
| 8 | 80 | 1.4972 | 34 | 63.63 |
| 9 | 90 | 1.6682 | 31 | 64.64 |
| 10 | 100 | 1.8392 | 28 | 64.37 |
