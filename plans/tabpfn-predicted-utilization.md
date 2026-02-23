# TabPFN Predicted Utilization (A100 / A6000)

Date: 2026-02-23

## Formula

```
batch = floor(11 * VRAM_GB / (n_c^0.75 * p^1.5))
```

Calibrated from **T-prediction** probe data on A100 80 GB (VIX n_c=1, macro5
n_c=5, all10 n_c=10 partial). Per-fold VRAM scales sub-linearly with
n_confounders (~n_c^0.75) and as p^1.5 with lag order. The coefficient (11)
targets **~72% worst-case peak utilization**, leaving headroom for CUDA
workspace overhead.

Batch is capped at n_folds (228) and floored at 1.

## Probe Data (A100 80 GB, T prediction, Feb 2026)

Probed with prior formula (`6 * VRAM / (n_c * p^1.5)`) to measure per-fold
VRAM cost at each (n_c, p) combination. Per-fold cost (`peak_alloc / batch`)
is approximately GPU-independent and used to predict utilization under the new
formula.

### VIX (n_c=1)

| p | probe_batch | peak_alloc (GB) | peak_res (GB) | per_fold (GB) | time (s) |
|--:|------------:|----------------:|--------------:|--------------:|---------:|
| 1 | 228 | 18.5 | 20.0 | 0.081 | 168 |
| 2 | 169 | 27.5 | 29.9 | 0.163 | 67 |
| 3 | 92 | 22.4 | 24.3 | 0.243 | 46 |
| 4 | 60 | 29.2 | 31.6 | 0.487 | 85 |
| 5 | 42 | 25.5 | 27.6 | 0.607 | 111 |
| 6 | 32 | 23.3 | 25.1 | 0.728 | 173 |
| 7 | 25 | 28.3 | 30.5 | 1.132 | 197 |
| 8 | 21 | 27.1 | 29.3 | 1.290 | 226 |
| 9 | 17 | 24.6 | 26.6 | 1.447 | 222 |
| 10 | 15 | 30.1 | 32.5 | 2.007 | 350 |

### macro5 (n_c=5)

| p | probe_batch | peak_alloc (GB) | peak_res (GB) | per_fold (GB) | time (s) |
|--:|------------:|----------------:|--------------:|--------------:|---------:|
| 1 | 96 | 11.8 | 12.8 | 0.123 | 156 |
| 2 | 33 | 13.5 | 14.5 | 0.409 | 79 |
| 3 | 18 | 13.2 | 14.3 | 0.733 | 119 |
| 4 | 12 | 15.6 | 16.9 | 1.300 | 220 |
| 5 | 8 | 16.2 | 17.5 | 2.025 | 349 |
| 6 | 6 | 16.0 | 17.3 | 2.667 | 389 |
| 7 | 5 | 18.4 | 19.9 | 3.680 | 602 |
| 8 | 4 | 19.4 | 20.9 | 4.850 | 761 |
| 9 | 3 | 17.4 | 18.8 | 5.800 | 796 |
| 10 | 3 | 21.7 | 23.5 | 7.233 | ~900 |

### all10 (n_c=10, partial — p=1,2 measured, rest extrapolated)

| p | probe_batch | peak_alloc (GB) | peak_res (GB) | per_fold (GB) | time (s) |
|--:|------------:|----------------:|--------------:|--------------:|---------:|
| 1 | 48 | 9.8 | 10.6 | 0.204 | 162 |
| 2 | 16 | 10.5 | 11.3 | 0.656 | — |

Extrapolated per_fold for p=3..10 uses macro5 values scaled by
(10/5)^0.75 = 1.682. Early data confirms this: predicted p=1 was 0.207
(actual 0.204, 1.5% off); predicted p=2 was 0.688 (actual 0.656, 4.9% off).

## Per-Fold Scaling Analysis

Per-fold VRAM = peak_alloc / batch_size. Comparing across n_c at the same p:

| p | n_c=1 | n_c=5 | n_c=10 | ratio 5/1 | ratio 10/5 |
|--:|------:|------:|-------:|----------:|-----------:|
| 1 | 0.081 | 0.123 | 0.204 | 1.52 | 1.66 |
| 2 | 0.163 | 0.409 | 0.656 | 2.51 | 1.60 |
| 3 | 0.243 | 0.733 | — | 3.01 | — |
| 4 | 0.487 | 1.300 | — | 2.67 | — |
| 5 | 0.607 | 2.025 | — | 3.34 | — |
| 6 | 0.728 | 2.667 | — | 3.66 | — |
| 7 | 1.132 | 3.680 | — | 3.25 | — |
| 8 | 1.290 | 4.850 | — | 3.76 | — |
| 9 | 1.447 | 5.800 | — | 4.01 | — |
| 10 | 2.007 | 7.233 | — | 3.60 | — |

If n_c scaled linearly, ratio 5/1 would be 5.0. The observed median is ~3.4,
matching **n_c^0.75** (since 5^0.75 = 3.34). The ratio 10/5 of ~1.63 also
matches 2^0.75 = 1.68.

## Summary (Predicted Peak Utilization)

| GPU | n_c | min % | max % | mean % | worst-case p |
|-----|----:|------:|------:|-------:|:-------------|
| A100 (80 GB) | 1 | 23 | 68 | 55 | p=10 |
| A100 (80 GB) | 5 | 35 | 72 | 56 | p=10 |
| A100 (80 GB) | 10\* | 40 | 62 | 54 | p=7 |
| A6000 (48 GB) | 1 | 39 | 67 | 58 | p=4,10 |
| A6000 (48 GB) | 5 | 40 | 61 | 54 | p=7,8 |
| A6000 (48 GB) | 10\* | 40 | 68 | 54 | p=8 |

\* n_c=10 per-fold extrapolated from macro5 using (10/5)^0.75 scaling; early
probe data (p=1,2) confirms predictions within 5%.

All worst cases under 75%. Low-p values for VIX on A100 undershoot because
batch is capped at n_folds=228.

## A100 (80 GB)

### n_c = 1 (VIX)

| p | batch | per_fold (GB) | est_peak (GB) | util % |
|--:|------:|--------------:|--------------:|-------:|
| 1 | 228\* | 0.081 | 18.5 | 23 |
| 2 | 228\* | 0.163 | 37.1 | 46 |
| 3 | 169 | 0.243 | 41.1 | 51 |
| 4 | 110 | 0.487 | 53.5 | 67 |
| 5 | 78 | 0.607 | 47.4 | 59 |
| 6 | 59 | 0.728 | 43.0 | 54 |
| 7 | 47 | 1.132 | 53.2 | 67 |
| 8 | 38 | 1.290 | 49.0 | 61 |
| 9 | 32 | 1.447 | 46.3 | 58 |
| 10 | 27 | 2.007 | 54.2 | 68 |

\* capped at n_folds=228

### n_c = 5 (macro5)

| p | batch | per_fold (GB) | est_peak (GB) | util % |
|--:|------:|--------------:|--------------:|-------:|
| 1 | 228\* | 0.123 | 28.0 | 35 |
| 2 | 93 | 0.409 | 38.1 | 48 |
| 3 | 50 | 0.733 | 36.7 | 46 |
| 4 | 32 | 1.300 | 41.6 | 52 |
| 5 | 23 | 2.025 | 46.6 | 58 |
| 6 | 17 | 2.667 | 45.3 | 57 |
| 7 | 14 | 3.680 | 51.5 | 64 |
| 8 | 11 | 4.850 | 53.4 | 67 |
| 9 | 9 | 5.800 | 52.2 | 65 |
| 10 | 8 | 7.233 | 57.9 | 72 |

\* capped at n_folds=228

### n_c = 10 (all10, per_fold extrapolated for p >= 3)

| p | batch | per_fold (GB) | est_peak (GB) | util % |
|--:|------:|--------------:|--------------:|-------:|
| 1 | 156 | 0.204 | 31.8 | 40 |
| 2 | 55 | 0.656 | 36.1 | 45 |
| 3 | 30 | 1.233\* | 37.0 | 46 |
| 4 | 19 | 2.187\* | 41.5 | 52 |
| 5 | 14 | 3.406\* | 47.7 | 60 |
| 6 | 10 | 4.487\* | 44.9 | 56 |
| 7 | 8 | 6.190\* | 49.5 | 62 |
| 8 | 6 | 8.158\* | 48.9 | 61 |
| 9 | 5 | 9.756\* | 48.8 | 61 |
| 10 | 4 | 12.168\* | 48.7 | 61 |

\* extrapolated from macro5 per_fold × 1.682

## A6000 (48 GB)

### n_c = 1 (VIX)

| p | batch | per_fold (GB) | est_peak (GB) | util % |
|--:|------:|--------------:|--------------:|-------:|
| 1 | 228\* | 0.081 | 18.5 | 39 |
| 2 | 186 | 0.163 | 30.3 | 63 |
| 3 | 101 | 0.243 | 24.5 | 51 |
| 4 | 66 | 0.487 | 32.1 | 67 |
| 5 | 47 | 0.607 | 28.5 | 59 |
| 6 | 35 | 0.728 | 25.5 | 53 |
| 7 | 28 | 1.132 | 31.7 | 66 |
| 8 | 23 | 1.290 | 29.7 | 62 |
| 9 | 19 | 1.447 | 27.5 | 57 |
| 10 | 16 | 2.007 | 32.1 | 67 |

\* capped at n_folds=228

### n_c = 5 (macro5)

| p | batch | per_fold (GB) | est_peak (GB) | util % |
|--:|------:|--------------:|--------------:|-------:|
| 1 | 157 | 0.123 | 19.3 | 40 |
| 2 | 55 | 0.409 | 22.5 | 47 |
| 3 | 30 | 0.733 | 22.0 | 46 |
| 4 | 19 | 1.300 | 24.7 | 51 |
| 5 | 14 | 2.025 | 28.4 | 59 |
| 6 | 10 | 2.667 | 26.7 | 56 |
| 7 | 8 | 3.680 | 29.4 | 61 |
| 8 | 6 | 4.850 | 29.1 | 61 |
| 9 | 5 | 5.800 | 29.0 | 60 |
| 10 | 4 | 7.233 | 28.9 | 60 |

### n_c = 10 (all10, per_fold extrapolated for p >= 3)

| p | batch | per_fold (GB) | est_peak (GB) | util % |
|--:|------:|--------------:|--------------:|-------:|
| 1 | 93 | 0.204 | 19.0 | 40 |
| 2 | 33 | 0.656 | 21.6 | 45 |
| 3 | 18 | 1.233\* | 22.2 | 46 |
| 4 | 11 | 2.187\* | 24.1 | 50 |
| 5 | 8 | 3.406\* | 27.2 | 57 |
| 6 | 6 | 4.487\* | 26.9 | 56 |
| 7 | 5 | 6.190\* | 31.0 | 64 |
| 8 | 4 | 8.158\* | 32.6 | 68 |
| 9 | 3 | 9.756\* | 29.3 | 61 |
| 10 | 2 | 12.168\* | 24.3 | 51 |

\* extrapolated from macro5 per_fold × 1.682

Note: A6000 n_c=10 p=10 gets batch=2 (floor discretization), dropping to 51%
utilization. This is the cost of small batches — each integer step is a large
relative change.
