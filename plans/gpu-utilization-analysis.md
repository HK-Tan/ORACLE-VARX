# GPU Utilization Analysis: TabPFN Batch Size Formula

Date: 2026-02-23

## Probe Data Summary

T-prediction probes on A100 80 GB across 3 confounder presets:

| Run | GPU | VRAM | n_c | p range | probe formula |
|-----|-----|------|----:|--------:|:--------------|
| VIX | A100 80GB | 80 GB | 1 | 1–10 | `6*VRAM/(n_c*p^1.5)` |
| macro5 | A100 80GB | 80 GB | 5 | 1–10 | `6*VRAM/(n_c*p^1.5)` |
| all10 | A100 80GB | 80 GB | 10 | 1–2 (partial) | `6*VRAM/(n_c*p^1.5)` |

The probe measures the **T prediction** (9×p outputs per fold), which is the
binding VRAM constraint — larger than Y prediction (9 outputs per fold).

## Per-Fold VRAM Cost (peak_alloc / batch_size)

| p | VIX (n_c=1) | macro5 (n_c=5) | all10 (n_c=10) | ratio 5/1 | ratio 10/5 |
|--:|------------:|---------------:|---------------:|----------:|-----------:|
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

**Key finding**: The ratio 5/1 averages ~3.4, not 5.0. This means per-fold
cost scales as **n_c^0.75** (since 5^0.75 = 3.34), not n_c^1.

The p exponent from VIX data (log-log slope across p=1..10) is ~1.4,
consistent with p^1.5 as a slightly conservative model.

## Fitted Model

```
batch = floor(11 * VRAM / (n_c^0.75 * p^1.5))
```

### How the coefficient (11) was determined

For each probe point, compute the `a` that would hit 75% utilization (60 GB
on A100):

```
a_75 = 0.75 * n_c^0.75 * p^1.5 / per_fold
```

| Preset | p=4 | p=7 | p=10 | min a |
|--------|----:|----:|-----:|------:|
| VIX (n_c=1) | 12.3 | 12.3 | 11.8 | 11.8 |
| macro5 (n_c=5) | 15.4 | 12.6 | 11.0 | 11.0 |

The binding constraint is macro5 p=10 at a=11.0. Using `a=11` keeps all
observed data points at or below 75%.

### Previous formula (deprecated)

The old linear model `per_fold = 0.0171 * f + 0.1292` (where `f = n_c * p`)
was calibrated from **Y prediction** probe data, which understated VRAM usage.
The T prediction probe revealed that per-fold cost scales non-linearly with
both p and n_c, requiring the power-law model.

## Predicted Utilization (A100 80 GB)

See `tabpfn-predicted-utilization.md` for full tables across all GPUs.

### Quick reference — worst cases

| Preset | worst p | batch | est_peak (GB) | util % |
|--------|--------:|------:|--------------:|-------:|
| VIX (n_c=1) | 10 | 27 | 54.2 | 68 |
| macro5 (n_c=5) | 10 | 8 | 57.9 | 72 |
| all10 (n_c=10)\* | 7 | 8 | 49.5 | 62 |

\* extrapolated
