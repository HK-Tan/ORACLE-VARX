# Toy Benchmark Plan: 3-Variable Time-Varying Confounded System

## 1. Objective

Design a synthetic benchmark that is fully aligned with the existing ORACLE-VARX pipeline and directly evaluates:

1. Unbiased estimation of lagged lead-lag coefficients (interpreted as a time-unrolled lagged DAG).
2. Recovery of the correct lag order (`p*`) over time across three temporal regimes.
3. Robustness under full, partial, and hidden confounder observability.
4. Forecast quality as a secondary diagnostic (not the primary scientific claim).

This plan is synthetic-first and intentionally excludes additional empirical benchmarks at this stage.

---

## 2. Core Design Decisions

1. Use three endogenous variables: `X_t`, `Y_t`, `Z_t` (ordered `[X, Y, Z]`).
2. Use three true confounders: `W1_t`, `W2_t`, `W3_t`.
3. Keep nonlinear nuisance effects in **all** experiments (fixed functional form).
4. Lag-1 cross effects form a cycle: `Z→X→Y→Z` (product 0.6³ = 0.216, stable).
5. Three time-varying self-effects decay to zero at different rates, creating three regimes.
6. Confounder edges are stationary (do not vary over time).
7. Evaluate full, partial-2, partial-1, and no-confounder observability.
8. Run the full existing model comparison pipeline and compare outputs.
9. No contemporaneous edges — `ε_X, ε_Y, ε_Z` are mutually independent.
10. No acceptance criteria — results are exploratory. The plots + metrics are the deliverable.

---

## 3. Time-Unrolled DAG Interpretation

We assume a repeated-time structural graph:

1. If `W_{t-1} → (X_t, Y_t, Z_t)` exists, the same relationship appears at all adjacent times by stationarity of structural form.
2. Lagged causal effects are represented by edges from node `_{t-k}` to node `_t`.
3. The estimated coefficient blocks at each lag correspond to weighted adjacency blocks of this lagged DAG.

---

## 4. Data-Generating Process (DGP)

### 4.1 Variables and Dimensions

- **Endogenous**: `X_t, Y_t, Z_t` (n=3, ordered `[X, Y, Z]`)
- **Confounders**: `W1_t, W2_t, W3_t` (n=3)
- **Time**: t = 0, 1, ..., 2999 (T = 3000)

### 4.2 Confounder Process (Independent AR(1))

```
W_{i,t} = 0.5 * W_{i,t-1} + η_{i,t},  i ∈ {1, 2, 3}
η_{i,t} ~ N(0, 1), mutually independent
```

### 4.3 Nonlinear Confounder Nuisance (Normalized, Fixed Form)

```
g_X(W_{t-1}) = (tanh(2·W1) + W2² + sin(π·W3)) / 2.2
g_Y(W_{t-1}) = (ReLU(W1 - 0.5) + sin(π·W2) + W3²) / 2.1
g_Z(W_{t-1}) = (W1·W2 + tanh(W3) + cos(π·W1)) / 1.7
```

Each function uses distinct nonlinear forms (tanh, quadratic, sine, ReLU, interaction, cosine) and is **normalized to std ≈ 1** under the AR(1) confounder distribution. This keeps nuisance consistently nonlinear and favors nonlinear first-stage learners for orthogonalization.

The nuisance enters the structural equations as **ν · λ · g(W_{t-1})**, where:
- **ν** (noise_scale) controls the overall noise floor — both innovation and confounding scale with ν
- **λ** (confounder_strength) scales confounding relative to innovation noise (λ=1 means equal strength)

This design means lowering ν produces uniformly cleaner estimates, while λ independently controls how much DML has to correct.

### 4.4 Structural Equations

```
X_t = 0.60·Z_{t-1} + a_XX(t)·X_{t-2}                + ν·λ·g_X(W_{t-1}) + ν·ε_X,t
Y_t = 0.60·X_{t-1} + a_YY(t)·Y_{t-2}                + ν·λ·g_Y(W_{t-1}) + ν·ε_Y,t
Z_t = 0.60·Y_{t-1}                    + a_ZZ(t)·Z_{t-3} + ν·λ·g_Z(W_{t-1}) + ν·ε_Z,t
```

- `ε_X, ε_Y, ε_Z ~ N(0,1)`, mutually independent (no contemporaneous edges)
- `ν = 0.05` is the noise scaling parameter (CLI: `--noise-scale`); scales both innovation noise and confounding
- `λ = 1.0` is the confounder strength relative to innovation noise (CLI: `--confounder-strength`)
- `c_x = c_y = c_z = 0` — the VAR model's intercept term absorbs any non-zero mean from nuisance functions

### 4.5 Time-Varying Coefficients

```
a_XX(t) = 0.30 · max(0, 1 - t/2000)     # X_{t-2}→X_t, zero at t=2000
a_YY(t) = 0.30 · max(0, 1 - t/2000)     # Y_{t-2}→Y_t, zero at t=2000
a_ZZ(t) = 0.30 · max(0, 1 - t/1000)     # Z_{t-3}→Z_t, zero at t=1000
```

### 4.6 Three Regimes

| Regime | Time Range | Active Edges | True Max Lag | What Disappears |
|--------|-----------|-------------|-------------|-----------------|
| 1 | [0, 1000) | 6 edges | 3 | — |
| 2 | [1000, 2000) | 5 edges | 2 | Z_{t-3}→Z_t gone |
| 3 | [2000, 3000] | 3 edges (lag-1 only) | 1 | X_{t-2}→X_t, Y_{t-2}→Y_t gone |

### 4.7 True Coefficient Matrices

Convention: `A_k[i, j]` = effect of variable j at lag k on variable i. Variables ordered `[X, Y, Z]`.

```
         X      Y      Z
A_1 = [[ 0,     0,     0.60],    ← Z_{t-1} → X_t
       [ 0.60,  0,     0   ],    ← X_{t-1} → Y_t
       [ 0,     0.60,  0   ]]    ← Y_{t-1} → Z_t

A_2(t) = [[a_XX(t), 0,       0      ],
          [0,       a_YY(t), 0      ],
          [0,       0,       0      ]]

A_3(t) = [[0, 0, 0      ],
          [0, 0, 0      ],
          [0, 0, a_ZZ(t)]]

A_4 = A_5 = zeros(3, 3)
```

### 4.8 Stability Analysis

A VAR(p) system is stable (stationary) iff all eigenvalues of the companion matrix have modulus < 1. For VAR(3) with 3 variables, the 9×9 companion matrix is:

```
C = [A_1  A_2  A_3]
    [I    0    0  ]
    [0    I    0  ]
```

The **spectral radius** ρ(C) must satisfy ρ(C) < 1.

Checking only the lag-1 cycle product (cross_effect³) is **insufficient** — the self-effects at lags 2–3 interact with the cycle and can push eigenvalues outside the unit circle.

| cross_effect | self_effect | ρ(C) | Stable? |
|:---:|:---:|:---:|:---:|
| 0.80 | 0.40 | **1.14** | No |
| 0.70 | 0.25 | 0.96 | Yes |
| 0.65 | 0.30 | 0.96 | Yes |
| 0.60 | 0.35 | 0.97 | Yes |
| 0.60 | 0.30 | **0.93** | **Yes (chosen)** |
| 0.30 | 0.20 | 0.72 | Yes (original) |

The chosen parameters (cross=0.60, self=0.30, ν=0.05) give ρ = 0.93, providing comfortable stability margin.

### 4.9 Estimation Variance and Autocorrelation in Edge Trajectories

The edge trajectory plots show estimated coefficients fluctuating around the true values. This variance has three key properties:

1. **It is estimation variance, not DGP noise.** The innovation noise is tiny (ν=0.05). The fluctuations arise from finite-sample OLS estimation with `ols_window=100` observations and up to 15 parameters per equation (VAR(5) with 3 variables). For VARX, it worsens to 30 parameters from 100 observations.

2. **The fluctuations appear autocorrelated** even though the innovation noise ε is i.i.d. This is because adjacent rolling OLS windows share 99 out of 100 observations — each day's estimate only differs by dropping one old observation and adding one new one. The estimates therefore change slowly, producing smooth "wiggles" rather than i.i.d. jitter.

3. **Nuisance contributes to estimation error** for VAR/VARX methods. For VAR (no confounders), the nonlinear nuisance g(W_{t-1}) enters the OLS error term, inflating residual variance. For VARX (confounders included linearly), the nonlinear residual still leaks through. DML methods (OR-VARX, ORACLE-VARX) should reduce this by partialling out the nonlinear confounder effect in the first stage.

The primary lever for reducing estimation variance is the ratio of observations to parameters: increasing `ols_window` or reducing `p_max` helps, at the cost of slower adaptation to regime transitions.

---

## 5. Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| T | 3000 | Covers all 3 regimes |
| n_assets | 3 | X, Y, Z |
| n_confounders | 3 | W1, W2, W3 |
| p_max | 5 | True max lag is 3 |
| ols_window | 200 | OLS training window |
| tree_train_window | None→200 | DML first-stage window (defaults to ols_window) |
| p_max_offset | 5 | Extra days for lag computation |
| test_size | 20 | Grid fold size |
| validation_days | 20 | Rolling validation window |
| lookback_var | 105 | = ols_window + p_max_offset |
| lookback_orvarx | 205 | = tree_train + ols + offset |
| ν (noise) | 0.05 | Innovation noise scaling |
| λ (confounder_strength) | 1.0 | Nuisance scaling; increase to stress-test DML |
| cross_effect | 0.60 | Lag-1 cycle strength |
| a_XX_init | 0.30 | Lag-2 self-effect (X) |
| a_YY_init | 0.30 | Lag-2 self-effect (Y) |
| a_ZZ_init | 0.30 | Lag-3 self-effect (Z) |
| seed | 42 | Fixed for reproducibility |
| burn_in | 50 | Discarded at DGP start |

Output days: VAR = 2875, DML = 2775. Both cover all three regimes.

---

## 6. Data Generation & Storage

Synthetic data is generated **once** and saved to CSV for reproducibility and easy parallelization across compute jobs.

**Files saved to `dataset/toy/`**:
- `Y.csv` — endogenous variables (T × 3), columns: X, Y, Z
- `W.csv` — confounders (T × 3), columns: W1, W2, W3
- `A_true.pt` — ground truth coefficient tensor (T × 3 × 3 × 3)
- `dgp_config.json` — all DGP parameters for full reproducibility

**Generation script**: `scripts/generate_toy_data.py` — runs standalone, generates and saves all files. Uses fixed seed=42.

**Experiment scripts** load from CSV (like real-data experiments), never regenerate. This decouples data generation from model fitting.

---

## 7. Observability Conditions

| Level | W_obs | Used By |
|-------|-------|---------|
| None | [] | VAR, ACLE-VAR |
| All | [W1, W2, W3] | VARX, ACLE-VARX, OR-VARX, ORACLE-VARX, OR-VARX-TabPFN, ORACLE-VARX-TabPFN |
| Partial-2 | [W1, W2] | VARX, ACLE-VARX, OR-VARX, ORACLE-VARX, OR-VARX-TabPFN, ORACLE-VARX-TabPFN |
| Partial-1 | [W1] | VARX, ACLE-VARX, OR-VARX, ORACLE-VARX, OR-VARX-TabPFN, ORACLE-VARX-TabPFN |

---

## 8. Experiment Matrix (phased)

### Phase 0 (CPU, sequential): OLS baselines — 8 runs

| # | Method | Confounders | Obs Level | Fit Function |
|---|--------|-------------|-----------|--------------|
| 1 | VAR | No | N/A | `fit_var(Y)` |
| 2 | ACLE-VAR | No | N/A | `fit_aclevarx(Y)` |
| 3 | VARX (all) | OLS | all | `fit_var(Y_combined)` slice |
| 4 | ACLE-VARX (all) | OLS | all | `fit_aclevarx(Y_combined)` slice |
| 5 | VARX (p2) | OLS | partial_2 | `fit_var(Y_combined)` slice |
| 6 | ACLE-VARX (p2) | OLS | partial_2 | `fit_aclevarx(Y_combined)` slice |
| 7 | VARX (p1) | OLS | partial_1 | `fit_var(Y_combined)` slice |
| 8 | ACLE-VARX (p1) | OLS | partial_1 | `fit_aclevarx(Y_combined)` slice |

### Phase 1 (CPU, parallelizable per obs level): DML methods — 6 runs per learner

| # | Method | Obs Level | Fit Function |
|---|--------|-----------|--------------|
| 9 | OR-VARX (all) | all | `fit_orvarx_batched(Y, W_obs)` |
| 10 | ORACLE-VARX (all) | all | `fit_oraclevarx_batched(Y, W_obs)` |
| 11 | OR-VARX (p2) | partial_2 | `fit_orvarx_batched(Y, W_obs)` |
| 12 | ORACLE-VARX (p2) | partial_2 | `fit_oraclevarx_batched(Y, W_obs)` |
| 13 | OR-VARX (p1) | partial_1 | `fit_orvarx_batched(Y, W_obs)` |
| 14 | ORACLE-VARX (p1) | partial_1 | `fit_oraclevarx_batched(Y, W_obs)` |

Default learner: `extra_trees`. Multi-learner mode (`--learner all`): runs all 4 learners (lgbm, xgboost, rf, extra_trees) = 24 additional runs.

### Phase 2 (GPU): TabPFN methods — 6 runs

| # | Method | Obs Level | Fit Function |
|---|--------|-----------|--------------|
| 15 | OR-VARX-TabPFN (all) | all | `fit_oraclevarx_tabpfn(Y, W_obs)` |
| 16 | ORACLE-VARX-TabPFN (all) | all | `fit_oraclevarx_tabpfn(Y, W_obs)` |
| 17 | OR-VARX-TabPFN (p2) | partial_2 | `fit_oraclevarx_tabpfn(Y, W_obs)` |
| 18 | ORACLE-VARX-TabPFN (p2) | partial_2 | `fit_oraclevarx_tabpfn(Y, W_obs)` |
| 19 | OR-VARX-TabPFN (p1) | partial_1 | `fit_oraclevarx_tabpfn(Y, W_obs)` |
| 20 | ORACLE-VARX-TabPFN (p1) | partial_1 | `fit_oraclevarx_tabpfn(Y, W_obs)` |

### Key optimization: shared DML first stage
For each observability level, `fit_orvarx_core()` runs once and its `core_results` are reused by both OR-VARX and ORACLE-VARX.

### Coefficient sharing
- **VAR and ACLE-VAR** share the same OLS coefficients (ACLE only differs in p-selection)
- **VARX and ACLE-VARX** share the same OLS coefficients (ACLE only differs in p-selection)
- **OR-VARX and ORACLE-VARX** share the same DML-deconfounded coefficients (ORACLE only differs in p-selection)

Only `VARXResult` stores `.coefficients`. `ACLEVARXResult` and `ORACLEVARXResult` do not. For coefficient evaluation of ACLE/ORACLE methods, use the coefficients from their base method (VAR/OR-VARX respectively).

---

## 9. Metrics & Evaluation

### (i) Edge Recovery: MAE/MSE on true coefficients

Compare `result.coefficients[d, k, i, j]` vs `A_true[t, k, i, j]` at each output day.

**Alignment**: Parse absolute time index from date strings (e.g., `"T226"` → t=226). Then index into `A_true[t]`.

Report:
- MAE/MSE on **non-zero** true edges (signal recovery)
- MAE on **zero** true edges (false positive mass)
- Breakdown by regime (1, 2, 3)
- Only compare first `min(p_max_est, p_max_true)` = 3 lags; lags 4-5 are pure false positives

### (ii) Forecast Error: MSE (tuning target) + MAE (reporting)

Compare `result.forecasts[:, d]` vs `Y[t, :]` for each output day d at absolute time t.
- **Tuned on rolling validation MSE** (same mechanism as real-data experiments)
- Report both **out-of-sample MSE** and **out-of-sample MAE**
- Report per-regime and overall

### (iii) Forecast Error Plot (replaces PnL plots)

Two separate plots per (method, observability) run:
- `forecast_error_mse.png`: rolling MSE (rolling window = 50)
- `forecast_error_mae.png`: rolling MAE (rolling window = 50)
- X-axis: time (output days), range [0, 3000]
- Regime boundaries at t=1000 and t=2000 (vertical dashed lines)

### (iv) Coefficient Recovery Plots

For each (method, observability) pair:
1. **Lag analysis**: `p_optimal` vs time (existing `plot_lag_analysis()`)
2. **Coefficient heatmaps**: at max_p day and last day (existing `plot_coefficient_heatmap()`)
3. **Per-p coefficient evolution grids**: at max_p day and last day (existing `plot_coefficient_evolution_per_p()` via `refit_*_coefficients_for_day()`)

**Edge trajectory plots**: Each experiment gets its own `edge_trajectories.png` — a 6-panel figure showing true (black) vs estimated coefficients over time for all 6 edges. X-axis range [0, 3000].

---

## 10. Required Outputs

For each (method, observability) run, saved to `results-toy/{method}_{obs_level}/`:
- `edge_trajectories.png` — true vs estimated coefficient trajectories (6 edges)
- `lag_analysis.png` — p_optimal over time
- `forecast_error_mse.png` — rolling MSE vs time
- `forecast_error_mae.png` — rolling MAE vs time
- `heatmap_max_p.png` — coefficient heatmap at the day with highest p_optimal
- `heatmap_last_day.png` — coefficient heatmap at the last output day
- `coef_evolution_max_p.png` — per-p refit grid at max_p day
- `coef_evolution_last_day.png` — per-p refit grid at last day
- `metrics.json` — edge_mae, edge_mse, zero_edge_mae, forecast_mae, forecast_mse (by regime)
- `result.pt` — saved result object

Top-level outputs in `results-toy/`:
- `metrics_summary.csv` — incrementally-appended consolidated table of all runs

---

## 11. Implementation

### Files created

| File | Purpose |
|------|---------|
| `src/synthetic/__init__.py` | Package exports: `ToyDGPConfig`, `GroundTruth`, `generate_toy_data`, `get_observed_confounders` |
| `src/synthetic/dgp.py` | DGP dataclasses, generation functions, observability slicing |
| `scripts/generate_toy_data.py` | Standalone data generation → `dataset/toy/` |
| `scripts/run_toy_benchmark.py` | Main experiment runner (Phase 0 + Phase 1 + Phase 2 + per-experiment evaluation) |
| `scripts/run_all_toy_experiments.py` | Tmux orchestrator for parallel Phase 1 (4 learners in 2×2 grid) |

### Key implementation details

<!-- FIX APPLIED: Date slicing differs between fit functions.
     - fit_var() expects n_test_days dates:       dates[lookback_var:]
     - fit_aclevarx() expects n_output_days dates: dates[lookback_var + validation_days:]
     - fit_orvarx_batched() expects n_output_days: dates[lookback_orvarx + validation_days:]
     - fit_oraclevarx_batched() same as orvarx:    dates[lookback_orvarx + validation_days:]
     The difference: fit_var internally trims validation_days from the dates it receives,
     while the other functions expect pre-trimmed dates. -->

**Date generation**: Uses string indices `"T{absolute_index}"`. Slicing differs per function:
- `fit_var`: `dates[lookback_var:]` (function trims validation internally)
- `fit_aclevarx`: `dates[lookback_var + validation_days:]` (expects output-day dates)
- `fit_orvarx_batched`: `dates[lookback_orvarx + validation_days:]` (expects output-day dates)
- `fit_oraclevarx_batched`: `dates[lookback_orvarx + validation_days:]` (expects output-day dates)
- `fit_oraclevarx_tabpfn`: `dates[lookback_orvarx + validation_days:]` (expects output-day dates)

**VARX/ACLE-VARX slicing** (following `run_combined_experiment.py` pattern):
```python
endo_indices = list(range(n_endo))  # [0, 1, 2]
coefficients = result_full.coefficients[:, :, endo_indices, :][:, :, :, endo_indices]
forecasts = result_full.forecasts[endo_indices, :]
```

**VARX/ACLE-VARX refit labeling**: The OLS refit operates on Y_combined (endo + confounders). The `refit_asset_names` parameter passes `combined_names = ENDO_NAMES + obs_names` so heatmaps and evolution plots show all dimensions with correct labels (e.g., X, Y, Z, W1, W2, W3 for obs=all).

**Heatmap generation for ACLE/ORACLE**: Since they don't store coefficients, use the base method's result (VAR for ACLE, OR-VARX for ORACLE) with `get_coefficient_heatmap_matrix()`.

### Key files reused

| Component | File | Function/Class |
|-----------|------|---------------|
| VAR fit | `src/models/var_pytorch.py` | `fit_var()` |
| ACLE fit | `src/models/acle_var.py` | `fit_aclevarx()` |
| DML core | `src/models/dml_pytorch.py` | `fit_orvarx_core()`, `fit_orvarx_batched()` |
| ORACLE fit | `src/models/oracle_var.py` | `fit_oraclevarx_batched()` |
| TabPFN fit | `src/models/oracle_var_tabpfn.py` | `fit_oraclevarx_tabpfn()` |
| Grid config | `src/modules/grid_config.py` | `GridConfig` |
| Result types | `src/results.py` | `VARXResult`, `ACLEVARXResult`, `ORACLEVARXResult` |
| CPU detect | `src/models/dml_pytorch.py` | `get_physical_cpu_count()` |
| Lag plot | `src/evaluation/plotting.py` | `plot_lag_analysis()` |
| Heatmap | `src/evaluation/plotting.py` | `plot_coefficient_heatmap()` |
| Per-p grid | `src/evaluation/plotting.py` | `plot_coefficient_evolution_per_p()` |
| Coef refit | `src/models/coefficient_refit.py` | `refit_var_coefficients_for_day()`, `refit_dml_coefficients_for_day()`, `get_target_days()` |

---

## 12. Verification

1. **Smoke test**: Run VAR on first 500 days (regime 1), expect `p_optimal ≈ 3`
2. **Coefficient check**: VAR A_1 estimate should be near `[[0,0,0.6],[0.6,0,0],[0,0.6,0]]`
3. **Regime transitions**: `p_optimal` should shift 3→2 near t=1000, 2→1 near t=2000
4. **DML advantage**: Under "all" confounders, OR-VARX edge MAE should be lower than VAR edge MAE
5. **Decay tracking**: Estimated a_XX(t) should track the linear decay from 0.30 to 0

---

## 13. Notes

- **Correlated noise**: Deferred. Independent noise chosen for clean evaluation. Confounders already create cross-sectional dependence via shared W_{t-1}→(X,Y,Z)_t. Future extension: add Σ covariance matrix.
- **Noise sweep**: ν ∈ {0.25, 0.5, 1.0, 2.0} as optional extension for SNR analysis.
- **Multi-learner Phase 1**: Supports `--learner all` to run lgbm, xgboost, rf, extra_trees (4 learners × 3 obs × 2 methods = 24 runs).
- **F1/recall/precision**: Extensible by thresholding estimated coefficients to binary edge presence. Noted as potential future tuning objective.
- **TabPFN Phase 2**: GPU-accelerated first-stage via `fit_oraclevarx_tabpfn`. Batch size heuristic generalized with `n_assets` parameter for smaller problems.
