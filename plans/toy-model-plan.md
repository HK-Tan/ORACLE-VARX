# Toy Model Plan: Unbiased Lead-Lag DAG Recovery in Time-Varying Confounded Systems

## 1. Objective

Design a synthetic benchmark that is fully aligned with the existing ORACLE-VARX pipeline and directly evaluates:

1. Unbiased estimation of lagged lead-lag coefficients (interpreted as a time-unrolled lagged DAG).
2. Recovery of the correct lag order (`p*`) over time.
3. Robustness under full, partial, and hidden confounder observability.
4. Forecast quality as a secondary diagnostic (not the primary scientific claim).

This plan is synthetic-first and intentionally excludes additional empirical benchmarks at this stage.

---

## 2. Core Design Decisions (Locked)

1. Use two endogenous variables: `X_t`, `Y_t`.
2. Use two true confounders: `W1_t`, `W2_t` (`W_t in R^2`).
3. Keep nonlinear nuisance effects in **all** experiments (fixed functional form).
4. Baseline lag structure includes:
1. `X_{t-1} -> Y_t`
2. `Y_{t-1} -> X_t`
3. `X_{t-2} -> X_t`
5. Only one edge is time-varying in the main benchmark: `X_{t-2} -> X_t`.
6. Decay shape for `X_{t-2} -> X_t` is piecewise linear to zero.
7. Confounder edges are stationary (do not vary over time in the main benchmark).
8. Evaluate full vs partial vs hidden observability of confounders.
9. Run the full existing model comparison pipeline and compare outputs.

Rationale for varying only one edge:
1. It makes lag-order transition attribution clean.
2. It allows a clear visual and statistical check that post-decay periods should favor `p*=1`.
3. It avoids ambiguity from simultaneously drifting confounding and lag dynamics.

---

## 3. Time-Unrolled DAG Interpretation

We assume a repeated-time structural graph:

1. If `W_{t-1} -> (X_t, Y_t)` exists, the same relationship appears at all adjacent times by stationarity of structural form (unless explicitly time-varying).
2. Therefore, yes: this implies instances like `W_{t-2} -> (X_{t-1}, Y_{t-1})` in the unrolled graph.
3. Lagged causal effects are represented by edges from node `_ {t-k}` to node `_ t`.
4. The estimated coefficient blocks at each lag correspond to weighted adjacency blocks of this lagged DAG.

---

## 4. Data-Generating Process (DGP)

### 4.1 Variables and Dimensions

1. Endogenous vector: `Z_t = [X_t, Y_t]^T` (dimension 2).
2. True confounder vector: `W_t = [W1_t, W2_t]^T` (dimension 2).
3. Time index: `t = 1, ..., n_days`.

### 4.2 Structural Equations

Use:

```text
X_t = c_x
    + a_yx1 * Y_{t-1}
    + a_xx2(t) * X_{t-2}
    + g_x(W_{t-1})
    + eps_x,t

Y_t = c_y
    + a_xy1 * X_{t-1}
    + g_y(W_{t-1})
    + eps_y,t
```

Where:

1. `a_xy1` and `a_yx1` are fixed nonzero constants.
2. `a_xx2(t)` is time-varying and decays to zero.
3. `g_x` and `g_y` are nonlinear nuisance functions (fixed across all experiments).
4. `eps_x,t`, `eps_y,t` are mean-zero noise terms.

### 4.3 Nonlinear Confounder Nuisance (Fixed Form)

Use one fixed nonlinear form for all scenarios:

```text
g_x(W_{t-1}) = b1 * tanh(W1_{t-1}) + b2 * (W2_{t-1}^2)
g_y(W_{t-1}) = c1 * sin(W1_{t-1}) + c2 * relu(W2_{t-1})
```

`relu(u) = max(u, 0)`.

This keeps nuisance consistently nonlinear and favors nonlinear first-stage learners for orthogonalization.

### 4.4 Time-Varying Coefficient Schedule

Define `a_xx2(t)` by piecewise linear decay:

1. Early regime (`t <= t_start`): `a_xx2(t) = a0` (positive nonzero).
2. Transition regime (`t_start < t <= t_end`): linear decrease from `a0` to `0`.
3. Late regime (`t > t_end`): `a_xx2(t) = 0`.

Expected consequence:

1. Early regime behaves as effective lag order 2.
2. Late regime behaves as effective lag order 1.
3. Methods with adaptive lag selection should show a visible shift in `p_optimal`.

### 4.5 Confounder Process

Generate `W_t` from a stable low-order process (e.g., VAR(1) or AR(1)-independent components) with moderate autocorrelation, such that:

1. Confounders have temporal persistence.
2. Endogenous variables and confounders remain stable with finite variance.

Keep confounder process stationary across time in the main benchmark.

---

## 5. Observability Conditions

Run the same DGP under three observation modes:

1. **Full observability**
1. Methods observe `W_obs = [W1, W2]`.
2. **Partial observability**
1. Methods observe `W_obs = [W1]` only; `W2` remains hidden.
3. **Hidden confounding stress**
1. Methods observe no confounders (`W_obs = []`) where method APIs permit.

Interpretation:

1. Full observability is the clean identification target.
2. Partial observability tests robustness under realistic missing confounding.
3. Hidden confounding quantifies inevitable degradation and provides an upper bound on bias risk.

---

## 6. Scenario Matrix

### 6.1 Main Scenarios

1. **S1: Static lag control**
1. Set `a_xx2(t) = a0` for all `t` (no decay).
2. Purpose: verify stable `p*=2` recovery benchmark.
2. **S2: Time-varying lag main scenario**
1. Use piecewise linear decay of `a_xx2(t)` to zero.
2. Purpose: verify transition `p*=2 -> p*=1`.

Each scenario is run under:

1. Full observability
2. Partial observability
3. Hidden confounding stress

### 6.2 Optional Stress Extensions (Secondary)

1. Increase noise variance.
2. Weaken lag-edge magnitudes.
3. Alter transition length (`t_end - t_start`).
4. Modify confounder autocorrelation strength.

---

## 7. Methods to Compare

Use existing pipeline methods:

1. `VAR`
2. `ACLE-VARX`
3. `OR-VARX`
4. `ORACLE-VARX`

Conventions:

1. Keep `p_max` fixed across all methods in each run.
2. Keep learner choice fixed for OR/ORACLE comparisons (e.g., `lgbm` as default).
3. Use identical train/validation/test structure across methods for fairness.

---

## 8. Ground Truth Objects to Log

For each simulated run, store:

1. True lag edge list by time segment.
2. True coefficient path for `a_xx2(t)`.
3. True effective lag-order path:
1. `p_true(t)=2` for early/transition where `a_xx2(t)` materially nonzero.
2. `p_true(t)=1` after decay completion.
4. Observability mode metadata.
5. Random seed and all generation hyperparameters.

---

## 9. Evaluation Targets

### 9.1 DAG/Edge Recovery

From estimated lag coefficients, construct edge sets and evaluate:

1. Precision
2. Recall
3. F1
4. False-positive mass on truly zero edges

Evaluate by:

1. Early regime
2. Transition regime
3. Late regime
4. Overall

### 9.2 Coefficient Recovery

Measure:

1. Signed bias on true nonzero edges.
2. Absolute error (MAE/MSE) on true nonzero edges.
3. Error on true zero edges.
4. Trajectory error for `X_{t-2} -> X_t` over time.

### 9.3 Lag-Order Recovery

From model output `p_optimal`:

1. Exact-match rate vs `p_true(t)`.
2. Segment-wise match rate (early/transition/late).
3. Transition timing metric:
1. Compare estimated first sustained switch to `p=1` against true `t_end`.

### 9.4 Forecast Metrics (Secondary)

Report:

1. RMSE
2. MAE

By regime and overall.

---

## 10. Required Plots

1. `p_optimal` over time (all methods, faceted by observability).
2. True vs estimated trajectory of `X_{t-2} -> X_t`.
3. Per-`p` coefficient evolution grids using existing refit utilities.
4. Representative lagged coefficient heatmaps:
1. One early window
2. One late window
5. Summary metric charts across methods and observability modes.

These plots are designed to make the `p*=2 -> p*=1` shift visually undeniable.

---

## 11. Acceptance Criteria

A run is considered successful if all conditions below hold directionally across seeds:

1. In full observability:
1. OR/ORACLE outperform non-orthogonal baselines on bias and edge recovery.
2. In time-varying scenario:
1. Post-decay period shows dominant `p_optimal=1`.
2. Estimated `X_{t-2} -> X_t` converges toward zero.
3. In partial observability:
1. OR/ORACLE advantages are reduced but still present relative to non-orthogonal baselines.
4. In hidden confounding:
1. All methods degrade; this is documented explicitly as expected.

---

## 12. Reproducibility Protocol

1. Use fixed seed list (minimum 20, recommended 50).
2. Log all config values and seed per run.
3. Save raw coefficients, `p_optimal`, and metrics per seed.
4. Report mean and uncertainty interval (e.g., std or bootstrap CI).
5. Keep generation code and evaluation code deterministic given seed.

---

## 13. Integration with Existing Codebase

Planned additions (for later implementation) should align with current structure:

1. `src/data/synthetic_leadlag.py`
1. Config dataclass for DGP parameters.
2. Synthetic generator returning `(Y, W_true, W_obs, truth)`.
2. `src/evaluation/leadlag_recovery.py`
1. Edge, coefficient, and lag-order metrics.
3. `scripts/run_toy_leadlag_benchmark.py`
1. End-to-end benchmark runner using existing model entry points.
4. Outputs under `results/toy_leadlag/...`.

No implementation is performed in this document; this section specifies future integration targets only.

---

## 14. Recommended Default Hyperparameters (Initial Pass)

These defaults are intentionally conservative and can be tuned later:

1. `n_days`: large enough to cover lookback + validation + clear transition window.
2. `p_max`: at least 4 (recommended 6) to test over-lag robustness.
3. `a_xy1`, `a_yx1`: moderate nonzero values.
4. `a0` for `a_xx2`: smaller than lag-1 effects but clearly detectable pre-decay.
5. Noise scale: moderate SNR (neither trivial nor impossible).
6. Transition window length: long enough to avoid one-day flip artifacts.

---

## 15. Risks and Mitigations

1. Risk: transition too weak to detect.
1. Mitigation: increase `a0` or reduce noise.
2. Risk: nonlinear nuisance too dominant, drowning lag signals.
1. Mitigation: cap nuisance amplitude relative to lag effects.
3. Risk: hidden confounding scenario yields unstable conclusions.
1. Mitigation: present as stress-test only, not main claim.
4. Risk: lag-threshold choice affects edge metrics.
1. Mitigation: report threshold sensitivity curves.

---

## 16. Deliverable from This Plan

A complete synthetic benchmark design that:

1. Uses your exact intended lead-lag structure.
2. Encodes the repeated-time DAG interpretation correctly.
3. Includes time-varying lag decay to test adaptive lag selection.
4. Enforces nonlinear nuisance across all experiments.
5. Evaluates partial/full/hidden confounding observability in one coherent framework.
6. Is directly actionable with the current ORACLE-VARX pipeline.

---

## 17. References Placeholder

References are intentionally deferred and will be appended at the end later, consistent with current project direction.
