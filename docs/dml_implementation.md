# Double Machine Learning (DML) Implementation (OR-VARX)

This document explains how orthogonalized regression is implemented in
`/home/hktan/ORACLE-VARX/src/models/dml_pytorch.py` and how the time-series
structure is respected. It is intentionally code-centric and mirrors the exact
logic and data layout used in the implementation.

## 1. Orthogonalized regression: what DML does here

The goal is to estimate lagged return effects while removing confounding by $W$.
The DML structure used in this code can be written as:

$$
Y_t = \Theta \, T_t + g(W_t) + u_t
$$
$$
T_t = f(W_t) + v_t
$$

where:
- $Y_t$ is the vector of asset returns at time $t$
- $T_t$ is the vector of lagged returns (treatment variables)
- $W$ is the vector of lagged confounders $[W_{t-1}, \dots, W_{t-p}]$ (no current $W_t$ to avoid lookahead bias)
- $g$ and $f$ are nuisance functions

DML orthogonalizes the regression by residualizing both $Y$ and $T$ against $W$:

$$
Y_{\text{res}} = Y - \mathbb{E}[Y \mid W]
$$
$$
T_{\text{res}} = T - \mathbb{E}[T \mid W]
$$

Then estimate the causal/orthogonalized coefficients with OLS:

$$
\Theta = (T_{\text{res}}^\top T_{\text{res}})^{-1} T_{\text{res}}^\top Y_{\text{res}}
$$

This is the "orthogonalized regression" step: the second-stage regression uses
residuals that are orthogonal to $W$, which reduces bias from confounding and makes
errors in the first-stage models less harmful.

## 2. Notation and exact data layout (build_dml_data)

Inputs:
- $Y$: $(\texttt{n\_days}, \texttt{n\_assets})$
- $W$: $(\texttt{n\_days}, \texttt{n\_confounders})$
- $p$: lag order
- $\texttt{day\_idx}$: absolute day being forecast (exclusive end of training window)
- $\texttt{lookback}$: window length

`build_dml_data` uses only the window
$[\texttt{day\_idx} - \texttt{lookback}, \texttt{day\_idx})$.

Let:
- $\texttt{start\_idx} = \texttt{day\_idx} - \texttt{lookback}$
- $T = \texttt{lookback} - p$  (number of regression rows)

For each row $r \in \{0, \dots, T-1\}$, the corresponding time index is
$t = \texttt{start\_idx} + p + r$.

Row construction (exactly as in code):

- $\texttt{outcome}[r] = Y_t$
- $\texttt{treatment}[r] = [Y_{t-1}, Y_{t-2}, \dots, Y_{t-p}]$ flattened by lag then asset
- $\texttt{controls}[r] = [W_{t-1}, W_{t-2}, \dots, W_{t-p}]$ flattened by lag then confounder

**Note:** Controls use only lagged confounders (no $W_t$) to avoid lookahead bias.
At prediction time, $W_t$ is not yet observed when forecasting $Y_t$.

Exact column order:

```
treatment:
  [lag1 asset1..assetN, lag2 asset1..assetN, ..., lag p asset1..assetN]

controls:
  [lag1 W1..Wk, lag2 W1..Wk, ..., lag p W1..Wk]
```

This ordering is important for interpreting $\Theta$ and for reshaping coefficients.

## 3. Two-tier time-series structure

The implementation has two nested time-series loops:

```
OUTER LOOP (fit_orvarx)
  day_idx = lookback, lookback+1, ..., n_days-1
  for each day and each p:
    - build DML data from the lookback window
    - run cross-fitting
    - estimate $\Theta$
    - produce a forecast

INNER LOOP (dml_cross_fit)
  RollingWindowSplit over the lookback window rows
  train on past data, predict residuals on the next block
```

This preserves time ordering twice:
- across forecast days (outer loop)
- within each lookback window (inner cross-fit)

## 4. Cross-fitting details (dml_cross_fit)

Inputs to cross-fitting:
- $\texttt{outcome}$: $(\texttt{n\_samples}, \texttt{n\_assets})$
- $\texttt{treatment}$: $(\texttt{n\_samples}, \texttt{n\_treatments})$
- $\texttt{controls}$: $(\texttt{n\_samples}, \texttt{n\_controls})$

Key implementation details:

- Residual arrays are initialized to zeros and are filled only on test indices.
- All models are sklearn-style regressors, so tensors are converted to numpy.
- The code fits:
  - one model_y per asset (W -> Y[:, asset])
  - one model_t per treatment dimension (W -> T[:, treat_idx])
- fit_orvarx_single_day uses the same model factory for model_y and model_t
  (`get_regressor(learner_name, use_gpu)`), so both nuisance models are the
  same learner class.

RollingWindowSplit defaults (for lookback=756, ~3 years):
- $\texttt{n\_splits} = 6$
- $\texttt{train\_size} = 242$ (~1 year for first-stage training)
- $\texttt{test\_size} = 84$
- Total test coverage = $6 \times 84 = 504$ rows (~2 years for second-stage OLS)

Fold $i$ (0-based) is:

```
train indices = [i*test_size, i*test_size + train_size)
test indices  = [train_size + i*test_size, train_size + (i+1)*test_size)
```

With defaults (lookback=756, p=10, available rows=746):
- Fold 1: train [0:242],   test [242:326]
- Fold 2: train [84:326],  test [326:410]
- Fold 3: train [168:410], test [410:494]
- Fold 4: train [252:494], test [494:578]
- Fold 5: train [336:578], test [578:662]
- Fold 6: train [420:662], test [662:746]

Only the test indices are filled with residuals. All other rows remain zero.
In the second-stage OLS, those zero rows contribute nothing, so the estimator
effectively uses only the test-fold rows (504 rows total). If there is not enough
data for all folds, RollingWindowSplit stops early.

### 4.1 First-stage learners (nuisance models)

The nuisance models are created by `get_regressor` in
`/home/hktan/ORACLE-VARX/src/modules/factory.py`. Each model must implement
`fit(X, y)` and `predict(X)` and accepts numpy arrays. The code trains
one model per target dimension (each asset for $Y$, each treatment column for $T$),
so multi-output regressors are not required.

**Supported learners** (all GPU-accelerated, flexible/nonlinear):

| Learner | Library | GPU Training |
|---------|---------|--------------|
| `xgboost` | xgboost | Native (`device='cuda'`) |
| `lgbm` | lightgbm | Native (`device='gpu'`) |
| `rf` | cuML | `cuml.ensemble.RandomForestRegressor` |
| `tabpfn` | tabpfn | Native (`device='cuda'`) |

**Note:** Linear models (ridge, OLS) are not suitable for DML nuisance function
estimation because they cannot capture nonlinear relationships between confounders
and outcomes/treatments.

## 5. Second-stage orthogonalized regression (estimate_theta)

Given residuals, the code computes:

$$
T^\top T = T_{\text{res}}^\top T_{\text{res}}
$$
$$
T^\top Y = T_{\text{res}}^\top Y_{\text{res}}
$$
$$
\Theta = \text{solve}(T^\top T,\; T^\top Y)
$$

Output shape:
- $\Theta$: $(\texttt{n\_assets} \cdot p, \texttt{n\_assets})$

Row $i$ corresponds to a specific (lag, source asset) pair from the treatment
vector. Column $j$ corresponds to the outcome asset.

## 6. Prediction step (fit_orvarx_single_day)

For the forecast day $\texttt{day\_idx}$, the code builds a single-row prediction input:

```
treatment_pred = [Y_{day_idx-1}, ..., Y_{day_idx-p}]
controls_pred  = [W_{day_idx-1}, ..., W_{day_idx-p}]
```

**Note:** Controls use only lagged confounders (no $W_{\text{day\_idx}}$) since the
current confounder value is not yet observed when forecasting.

It then residualizes the treatment with the last fold's models:

$$
T_{\text{pred,res}} = T_{\text{pred}} - \widehat{\mathbb{E}}[T \mid W]_{\text{last fold}}
$$

### Prediction modes

The forecast formula depends on the `include_confounder_baseline` parameter:

**Default (`include_confounder_baseline=False`):**
$$
\widehat{Y} = T_{\text{pred,res}} \, \Theta
$$
This gives only the causal effect of lagged returns, useful for lead-lag analysis.

**With baseline (`include_confounder_baseline=True`):**
$$
\widehat{Y} = \widehat{\mathbb{E}}[Y \mid W]_{\text{last fold}} + T_{\text{pred,res}} \, \Theta
$$
This adds the confounder baseline for better prediction accuracy, using `model_y_last`.

## 7. Concrete toy example (small numbers, exact ordering)

Assume:
- $\texttt{n\_assets} = 2$ (assets A and B)
- $\texttt{n\_confounders} = 1$ ($W$)
- $p = 2$
- $\texttt{lookback} = 6$
- $\texttt{day\_idx} = 6$
- window covers $t = 0..5$
- $T = \texttt{lookback} - p = 4$ rows ($t = 2..5$)

Rows created by `build_dml_data`:

```
t = 2
  outcome   = [A_2, B_2]
  treatment = [A_1, B_1, A_0, B_0]
  controls  = [W_1, W_0]          # Only lagged, no W_2

t = 3
  outcome   = [A_3, B_3]
  treatment = [A_2, B_2, A_1, B_1]
  controls  = [W_2, W_1]          # Only lagged, no W_3

t = 4
  outcome   = [A_4, B_4]
  treatment = [A_3, B_3, A_2, B_2]
  controls  = [W_3, W_2]          # Only lagged, no W_4

t = 5
  outcome   = [A_5, B_5]
  treatment = [A_4, B_4, A_3, B_3]
  controls  = [W_4, W_3]          # Only lagged, no W_5
```

Shapes:
- $\texttt{outcome}$: $(4, 2)$
- $\texttt{treatment}$: $(4, 4)$
- $\texttt{controls}$: $(4, 2)$ = $n\_confounders \times p$

$\Theta$ has shape $(4, 2)$. Reshaping with `theta.view(p, n_assets, n_assets)` gives
index order $[lag, \texttt{source\_asset}, \texttt{target\_asset}]$, for example:

```
theta_reshaped[0, 0, 0] = effect of A_{t-1} on A_t
theta_reshaped[0, 1, 0] = effect of B_{t-1} on A_t
theta_reshaped[1, 0, 1] = effect of A_{t-2} on B_t
theta_reshaped[1, 1, 1] = effect of B_{t-2} on B_t
```

Prediction at $\texttt{day\_idx} = 6$ uses:

```
treatment_pred = [A_5, B_5, A_4, B_4]
controls_pred  = [W_5, W_4]       # Only lagged, no W_6

# Default (include_confounder_baseline=False):
Y_hat = (treatment_pred - E[T | W]) @ Theta

# With baseline (include_confounder_baseline=True):
Y_hat = E[Y | W] + (treatment_pred - E[T | W]) @ Theta
```

## 8. Key functions and responsibilities

- `build_dml_data`: Build outcome, treatment, controls for a single day.
- `dml_cross_fit`: Rolling cross-fitting and residual computation.
- `estimate_theta`: Orthogonalized OLS on residuals.
- `compute_se_oracle`: OLS standard errors used by ORACLE-VARX.
- `fit_orvarx_single_day`: End-to-end DML for one day and one p.
- `fit_orvarx`: Loop over days and p, then select p by validation RMSE.

## 9. Related files

- `/home/hktan/ORACLE-VARX/src/models/dml_pytorch.py`
- `/home/hktan/ORACLE-VARX/src/modules/rolling_split.py`
- `/home/hktan/ORACLE-VARX/src/modules/factory.py`
- `/home/hktan/ORACLE-VARX/src/results.py`
- `/home/hktan/ORACLE-VARX/tests/test_orvarx.py`
- `/home/hktan/ORACLE-VARX/scripts/example_dml_usage.py`

## References

1. Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).
   "Double/debiased machine learning for treatment and structural parameters."
   The Econometrics Journal, 21(1), C1-C68.

2. Athey, S., & Imbens, G. W. (2019).
   "Machine learning methods that economists should know about."
   Annual Review of Economics, 11, 685-725.
