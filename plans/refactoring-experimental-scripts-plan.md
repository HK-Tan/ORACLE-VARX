# Experiment Execution Plan: Full Method Comparison Suite

## Context

Run a comprehensive comparison of all VAR-family methods across 3 confounder configurations and all 4 tree learners (+ TabPFN). OR-VARX and ORACLE-VARX share the same first-stage DML residuals (`fit_orvarx_core()` in `dml_pytorch.py`), so bundling them amortizes the expensive computation. VARX/ACLE-VARX are fast OLS and are bundled too.

---

## 1. Architecture: Combined Runs with Amortized First Stage

### How the methods share computation

Both `fit_orvarx_batched()` and `fit_oraclevarx_batched()` call the same `fit_orvarx_core()` for DML orthogonalization. They differ only in second-stage p-selection:
- **OR-VARX**: p-selection via validation RMSE (cheap)
- **ORACLE-VARX**: significance-based p-selection + alpha-grid (cheap)

First stage ~15-20 min, second stage seconds. Running them separately wastes ~50% of compute.

### Combined script: `scripts/run_combined_experiment.py`

Each invocation is parameterized by `(confounder_config, learner)` and runs up to 4 methods:

```bash
python scripts/run_combined_experiment.py \
    --confounders vix --learner lgbm --no-show --verbose
```

**What happens inside:**

1. **Load data** — load ETF returns + confounders (once, shared by all methods)
2. **VARX** (OLS, ~2 min) — confounders as endogenous log returns. Skipped if results already exist for this confounder config (identical across learners)
3. **ACLE-VARX** (OLS, ~3 min) — same skip logic as VARX
4. **`fit_orvarx_core()`** (~15-20 min) — expensive DML first stage, run once
5. **OR-VARX second stage** (seconds) — validation RMSE p-selection on core results
6. **ORACLE-VARX second stage** (seconds) — significance + alpha-grid on core results
7. **Evaluate + save** all results with plots (same format as existing individual scripts)

Also supports `--no-confounders` mode for VAR + ACLE-VAR baseline.

### Confounder conversion: log returns everywhere (consistency)

All confounders are converted to log returns via `ln(C_t / C_{t-1})` in both paths:
- **VARX/ACLE-VARX path**: confounders as endogenous log returns (via `load_opcl_with_confounders()`)
- **DML path**: confounders as exogenous log returns (via `prepare_tensors()` which calls `load_confounder(log_returns=True)`)

---

## 2. Experiment Matrix: 16 Runs -> 35 Unique Model Outputs

### Run table

| Run | Config | Learner | Methods executed | Compute | Est. Time |
|-----|--------|---------|-----------------|---------|-----------|
| 0 | none | OLS | VAR + ACLE-VAR | CPU | 25 min |
| 1 | vix | lgbm | VARX + ACLE-VARX + OR-VARX + ORACLE-VARX | CPU | 25 min |
| 2 | vix | xgboost | OR-VARX + ORACLE-VARX (VARX/ACLE-VARX skipped) | CPU | 22 min |
| 3 | vix | rf | OR-VARX + ORACLE-VARX | CPU | 27 min |
| 4 | vix | extra_trees | OR-VARX + ORACLE-VARX | CPU | 22 min |
| 5 | vix | TabPFN | ORACLE-VARX TabPFN | **GPU** | 35 min |
| 6 | macro5 | lgbm | VARX + ACLE-VARX + OR-VARX + ORACLE-VARX | CPU | 30 min |
| 7 | macro5 | xgboost | OR-VARX + ORACLE-VARX | CPU | 28 min |
| 8 | macro5 | rf | OR-VARX + ORACLE-VARX | CPU | 32 min |
| 9 | macro5 | extra_trees | OR-VARX + ORACLE-VARX | CPU | 28 min |
| 10 | macro5 | TabPFN | ORACLE-VARX TabPFN | **GPU** | 40 min |
| 11 | all10 | lgbm | VARX + ACLE-VARX + OR-VARX + ORACLE-VARX | CPU | 35 min |
| 12 | all10 | xgboost | OR-VARX + ORACLE-VARX | CPU | 32 min |
| 13 | all10 | rf | OR-VARX + ORACLE-VARX | CPU | 37 min |
| 14 | all10 | extra_trees | OR-VARX + ORACLE-VARX | CPU | 32 min |
| 15 | all10 | TabPFN | ORACLE-VARX TabPFN | **GPU** | 45 min |

### Confounder configs

| Config | Variables | Data starts | First DML forecast ~ | Eval years |
|--------|-----------|-------------|---------------------|------------|
| vix | VIX | ~2000 | ~2004 | ~17 yrs |
| macro5 | VIX, DFF, T5YIE, DCOILWTICO, USEPUINDXD | ~2000 | ~2004 | ~17 yrs |
| all10 | All 10 confounders | ~2000 | ~2004 | ~17 yrs |

> **Note**: Leading NaN confounders (from late-starting series like GVZCLS, mid-2008) are backfilled with 0.0 (log-return of 0 = "no change"), preserving all asset data from ~2000 onward.

---

## 3. Code Changes Made

### 3a. `src/data/constants.py`
- Added 5 missing confounders to `CONFOUNDER_FILES`
- Added `CONFOUNDER_PRESETS` dict with vix, macro5, all10 presets
- Annotated unused legacy constants

### 3b. `src/data/loader.py`
- Added `log_returns=True` parameter to `load_confounder()`
- Added `load_opcl_with_confounders()` function
- Fixed `load_opcl_with_vix()` to pass `log_returns=False` (avoid double conversion)

### 3c. `src/models/dml_pytorch.py`
- Renamed `_fit_orvarx_core()` -> `fit_orvarx_core()` (public)
- Added `core_results` and `return_core` parameters to `fit_orvarx_batched()`

### 3d. `src/models/oracle_var.py`
- Added `core_results` parameter to `fit_oraclevarx_batched()`

### 3e. Experiment scripts
- Added `--confounders` CLI arg to `run_orvarx_experiment.py`, `run_oraclevarx_experiment.py`, `run_oraclevarx_tabpfn_experiment.py`

### 3f. New scripts
- Created `scripts/run_combined_experiment.py` — combined runner with amortized first stage
- Created `scripts/run_all_experiments.py` — tmux orchestrator

---

## 4. Infrastructure

See `plans/run-experiment-guide.md` for detailed setup and execution instructions.
