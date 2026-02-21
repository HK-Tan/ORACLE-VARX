# Plan: Remove 5D Coefficient Tensors + Add Per-p Refit Visualization

## Status: Implemented

## Context

The `coefficients_all` tensor (shape `n_days x n_alphas x p_max x n_assets x n_assets`) in `ORACLEVARXResult` and `ACLEVARXResult` was both expensive (~11M floats per experiment) and misleading (only contained VAR(p_max) coefficients, not genuine VAR(p*) coefficients).

## Changes Made

### Removals
- `coefficients_all` field from `ORACLEVARXResult` and `ACLEVARXResult`
- `coefficients` property from both classes
- `get_leadlag_matrix()` from both classes
- `get_coefficient_heatmap_matrix()` from both classes
- `coefficients_all` construction blocks from `acle_var.py`, `oracle_var.py`, `oracle_var_tabpfn.py`
- `plot_coefficient_heatmap_evolution()` from `plotting.py`
- Plot 4 (mean coefficient heatmap) and Plot 5 (lag decomposition) from all experiment scripts

### Additions
- `src/models/coefficient_refit.py`: Per-p coefficient refitting module
  - `PerPCoefficients` dataclass
  - `refit_var_coefficients_for_day()` for ACLE-VAR/ACLE-VARX
  - `refit_dml_coefficients_for_day()` for ORACLE-VARX (tree-based)
  - `refit_dml_coefficients_for_day_tabpfn()` for ORACLE-VARX (TabPFN)
  - `get_target_days()` utility
- `plot_coefficient_evolution_per_p()` in `plotting.py`: p* x p* grid visualization
- Per-p coefficient evolution plots for 2 target days in each experiment script

### Updates
- `VARXResult.get_coefficient_heatmap_matrix()`: now requires `day_idx` (no silent averaging)
- `src/evaluation/__init__.py`: updated exports

## Files Modified
- `src/results.py`
- `src/models/acle_var.py`
- `src/models/oracle_var.py`
- `src/models/oracle_var_tabpfn.py`
- `src/models/coefficient_refit.py` (new)
- `src/evaluation/plotting.py`
- `src/evaluation/__init__.py`
- `scripts/run_aclevar_experiment.py`
- `scripts/run_aclevarx_experiment.py`
- `scripts/run_oraclevarx_experiment.py`
- `scripts/run_oraclevarx_tabpfn_experiment.py`
