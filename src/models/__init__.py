"""Models module for ORACLE-VARX implementation.

This module provides implementations of:
- VAR: Vector autoregression
- OR-VARX: Orthogonalized VAR with exogenous variables (DML)
- ORACLE-VARX: Adaptive orthogonalization with optimal alpha selection
"""

from src.models.var_pytorch import (
    fit_var,
    batch_var_all_days,
    build_var_design_batch,
    build_pred_features,
    select_optimal_p,
)

from src.models.dml_pytorch import (
    fit_orvarx_batched,
    build_dml_data,
    estimate_theta,
    compute_se_oracle,
    fit_orvarx_single_day,
    # Grid-based functions
    compute_fold_boundaries,
    get_active_folds_for_day,
    ensure_fold_trained,
    compute_residuals,
    # Vectorized helpers
    get_all_required_folds,
    precompute_all_residuals,
)

from src.modules.grid_config import GridConfig

__all__ = [
    # VAR functions
    'fit_var',
    'batch_var_all_days',
    'build_var_design_batch',
    'build_pred_features',
    'select_optimal_p',
    # DML / OR-VARX functions
    'fit_orvarx_batched',
    'build_dml_data',
    'estimate_theta',
    'compute_se_oracle',
    'fit_orvarx_single_day',
    # Grid-based functions
    'compute_fold_boundaries',
    'get_active_folds_for_day',
    'ensure_fold_trained',
    'compute_residuals',
    # Vectorized helpers
    'get_all_required_folds',
    'precompute_all_residuals',
    # Configuration
    'GridConfig',
]
