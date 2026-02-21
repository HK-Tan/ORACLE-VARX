"""Per-p coefficient refitting for visualization.

Provides functions to re-estimate VAR/DML coefficients at each lag order
p = 1, ..., p* separately for specific target days. This produces genuine
per-p estimates (not truncated VAR(p_max) coefficients).

Usage:
    from src.models.coefficient_refit import (
        refit_var_coefficients_for_day,
        refit_dml_coefficients_for_day,
        refit_dml_coefficients_for_day_tabpfn,
        get_target_days,
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from src.modules.batch_utils import batched_ols
from src.models.var_pytorch import build_var_design_batch, build_pred_features


@dataclass
class PerPCoefficients:
    """Per-p coefficient estimates for a single day."""
    day_idx: int                          # Index in result.dates
    date: str                             # Date string
    p_star: int                           # Optimal lag order
    coefficients: Dict[int, torch.Tensor] = field(default_factory=dict)
    # p -> (p, n_assets, n_assets) from VAR(p) fit
    asset_names: List[str] = field(default_factory=list)


def get_target_days(result) -> List[Tuple[str, int]]:
    """Returns [(label, day_idx), ...] for coefficient visualization.

    Always returns exactly 2 entries:
    - max_p: day with the largest p* (most interesting lag structure)
    - last_day: last day in the result (most recent)
    """
    max_p_day = result.p_optimal.argmax().item()
    last_day = len(result.dates) - 1
    return [("max_p", max_p_day), ("last_day", last_day)]


def refit_var_coefficients_for_day(
    Y: torch.Tensor,
    day_idx: int,
    p_star: int,
    lookback: int,
    asset_names: List[str],
    date: str,
) -> PerPCoefficients:
    """Refit VAR(1), ..., VAR(p*) separately for a single day.

    For ACLE-VAR / ACLE-VARX models. Each p gets its own genuine VAR(p)
    coefficient estimates via OLS.

    Args:
        Y: Full return series, shape (n_total_days, n_assets)
        day_idx: Absolute day index (index into Y, not result.dates)
        p_star: Optimal lag order for this day
        lookback: Training window size (e.g., config.lookback_var)
        asset_names: List of asset names
        date: Date string for this day

    Returns:
        PerPCoefficients with coefficients[p] = (p, n_assets, n_assets)
    """
    result = PerPCoefficients(
        day_idx=day_idx,
        date=date,
        p_star=p_star,
        asset_names=asset_names,
    )

    # Extract training window: Y[day_idx - lookback : day_idx]
    window = Y[day_idx - lookback:day_idx].unsqueeze(0)  # (1, lookback, n_assets)
    n_assets = Y.shape[1]

    for p in range(1, p_star + 1):
        # Build VAR(p) design matrix with batch=1
        X_batch, Y_batch = build_var_design_batch(window, p)
        # X_batch: (1, T-p, 1+n_assets*p), Y_batch: (1, T-p, n_assets)

        # Solve OLS: beta = (X'X)^{-1} X'Y
        beta = batched_ols(X_batch, Y_batch)  # (1, 1+n_assets*p, n_assets)

        # Extract A matrices (skip intercept)
        beta_no_intercept = beta[:, 1:, :]  # (1, n_assets*p, n_assets)
        A_matrices = beta_no_intercept.view(1, p, n_assets, n_assets)
        A_matrices = A_matrices.transpose(-2, -1)  # convention: [k, i, j] = j->i at lag k+1

        result.coefficients[p] = A_matrices.squeeze(0)  # (p, n_assets, n_assets)

    return result


def refit_dml_coefficients_for_day(
    Y: torch.Tensor,
    W: torch.Tensor,
    day_idx: int,
    p_star: int,
    lookback: int,
    asset_names: List[str],
    date: str,
    config,
    learner_name: str = 'lgbm',
    n_jobs: int = -1,
) -> PerPCoefficients:
    """Refit DML coefficients at each p=1,...,p* for a single day.

    For ORACLE-VARX models (tree-based learners). Uses simplified 1-model
    DML: train on tree_train window, predict on ols_region, compute
    residuals, run OLS.

    Args:
        Y: Full return series, shape (n_total_days, n_assets)
        W: Full confounder series, shape (n_total_days, n_confounders)
        day_idx: Absolute day index (index into Y)
        p_star: Optimal lag order for this day
        lookback: Training window size (config.lookback_orvarx)
        asset_names: List of asset names
        date: Date string
        config: GridConfig instance (for train_size, ols_window)
        learner_name: sklearn learner name ('lgbm', 'xgboost', etc.)
        n_jobs: CPU cores for learner

    Returns:
        PerPCoefficients with coefficients[p] = (p, n_assets, n_assets)
    """
    from src.models.dml_pytorch import _build_lagged_features, _build_test_features, estimate_theta
    from src.modules.factory import get_multi_output_regressor

    result = PerPCoefficients(
        day_idx=day_idx,
        date=date,
        p_star=p_star,
        asset_names=asset_names,
    )

    Y_np = Y.cpu().numpy().astype(np.float32)
    W_np = W.cpu().numpy().astype(np.float32)
    n_assets = Y.shape[1]

    # Window boundaries: the full lookback window for this day
    window_start = day_idx - lookback
    tree_train_end = window_start + config.train_size
    ols_start = tree_train_end
    ols_end = day_idx  # ols_region ends at day_idx (exclusive)

    for p in range(1, p_star + 1):
        n_treatments = n_assets * p

        # Build training data (tree_train region)
        outcome_train, treatment_train, controls_train = _build_lagged_features(
            Y_np, W_np, p, window_start, tree_train_end
        )

        # Build OLS region data (allowing lags from before ols_start)
        outcome_ols, treatment_ols, controls_ols = _build_test_features(
            Y_np, W_np, p, ols_start, ols_end
        )

        # Train Y-model and T-model on tree_train
        Y_model = get_multi_output_regressor(learner_name, n_jobs=n_jobs)
        T_model = get_multi_output_regressor(learner_name, n_jobs=n_jobs)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y_model.fit(controls_train, outcome_train)
            T_model.fit(controls_train, treatment_train)

        # Predict on OLS region
        Y_pred = Y_model.predict(controls_ols)
        T_pred = T_model.predict(controls_ols)

        # Compute residuals
        R_Y = torch.from_numpy((outcome_ols - Y_pred).astype(np.float32)).to(Y.device)
        R_T = torch.from_numpy((treatment_ols - T_pred).astype(np.float32)).to(Y.device)

        # OLS on residuals: theta = (R_T' R_T)^{-1} R_T' R_Y
        theta = estimate_theta(R_Y, R_T)  # (n_treatments, n_assets)

        # Reshape to (p, n_assets, n_assets) - same convention as VAR
        A_matrices = theta.view(p, n_assets, n_assets)
        result.coefficients[p] = A_matrices

        # Free models immediately
        del Y_model, T_model

    return result


def refit_dml_coefficients_for_day_tabpfn(
    Y: torch.Tensor,
    W: torch.Tensor,
    day_idx: int,
    p_star: int,
    lookback: int,
    asset_names: List[str],
    date: str,
    config,
    device: str = 'cuda',
) -> PerPCoefficients:
    """Refit DML coefficients using TabPFN for a single day.

    Same structure as refit_dml_coefficients_for_day but uses TabPFN
    models on GPU instead of sklearn regressors.

    Args:
        Y: Full return series, shape (n_total_days, n_assets)
        W: Full confounder series, shape (n_total_days, n_confounders)
        day_idx: Absolute day index
        p_star: Optimal lag order for this day
        lookback: Training window size (config.lookback_orvarx)
        asset_names: List of asset names
        date: Date string
        config: GridConfig instance
        device: Device for TabPFN ('cuda' or 'cpu')

    Returns:
        PerPCoefficients with coefficients[p] = (p, n_assets, n_assets)
    """
    from src.models.dml_pytorch import _build_lagged_features, _build_test_features, estimate_theta
    from src.modules.factory import get_multi_output_regressor

    result = PerPCoefficients(
        day_idx=day_idx,
        date=date,
        p_star=p_star,
        asset_names=asset_names,
    )

    Y_np = Y.cpu().numpy().astype(np.float32)
    W_np = W.cpu().numpy().astype(np.float32)
    n_assets = Y.shape[1]

    window_start = day_idx - lookback
    tree_train_end = window_start + config.train_size
    ols_start = tree_train_end
    ols_end = day_idx

    for p in range(1, p_star + 1):
        n_treatments = n_assets * p

        outcome_train, treatment_train, controls_train = _build_lagged_features(
            Y_np, W_np, p, window_start, tree_train_end
        )

        outcome_ols, treatment_ols, controls_ols = _build_test_features(
            Y_np, W_np, p, ols_start, ols_end
        )

        # Use TabPFN via factory
        Y_model = get_multi_output_regressor('tabpfn')
        T_model = get_multi_output_regressor('tabpfn')

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y_model.fit(controls_train, outcome_train)
            T_model.fit(controls_train, treatment_train)

        Y_pred = Y_model.predict(controls_ols)
        T_pred = T_model.predict(controls_ols)

        R_Y = torch.from_numpy((outcome_ols - Y_pred).astype(np.float32)).to(Y.device)
        R_T = torch.from_numpy((treatment_ols - T_pred).astype(np.float32)).to(Y.device)

        theta = estimate_theta(R_Y, R_T)
        A_matrices = theta.view(p, n_assets, n_assets)
        result.coefficients[p] = A_matrices

        del Y_model, T_model

    return result
