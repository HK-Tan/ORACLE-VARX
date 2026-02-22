"""Grid-based Double Machine Learning (DML) implementation for OR-VARX.

This module implements Double Machine Learning for causal inference in time series
using a grid-based memoization strategy for ~250x speedup.

Key Insight:
Consecutive days have 99.9% overlapping data. Instead of retraining for each day:
1. Pre-train models on a fixed grid (every 21 days)
2. Reuse cached models for residual computation
3. Only train new models when reaching a new grid point

This reduces model trainings from ~33M to ~131K.

Algorithm:
1. For each day in test period:
   - Determine which grid folds provide residuals for this day
   - Ensure those folds are trained (cache miss -> train and cache)
   - Compute residuals using cached models
   - Estimate deconfounded coefficients theta via OLS on residuals
   - Generate predictions

References:
    Chernozhukov et al. (2018). "Double/debiased machine learning for treatment
    and structural parameters." The Econometrics Journal.
"""

import gc
import os
import time
import warnings

import torch
import numpy as np
from typing import Tuple, List, Optional, Any, Dict, Union

from src.results import VARXResult
from src.modules.grid_config import GridConfig
from src.models.var_pytorch import select_optimal_p
from src.modules.factory import get_regressor, get_multi_output_regressor
from src.modules.batch_utils import batched_ols

# Type alias for core result tuple
CoreResult = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def estimate_theta(
    Y_residuals: torch.Tensor,
    T_residuals: torch.Tensor,
) -> torch.Tensor:
    """Estimate causal effect theta from residuals using OLS.

    The DML estimator is:
        theta = (T_res' T_res)^{-1} T_res' Y_res

    This gives the deconfounded effect of treatment T on outcome Y.

    Args:
        Y_residuals: Residualized outcomes, shape (n_samples, n_assets)
        T_residuals: Residualized treatments, shape (n_samples, n_treatments)

    Returns:
        theta: Causal effect coefficients, shape (n_treatments, n_assets)
               theta[i, j] = effect of treatment i on asset j

    Raises:
        RuntimeError: If the matrix is singular (insufficient variation in residuals)

    Example:
        >>> Y_res = torch.randn(500, 5)
        >>> T_res = torch.randn(500, 10)
        >>> theta = estimate_theta(Y_res, T_res)
        >>> print(theta.shape)  # (10, 5)
    """
    if Y_residuals.shape[0] != T_residuals.shape[0]:
        raise ValueError("Y_residuals and T_residuals must have same number of samples")

    # OLS: theta = (T'T)^{-1} T'Y
    TtT = torch.mm(T_residuals.T, T_residuals)  # (n_treatments, n_treatments)
    TtY = torch.mm(T_residuals.T, Y_residuals)  # (n_treatments, n_assets)

    try:
        theta = torch.linalg.solve(TtT, TtY)  # (n_treatments, n_assets)
    except torch.linalg.LinAlgError as e:
        raise RuntimeError(
            "Singular matrix in theta estimation. "
            "This may indicate insufficient variation in residualized treatments. "
            "Try a different learner or check for multicollinearity."
        ) from e

    return theta


def compute_se_oracle(
    theta: torch.Tensor,
    Y_residuals: torch.Tensor,
    T_residuals: torch.Tensor,
) -> torch.Tensor:
    """Compute standard errors for ORACLE significance test.

    Uses the OLS formula:
        SE(theta) = sqrt(diag((T'T)^{-1} * sigma^2))
    where sigma^2 is the residual variance:
        sigma^2 = RSS / (n - k)
        RSS = sum((Y_res - T_res * theta)^2)

    Args:
        theta: Estimated coefficients, shape (n_treatments, n_assets)
        Y_residuals: Residualized outcomes, shape (n_samples, n_assets)
        T_residuals: Residualized treatments, shape (n_samples, n_treatments)

    Returns:
        se: Standard errors, shape (n_treatments, n_assets)

    Raises:
        RuntimeError: If computation fails

    Example:
        >>> theta = torch.randn(10, 5)
        >>> Y_res = torch.randn(500, 5)
        >>> T_res = torch.randn(500, 10)
        >>> se = compute_se_oracle(theta, Y_res, T_res)
        >>> print(se.shape)  # (10, 5)
    """
    n_samples = Y_residuals.shape[0]
    n_treatments = theta.shape[0]
    n_assets = theta.shape[1]

    # Compute fitted values: T_res * theta
    Y_fitted = torch.mm(T_residuals, theta)  # (n_samples, n_assets)

    # Compute residuals
    residuals = Y_residuals - Y_fitted  # (n_samples, n_assets)

    # Compute residual variance for each asset
    RSS = torch.sum(residuals ** 2, dim=0)  # (n_assets,)
    df = n_samples - n_treatments  # Degrees of freedom
    if df <= 0:
        raise RuntimeError(f"Insufficient degrees of freedom: n={n_samples}, k={n_treatments}")

    sigma_sq = RSS / df  # (n_assets,)

    # Compute (T'T)^{-1}
    TtT = torch.mm(T_residuals.T, T_residuals)  # (n_treatments, n_treatments)
    try:
        TtT_inv = torch.linalg.inv(TtT)  # (n_treatments, n_treatments)
    except torch.linalg.LinAlgError as e:
        raise RuntimeError("Failed to invert T'T matrix for SE computation") from e

    # Extract diagonal of (T'T)^{-1}
    TtT_inv_diag = torch.diag(TtT_inv)  # (n_treatments,)

    # Compute SE for each coefficient
    # SE[i, j] = sqrt(TtT_inv_diag[i] * sigma_sq[j])
    se = torch.sqrt(TtT_inv_diag.unsqueeze(1) * sigma_sq.unsqueeze(0))  # (n_treatments, n_assets)

    return se


# =============================================================================
# Utilities
# =============================================================================


def get_physical_cpu_count() -> int:
    """Get the number of physical CPU cores (not logical/hyperthreaded).

    Returns:
        Number of physical CPU cores
    """
    try:
        import psutil
        return psutil.cpu_count(logical=False) or 1
    except ImportError:
        # Fallback: assume hyperthreading (divide by 2)
        logical = os.cpu_count() or 2
        return max(1, logical // 2)


def _clear_memory():
    """Force memory cleanup (CPU and GPU if available)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Grid-Based Functions
# =============================================================================


def compute_fold_boundaries(
    grid_idx: int,
    config: GridConfig,
) -> Tuple[int, int, int, int]:
    """Compute train/test boundaries for a grid fold.

    The grid divides time into fixed intervals. Each fold has:
    - A training window of size train_size
    - A test window of size test_size immediately following

    Args:
        grid_idx: Index of the grid fold (0-indexed)
        config: GridConfig with train_size and test_size

    Returns:
        Tuple of (train_start, train_end, test_start, test_end) where:
        - train_start = grid_idx * config.test_size
        - train_end = train_start + config.train_size
        - test_start = train_end
        - test_end = test_start + config.test_size

    Example:
        >>> config = GridConfig(train_size=504, test_size=21)
        >>> compute_fold_boundaries(0, config)
        (0, 504, 504, 525)
        >>> compute_fold_boundaries(1, config)
        (21, 525, 525, 546)
    """
    train_start = grid_idx * config.test_size
    train_end = train_start + config.train_size
    test_start = train_end
    test_end = test_start + config.test_size

    return train_start, train_end, test_start, test_end


def get_active_folds_for_day(
    day_idx: int,
    p: int,
    config: GridConfig,
) -> List[int]:
    """Get grid indices that provide residuals for this day.

    For a given day, we need residuals from row (day_idx - lookback + p) to row (day_idx - 1).
    This function determines which grid folds cover these rows.

    Args:
        day_idx: Current day index (absolute, 0-indexed)
        p: Lag order
        config: GridConfig with lookback, train_size, test_size

    Returns:
        List of grid indices that provide test residuals for this day's lookback window.

    Example:
        >>> config = GridConfig()  # defaults: train_size=504, test_size=21, lookback=766
        >>> # For day 800, with p=1, we need rows from (800 - 766 + 1) = 35 to 799
        >>> folds = get_active_folds_for_day(800, 1, config)
    """
    # Row range in absolute indices (within the lookback window, after lagging)
    # The lookback window for day_idx is [day_idx - config.lookback_orvarx, day_idx)
    # After lagging with p, rows start at relative index p (absolute: day_idx - config.lookback_orvarx + p)
    row_start_abs = day_idx - config.lookback_orvarx + p
    row_end_abs = day_idx - 1  # Last row we need residuals for (exclusive is day_idx)

    # Each fold covers test rows [test_start, test_end)
    # For fold grid_idx: test_start = grid_idx * test_size + train_size
    #                    test_end = test_start + test_size

    # First grid_idx that could cover any row in this range
    # test_end > row_start_abs => grid_idx * test_size + train_size + test_size > row_start_abs
    # => grid_idx > (row_start_abs - train_size - test_size) / test_size
    first_grid_idx = max(0, (row_start_abs - config.train_size) // config.test_size)

    # Last grid_idx needed (covers up to row_end_abs)
    # test_start <= row_end_abs => grid_idx * test_size + train_size <= row_end_abs
    # => grid_idx <= (row_end_abs - train_size) / test_size
    last_grid_idx = (row_end_abs - config.train_size) // config.test_size

    # Ensure non-negative
    last_grid_idx = max(last_grid_idx, first_grid_idx)

    return list(range(first_grid_idx, last_grid_idx + 1))


def _build_lagged_features(
    Y: np.ndarray,
    W: np.ndarray,
    p: int,
    start_idx: int,
    end_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build lagged features for a contiguous range of rows.

    Given raw Y and W arrays, construct outcome (Y_t), treatment (lagged Y),
    and controls (lagged W) for rows in range [start_idx + p, end_idx).

    Args:
        Y: Full Y array, shape (n_days, n_assets)
        W: Full W array, shape (n_days, n_confounders)
        p: Lag order
        start_idx: Start of range (inclusive, before accounting for lags)
        end_idx: End of range (exclusive)

    Returns:
        outcome: Y values, shape (n_rows, n_assets) where n_rows = end_idx - start_idx - p
        treatment: Lagged Y values, shape (n_rows, n_assets * p)
        controls: Lagged W values, shape (n_rows, n_confounders * p)
    """
    n_assets = Y.shape[1]
    n_confounders = W.shape[1]
    n_rows = end_idx - start_idx - p

    # Outcome: Y[start_idx + p : end_idx]
    outcome = Y[start_idx + p:end_idx]

    # Treatment: lagged Y values
    treatment = np.zeros((n_rows, n_assets * p))
    for lag in range(1, p + 1):
        lag_start = start_idx + p - lag
        lag_end = end_idx - lag
        treatment[:, (lag - 1) * n_assets:lag * n_assets] = Y[lag_start:lag_end]

    # Controls: lagged W values
    controls = np.zeros((n_rows, n_confounders * p))
    for lag in range(1, p + 1):
        lag_start = start_idx + p - lag
        lag_end = end_idx - lag
        controls[:, (lag - 1) * n_confounders:lag * n_confounders] = W[lag_start:lag_end]

    return outcome, treatment, controls


def _build_test_features(
    Y: np.ndarray,
    W: np.ndarray,
    p: int,
    test_start: int,
    test_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build test features allowing lags from BEFORE test_start.

    Unlike _build_lagged_features which requires p "burn-in" rows,
    this function predicts ALL rows in [test_start, test_end) by
    sourcing lags from [test_start - p, test_end - 1].

    Args:
        Y: Full Y array, shape (n_days, n_assets)
        W: Full W array, shape (n_days, n_confounders)
        p: Lag order
        test_start: First row to predict (inclusive)
        test_end: Last row to predict (exclusive)

    Returns:
        outcome: Y values, shape (n_rows, n_assets) where n_rows = test_end - test_start
        treatment: Lagged Y values, shape (n_rows, n_assets * p)
        controls: Lagged W values, shape (n_rows, n_confounders * p)
    """
    n_assets = Y.shape[1]
    n_confounders = W.shape[1]
    n_rows = test_end - test_start

    # Outcome: Y[test_start : test_end]
    outcome = Y[test_start:test_end]

    # Treatment: lagged Y values (can go before test_start)
    treatment = np.zeros((n_rows, n_assets * p), dtype=np.float32)
    for lag in range(1, p + 1):
        lag_start = test_start - lag
        lag_end = test_end - lag
        treatment[:, (lag - 1) * n_assets:lag * n_assets] = Y[lag_start:lag_end]

    # Controls: lagged W values (can go before test_start)
    controls = np.zeros((n_rows, n_confounders * p), dtype=np.float32)
    for lag in range(1, p + 1):
        lag_start = test_start - lag
        lag_end = test_end - lag
        controls[:, (lag - 1) * n_confounders:lag * n_confounders] = W[lag_start:lag_end]

    return outcome, treatment, controls


def _build_forecast_controls(W: np.ndarray, day_idx: int, p: int) -> np.ndarray:
    """Build lagged control features for a single forecast day.

    For forecasting day_idx, we need W values at [day_idx-1, ..., day_idx-p].

    Args:
        W: Full W array, shape (n_days, n_confounders)
        day_idx: The day index to forecast (absolute index)
        p: Lag order

    Returns:
        controls: Lagged W values, shape (n_confounders * p,)
    """
    n_confounders = W.shape[1]
    controls = np.zeros(n_confounders * p, dtype=np.float32)
    for lag in range(1, p + 1):
        controls[(lag - 1) * n_confounders:lag * n_confounders] = W[day_idx - lag]
    return controls


def _prepare_all_fold_data(
    Y_np: np.ndarray,
    W_np: np.ndarray,
    all_folds: List[int],
    p_max: int,
    config: GridConfig,
) -> Dict[int, List[Dict[str, Any]]]:
    """Pre-compute fold data (training/test features) for all folds and p values.

    This follows the TabPFN pattern of pre-computing all data before training.
    The returned dict is keyed by p, with each value being a list of fold data dicts.

    Args:
        Y_np: Full Y array, shape (n_days, n_assets)
        W_np: Full W array, shape (n_days, n_confounders)
        all_folds: List of grid fold indices to process
        p_max: Maximum lag order
        config: GridConfig instance

    Returns:
        Dict[p] -> List of fold data dicts with keys:
            'grid_idx', 'X_train', 'Y_train', 'T_train',
            'X_test', 'Y_test', 'T_test',
            'test_start', 'test_end',
            'forecast_controls', 'forecast_day_indices'
    """
    n_days = Y_np.shape[0]

    fold_data: Dict[int, List[Dict[str, Any]]] = {p: [] for p in range(1, p_max + 1)}

    for grid_idx in all_folds:
        train_start, train_end, test_start, test_end = compute_fold_boundaries(grid_idx, config)

        # Clamp to data size
        if train_end > n_days:
            continue
        test_end = min(test_end, n_days)

        for p in range(1, p_max + 1):
            # Build training data
            outcome_train, treatment_train, controls_train = _build_lagged_features(
                Y_np, W_np, p, train_start, train_end
            )

            # Check if we have enough history for lags (need data at test_start - p)
            if test_start - p < 0:
                continue

            # Use _build_test_features that allows lags from before test_start
            outcome_test, treatment_test, controls_test = _build_test_features(
                Y_np, W_np, p, test_start, test_end
            )

            # Collect forecast control features for test days in this fold's window
            forecast_controls = []
            forecast_day_indices = []
            for test_day_idx in range(test_start, test_end):
                fc = _build_forecast_controls(W_np, test_day_idx, p)
                forecast_controls.append(fc)
                forecast_day_indices.append(test_day_idx)

            fold_data[p].append({
                'grid_idx': grid_idx,
                'X_train': controls_train,
                'Y_train': outcome_train,
                'T_train': treatment_train,
                'X_test': controls_test,
                'Y_test': outcome_test,
                'T_test': treatment_test,
                'test_start': test_start,
                'test_end': test_end,
                'forecast_controls': forecast_controls,
                'forecast_day_indices': forecast_day_indices,
            })

    return fold_data


def _process_folds_for_p(
    folds_p: List[Dict[str, Any]],
    Y_np: np.ndarray,
    n_assets: int,
    p: int,
    learner_name: str,
    n_jobs: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Train-predict-discard for all folds at a single p value.

    This is the memory-efficient core: train models, get predictions,
    DELETE models immediately after use.

    Args:
        folds_p: List of fold data dicts for this p
        Y_np: Full Y array (for building treatment features)
        n_assets: Number of assets
        p: Lag order
        learner_name: Name of the sklearn learner
        n_jobs: Number of CPU cores
        verbose: Print progress

    Returns:
        Tuple of:
        - R_Y: Residualized outcomes, shape (n_residual_rows, n_assets)
        - R_T: Residualized treatments, shape (n_residual_rows, n_treatments)
        - first_row: Absolute row index of first residual row
        - forecast_Y_preds: Dict[day_idx -> prediction array]
        - forecast_T_preds: Dict[day_idx -> prediction array]
    """
    if not folds_p:
        raise ValueError(f"No folds provided for p={p}")

    n_treatments = n_assets * p

    # Determine global row range
    first_row = min(f['test_start'] for f in folds_p)
    last_row = max(f['test_end'] for f in folds_p)
    n_rows = last_row - first_row

    # Allocate contiguous arrays for residuals
    R_Y = np.zeros((n_rows, n_assets), dtype=np.float32)
    R_T = np.zeros((n_rows, n_treatments), dtype=np.float32)

    # Storage for forecast predictions
    forecast_Y_preds: Dict[int, np.ndarray] = {}
    forecast_T_preds: Dict[int, np.ndarray] = {}

    for fold_idx, fold in enumerate(folds_p):
        # Train models
        model_y = get_multi_output_regressor(learner_name, n_jobs=n_jobs)
        model_t = get_multi_output_regressor(learner_name, n_jobs=n_jobs)

        model_y.fit(fold['X_train'], fold['Y_train'])
        model_t.fit(fold['X_train'], fold['T_train'])

        # Predict on test data (suppress sklearn warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            Y_pred = model_y.predict(fold['X_test'])
            T_pred = model_t.predict(fold['X_test'])

        # Compute residuals
        local_start = fold['test_start'] - first_row
        local_end = fold['test_end'] - first_row
        R_Y[local_start:local_end] = fold['Y_test'] - Y_pred
        R_T[local_start:local_end] = fold['T_test'] - T_pred

        # Predict for forecast days
        for local_idx, day_idx in enumerate(fold['forecast_day_indices']):
            fc = fold['forecast_controls'][local_idx].reshape(1, -1)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                forecast_Y_preds[day_idx] = model_y.predict(fc).squeeze(0)
                forecast_T_preds[day_idx] = model_t.predict(fc).squeeze(0)

        # DELETE models immediately to free memory
        del model_y, model_t

        if verbose and (fold_idx + 1) % 5 == 0:
            print(f"      p={p}: processed {fold_idx + 1}/{len(folds_p)} folds")

    return R_Y, R_T, first_row, forecast_Y_preds, forecast_T_preds


# =============================================================================
# Vectorized OR-VARX (Batched)
# =============================================================================


def get_all_required_folds(
    n_days: int,
    p_max: int,
    config: GridConfig,
) -> List[int]:
    """Determine all grid folds needed for the entire test period.

    This function computes the union of all folds needed across all test days
    and all lag orders. Used to pre-train all models upfront.

    Args:
        n_days: Total number of days in dataset
        p_max: Maximum lag order
        config: GridConfig with lookback and grid parameters

    Returns:
        Sorted list of unique grid indices needed

    Example:
        >>> config = GridConfig()  # defaults
        >>> folds = get_all_required_folds(n_days=1200, p_max=10, config=config)
        >>> print(len(folds))  # Number of unique folds needed
    """
    lookback = config.lookback_orvarx
    n_test_days = n_days - lookback

    if n_test_days < 1:
        return []

    # Collect all folds needed across all test days and p values
    all_folds = set()

    for day_rel_idx in range(n_test_days):
        day_idx = lookback + day_rel_idx
        for p in range(1, p_max + 1):
            folds = get_active_folds_for_day(day_idx, p, config)
            all_folds.update(folds)

    return sorted(all_folds)


def fit_orvarx_core(
    Y: torch.Tensor,
    W: torch.Tensor,
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
) -> CoreResult:
    """Core OR-VARX DML computation without validation trimming or p-selection.

    This is the shared computation layer used by both fit_orvarx_batched() and
    fit_oraclevarx_batched(). It performs:
    1. Pre-computing fold data (training/test features) for all folds
    2. Train-predict-discard: train models, get predictions, delete models immediately
    3. Batched OLS for all days and all lags

    Memory-Efficient Design:
    - Models are deleted immediately after predictions are computed
    - Fold data is deleted for each p after processing
    - Only residuals and forecast predictions are retained

    The function returns raw data for ALL test days without any trimming or
    p-selection. Callers are responsible for:
    - OR-VARX: p-selection via validation RMSE, then trimming
    - ORACLE-VARX: significance-based p-selection + α-selection, then trimming

    Args:
        Y: Asset returns, shape (n_days, n_assets)
        W: Confounders, shape (n_days, n_confounders)
        p_max: Maximum lag order to consider (default: 10)
        config: GridConfig instance (default: None, uses default GridConfig)
        learner_name: First-stage learner ('xgboost', 'lgbm', 'rf', 'extra_trees')
        n_jobs: Number of CPU cores (-1 for all, 5 recommended)
        verbose: If True, print detailed progress

    Returns:
        Tuple of (forecasts_all, coefficients, standard_errors, actuals) where:
        - forecasts_all: shape (n_total_test_days, n_assets, p_max)
        - coefficients: shape (n_total_test_days, p_max, n_assets, n_assets)
        - standard_errors: shape (n_total_test_days, p_max, n_assets, n_assets)
        - actuals: shape (n_total_test_days, n_assets) - actual Y values for validation

    Notes:
        - Returns ALL test days (n_days - lookback), no trimming
        - Always computes standard errors (needed by both callers)
        - Does NOT perform p-selection
        - Memory efficient: ~1.5 GB peak vs 20-100 GB with model caching
    """
    import time
    t0 = time.time()

    # Use default config if not provided
    if config is None:
        config = GridConfig()

    n_days, n_assets = Y.shape
    n_days_w, n_confounders = W.shape

    # Validation
    if n_days != n_days_w:
        raise ValueError(f"Y and W must have same number of days: {n_days} vs {n_days_w}")
    if p_max < 1:
        raise ValueError(f"p_max must be >= 1, got {p_max}")
    lookback = config.lookback_orvarx
    if lookback < p_max:
        raise ValueError(f"lookback {lookback} must be >= p_max {p_max}")
    if n_days <= lookback:
        raise ValueError(f"Need at least {lookback + 1} days, got {n_days}")

    n_total_test_days = n_days - lookback

    device = Y.device
    dtype = Y.dtype

    # Convert to numpy for sklearn models
    Y_np = Y.cpu().numpy().astype(np.float32)
    W_np = W.cpu().numpy().astype(np.float32)

    n_cpus = get_physical_cpu_count()
    n_jobs_resolved = max(1, n_cpus - 1) if n_jobs == -1 else n_jobs
    print(f"  Core DML: {n_total_test_days} test days, p_max={p_max}, learner={learner_name}")
    print(f"  Grid config: train_size={config.train_size}, test_size={config.test_size}, lookback={lookback}")
    print(f"  Using {n_jobs_resolved} CPU cores ({n_cpus} physical cores available)")

    # =========================================================================
    # Step 1: Determine all required folds and prepare fold data
    # =========================================================================
    print("    Step 1: Preparing fold data...")
    all_folds = get_all_required_folds(n_days, p_max, config)
    print(f"      Total folds needed: {len(all_folds)}")

    fold_data = _prepare_all_fold_data(Y_np, W_np, all_folds, p_max, config)
    print(f"      Built data for {sum(len(v) for v in fold_data.values())} fold-p combinations")

    # Initialize storage for ALL test days (no trimming)
    forecasts_all = torch.zeros(n_total_test_days, n_assets, p_max, device=device, dtype=dtype)
    coefficients = torch.zeros(n_total_test_days, p_max, n_assets, n_assets, device=device, dtype=dtype)
    standard_errors = torch.zeros(n_total_test_days, p_max, n_assets, n_assets, device=device, dtype=dtype)

    # Storage for residuals and forecast predictions (matches TabPFN pattern)
    R_Y_all: Dict[int, np.ndarray] = {}
    R_T_all: Dict[int, np.ndarray] = {}
    first_residual_row: Dict[int, int] = {}
    forecast_Y_preds: Dict[int, Dict[int, np.ndarray]] = {p: {} for p in range(1, p_max + 1)}
    forecast_T_preds: Dict[int, Dict[int, np.ndarray]] = {p: {} for p in range(1, p_max + 1)}

    # =========================================================================
    # Step 2: Train-predict-discard for each p value
    # =========================================================================
    print("    Step 2: Training models and computing predictions...")

    total_train_time = 0.0
    for p in range(1, p_max + 1):
        folds_p = fold_data[p]
        if not folds_p:
            continue

        p_start = time.time()

        # Train-predict-discard for all folds at this p
        R_Y, R_T, first_row, forecast_Y_p, forecast_T_p = _process_folds_for_p(
            folds_p, Y_np, n_assets, p, learner_name, n_jobs_resolved, verbose
        )

        # Store residuals and predictions
        R_Y_all[p] = R_Y
        R_T_all[p] = R_T
        first_residual_row[p] = first_row
        forecast_Y_preds[p] = forecast_Y_p
        forecast_T_preds[p] = forecast_T_p

        p_elapsed = time.time() - p_start
        total_train_time += p_elapsed

        # Delete fold_data for this p to free memory
        del fold_data[p]
        _clear_memory()

        if verbose or p == 1 or p == p_max:
            print(f"      p={p}: {len(folds_p)} folds, {R_Y.shape[0]} residual rows, "
                  f"{len(forecast_Y_p)} forecast days ({p_elapsed:.1f}s)")

    print(f"      Training completed in {total_train_time:.1f}s")

    # =========================================================================
    # Step 3: Batched OLS for all days and all lags
    # =========================================================================
    print("    Step 3: Running batched OLS...")

    ols_window = config.ols_window

    for p in range(1, p_max + 1):
        if p not in R_Y_all:
            continue

        n_treatments = n_assets * p

        # Convert residuals to torch
        R_Y = torch.from_numpy(R_Y_all[p]).to(device=device, dtype=dtype)
        R_T = torch.from_numpy(R_T_all[p]).to(device=device, dtype=dtype)
        first_row = first_residual_row[p]
        n_residual_rows = R_Y.shape[0]

        # Batched OLS: Use unfold() to create sliding windows
        theta_all_batch = None
        offset = first_row + ols_window - lookback
        n_windows = 0

        if n_residual_rows >= ols_window:
            # Create sliding windows using unfold
            T_windows = R_T.unfold(0, ols_window, 1).transpose(1, 2)
            Y_windows = R_Y.unfold(0, ols_window, 1).transpose(1, 2)
            n_windows = T_windows.shape[0]

            try:
                theta_all_batch = batched_ols(T_windows, Y_windows, chunk_size=config.batch_chunk_size)

                # Batched SE computation
                TtT = torch.bmm(T_windows.transpose(1, 2), T_windows)
                TtT_inv = torch.linalg.inv(TtT)
                TtT_inv_diag = torch.diagonal(TtT_inv, dim1=-2, dim2=-1)

                Y_pred = torch.bmm(T_windows, theta_all_batch)
                residuals = Y_windows - Y_pred
                RSS = (residuals ** 2).sum(dim=1)
                df = ols_window - n_treatments
                sigma_sq = RSS / df

                se_all = torch.sqrt(TtT_inv_diag.unsqueeze(-1) * sigma_sq.unsqueeze(1))

                # Store coefficients and SEs
                for i in range(n_windows):
                    day_rel_idx = i + offset
                    if 0 <= day_rel_idx < n_total_test_days:
                        theta_reshaped = theta_all_batch[i].view(p, n_assets, n_assets)
                        coefficients[day_rel_idx, :p, :, :] = theta_reshaped

                        se_reshaped = se_all[i].view(p, n_assets, n_assets)
                        standard_errors[day_rel_idx, :p, :, :] = se_reshaped

            except RuntimeError:
                theta_all_batch = None

        # =====================================================================
        # Forecast generation using pre-computed E[Y|W] and E[T|W]
        # =====================================================================
        for day_rel_idx in range(n_total_test_days):
            day_idx = lookback + day_rel_idx

            # Get theta for this day from batched result
            window_idx = day_rel_idx - offset
            if theta_all_batch is None or window_idx < 0 or window_idx >= n_windows:
                continue

            theta = theta_all_batch[window_idx]  # (n_treatments, n_assets)

            # Check if we have forecast predictions for this day
            if day_idx not in forecast_Y_preds[p] or day_idx not in forecast_T_preds[p]:
                continue

            # Get exact E[Y|W] and E[T|W] from Step 2
            E_Y_given_W = torch.from_numpy(forecast_Y_preds[p][day_idx]).to(device=device, dtype=dtype)
            E_T_given_W = torch.from_numpy(forecast_T_preds[p][day_idx]).to(device=device, dtype=dtype)

            # Build T_actual: actual treatment values [Y_{day-1}, ..., Y_{day-p}]
            indices = list(range(day_idx - 1, day_idx - p - 1, -1))
            T_actual = torch.from_numpy(Y_np[indices, :].reshape(1, n_treatments)).to(device=device, dtype=dtype)

            # Compute T_residual = T_actual - E[T|W]
            T_residual = T_actual - E_T_given_W.reshape(1, n_treatments)

            # Causal effect: T_residual × θ
            causal_effect = torch.mm(T_residual, theta).squeeze(0)

            # Forecast = E[Y|W] + causal_effect
            forecast = E_Y_given_W + causal_effect
            forecasts_all[day_rel_idx, :, p - 1] = forecast

        if verbose:
            print(f"      p={p}: completed")

    print("    Step 3: Complete")

    # Clean up residual storage
    del R_Y_all, R_T_all, forecast_Y_preds, forecast_T_preds
    _clear_memory()

    # Actuals for ALL test days (no trimming)
    actuals = Y[lookback:, :]  # (n_total_test_days, n_assets)

    total_time = time.time() - t0
    print(f"  Core DML complete in {total_time:.1f}s")

    return forecasts_all, coefficients, standard_errors, actuals


def fit_orvarx_batched(
    Y: torch.Tensor,
    W: torch.Tensor,
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    validation_days: int = 21,
    asset_names: Optional[List[str]] = None,
    confounder_names: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
    return_se: bool = False,
    core_results: Optional[CoreResult] = None,
    return_core: bool = False,
) -> Union[VARXResult, Tuple[VARXResult, torch.Tensor], Tuple[VARXResult, CoreResult], Tuple[VARXResult, torch.Tensor, CoreResult]]:
    """Fit OR-VARX model using vectorized/batched operations.

    This function uses fit_orvarx_core() for DML computation, then applies
    p-selection via validation RMSE and output trimming.

    Args:
        Y: Asset returns, shape (n_days, n_assets)
        W: Confounders, shape (n_days, n_confounders)
        p_max: Maximum lag order to consider (default: 10)
        config: GridConfig instance (default: None, uses default GridConfig)
        validation_days: Number of days for p-selection via validation RMSE.
                         Must be >= 1. This determines the output trimming.
        asset_names: Names of assets (default: None, uses A1, A2, ...)
        confounder_names: Names of confounders (default: None, uses W1, W2, ...)
        dates: Date strings for forecast days (default: None, uses indices)
        learner_name: First-stage learner ('xgboost', 'lgbm', 'rf', 'extra_trees')
        n_jobs: Number of CPU cores (-1 for all, 5 recommended)
        verbose: If True, print detailed progress
        return_se: If True, compute and return standard errors (default: False)
        core_results: Pre-computed results from fit_orvarx_core(). If provided,
                      skips the expensive DML computation and reuses these results.
                      Tuple of (forecasts_all, coefficients, standard_errors, actuals).
        return_core: If True, also return the core results tuple for reuse
                     by other methods (e.g., ORACLE-VARX).

    Returns:
        If return_se=False: VARXResult with is_orthogonalized=True
        If return_se=True: Tuple of (VARXResult, standard_errors) where
            standard_errors has shape (n_output_days, p_max, n_assets, n_assets)

    Output Shape:
        n_output_days = (n_days - lookback) - validation_days

    Notes:
        - Uses _fit_orvarx_core() for shared DML computation
        - p-selection is done via validation RMSE (different from ORACLE-VARX)
        - Standard errors use OLS formula: SE[j,k] = sqrt(diag((T'T)^{-1})[j] * σ²[k])
    """
    # Use default config if not provided
    if config is None:
        config = GridConfig()

    n_days, n_assets = Y.shape
    n_days_w, n_confounders = W.shape

    # Validation
    if n_days != n_days_w:
        raise ValueError(f"Y and W must have same number of days: {n_days} vs {n_days_w}")
    if p_max < 1:
        raise ValueError(f"p_max must be >= 1, got {p_max}")
    lookback = config.lookback_orvarx
    if lookback < p_max:
        raise ValueError(f"lookback {lookback} must be >= p_max {p_max}")
    if n_days <= lookback:
        raise ValueError(f"Need at least {lookback + 1} days, got {n_days}")
    if validation_days < 1:
        raise ValueError(f"validation_days must be >= 1, got {validation_days}")

    n_total_test_days = n_days - lookback
    n_output_days = n_total_test_days - validation_days

    if n_output_days < 1:
        raise ValueError(
            f"Insufficient data: need > {lookback + validation_days} days, got {n_days}. "
            f"n_total_test_days ({n_total_test_days}) must be > validation_days ({validation_days})."
        )

    # Set default names
    if asset_names is None:
        asset_names = [f"A{i+1}" for i in range(n_assets)]
    elif len(asset_names) != n_assets:
        raise ValueError(f"Expected {n_assets} asset names, got {len(asset_names)}")

    if confounder_names is None:
        confounder_names = [f"W{i+1}" for i in range(n_confounders)]
    elif len(confounder_names) != n_confounders:
        raise ValueError(f"Expected {n_confounders} confounder names, got {len(confounder_names)}")

    if dates is None:
        dates = [f"D{lookback + validation_days + i}" for i in range(n_output_days)]
    elif len(dates) != n_output_days:
        raise ValueError(f"Expected {n_output_days} dates, got {len(dates)}")

    print(f"Fitting OR-VARX (batched) for {n_total_test_days} test days, p_max={p_max}, learner={learner_name}")

    # =========================================================================
    # Call core function for DML computation (returns ALL test days, no trimming)
    # =========================================================================
    if core_results is not None:
        forecasts_all, coefficients, standard_errors, actuals = core_results
        print("  Using pre-computed core results (skipping DML first stage)")
    else:
        forecasts_all, coefficients, standard_errors, actuals = fit_orvarx_core(
            Y=Y,
            W=W,
            p_max=p_max,
            config=config,
            learner_name=learner_name,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    # Save raw core results for potential return
    forecasts_all_raw = forecasts_all
    coefficients_raw = coefficients
    standard_errors_raw = standard_errors

    # =========================================================================
    # p-selection via validation RMSE
    # =========================================================================
    p_optimal = select_optimal_p(forecasts_all, actuals, validation_days)

    # Extract forecasts at optimal p
    forecasts_all_output = forecasts_all[validation_days:, :, :]
    p_indices = (p_optimal - 1).unsqueeze(1).unsqueeze(2)
    p_indices = p_indices.expand(-1, n_assets, -1)
    forecasts = torch.gather(forecasts_all_output, dim=2, index=p_indices).squeeze(2)

    print(f"  Per-day optimal p: min={p_optimal.min().item()}, max={p_optimal.max().item()}, "
          f"mean={p_optimal.float().mean().item():.2f}")

    # Trim coefficients to output days
    coefficients_output = coefficients[validation_days:, :, :, :]

    # Trim standard errors to output days if computed
    if return_se:
        standard_errors_output = standard_errors[validation_days:, :, :, :]
    else:
        standard_errors_output = None

    # Transpose to match VARXResult expected shape
    forecasts = forecasts.T
    forecasts_all = forecasts_all.transpose(0, 1)[:, validation_days:, :]

    result = VARXResult(
        forecasts=forecasts,
        forecasts_all=forecasts_all,
        p_optimal=p_optimal,
        p_max=p_max,
        coefficients=coefficients_output,
        asset_names=asset_names,
        confounder_names=confounder_names,
        dates=dates,
    )

    # Build return value based on what was requested
    if return_se and return_core:
        return result, standard_errors_output, (forecasts_all_raw, coefficients_raw, standard_errors_raw, actuals)
    elif return_se:
        return result, standard_errors_output
    elif return_core:
        return result, (forecasts_all_raw, coefficients_raw, standard_errors_raw, actuals)
    else:
        return result
