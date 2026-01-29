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

import os
import time
import warnings

import torch
import numpy as np
from typing import Tuple, List, Optional, Any, Dict, Union

from src.results import VARXResult
from src.modules.grid_config import GridConfig
from src.models.var_pytorch import select_optimal_p
from src.modules.model_cache import ModelCache, FoldModels
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


def ensure_fold_trained(
    cache: ModelCache,
    grid_idx: int,
    p: int,
    Y: np.ndarray,
    W: np.ndarray,
    config: GridConfig,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
) -> Tuple[FoldModels, bool, float]:
    """Ensure a fold is trained and cached; train if not present.

    Checks the cache for the specified fold and lag order. If not found,
    trains new models and adds them to the cache.

    Args:
        cache: ModelCache instance
        grid_idx: Grid fold index
        p: Lag order
        Y: Full Y array, shape (n_days, n_assets)
        W: Full W array, shape (n_days, n_confounders)
        config: GridConfig instance
        learner_name: Name of the learner to use
        n_jobs: Number of CPU cores (-1 for all, 5 recommended)
        verbose: If True, print training details for cache misses

    Returns:
        Tuple of (FoldModels, was_cache_hit, train_time)
        - FoldModels: trained models and boundaries
        - was_cache_hit: True if model was retrieved from cache
        - train_time: Training time in seconds (0.0 if cache hit)
    """
    # Check cache first
    cached = cache.get_fold(grid_idx, p)
    if cached is not None:
        return cached, True, 0.0

    # Compute boundaries
    train_start, train_end, test_start, test_end = compute_fold_boundaries(grid_idx, config)

    # Validate boundaries against data size
    n_days = Y.shape[0]
    if train_end > n_days:
        raise ValueError(
            f"Fold {grid_idx} train_end {train_end} exceeds data size {n_days}"
        )

    # Build lagged features for training data
    # We need p extra rows at the start for lagging
    outcome_train, treatment_train, controls_train = _build_lagged_features(
        Y, W, p, train_start, train_end
    )

    # Train models using MultiOutputRegressor
    train_start_time = time.time()
    model_y = get_multi_output_regressor(learner_name, n_jobs=n_jobs)
    model_t = get_multi_output_regressor(learner_name, n_jobs=n_jobs)

    model_y.fit(controls_train, outcome_train)
    model_t.fit(controls_train, treatment_train)
    train_elapsed = time.time() - train_start_time

    if verbose:
        print(f"    Training fold={grid_idx}, p={p}... {train_elapsed:.2f}s")

    # Create FoldModels
    fold = FoldModels(
        model_y=model_y,
        model_t=model_t,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        p=p,
    )

    # Add to cache
    cache.add_fold(grid_idx, p, fold)

    return fold, False, train_elapsed


def compute_residuals(
    cache: ModelCache,
    day_idx: int,
    p: int,
    Y: np.ndarray,
    W: np.ndarray,
    config: GridConfig,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute residuals for a day using cached fold models.

    For each active fold, predicts on the test rows and computes residuals.
    Only returns residuals from test windows (not the training portion).

    Note: The first train_size rows of the lookback window have no test residuals
    because they're only used for training the first fold. This is by design -
    DML cross-fitting only uses test fold residuals for the second-stage OLS.

    Args:
        cache: ModelCache instance
        day_idx: Current day index
        p: Lag order
        Y: Full Y array, shape (n_days, n_assets)
        W: Full W array, shape (n_days, n_confounders)
        config: GridConfig instance
        learner_name: Name of the learner
        n_jobs: Number of CPU cores (-1 for all, 5 recommended)
        verbose: If True, print training details for cache misses

    Returns:
        Tuple of (Y_residuals, T_residuals, stats) where:
        - Y_residuals, T_residuals: numpy arrays of residuals
        - stats: dict with 'cache_hits', 'cache_misses', 'total_train_time'
    """
    # Get active folds for this day
    active_folds = get_active_folds_for_day(day_idx, p, config)

    # Collect residuals from all active folds
    Y_residuals_list = []
    T_residuals_list = []

    # Determine the lookback window boundaries
    lookback_start = day_idx - config.lookback_orvarx
    row_end_abs = day_idx

    # Track cache stats
    cache_hits = 0
    cache_misses = 0
    total_train_time = 0.0

    for grid_idx in active_folds:
        # Ensure fold is trained
        fold, was_hit, train_time = ensure_fold_trained(
            cache, grid_idx, p, Y, W, config, learner_name, n_jobs, verbose
        )

        if was_hit:
            cache_hits += 1
        else:
            cache_misses += 1
            total_train_time += train_time

        # Determine fold's test window rows
        # Test window covers absolute rows [test_start + p, test_end)
        # (we need p rows for lagging, so actual outcome rows start at test_start + p)
        fold_test_row_start = fold.test_start + p
        fold_test_row_end = min(fold.test_end, Y.shape[0])  # Clamp to data size

        # Only use rows within our lookback window
        overlap_start = max(lookback_start + p, fold_test_row_start)
        overlap_end = min(row_end_abs, fold_test_row_end)

        if overlap_start >= overlap_end:
            continue  # No overlap with lookback window

        # Build lagged features for this overlap
        outcome_test, treatment_test, controls_test = _build_lagged_features(
            Y, W, p, overlap_start - p, overlap_end
        )

        # Predict (suppress sklearn feature name mismatch warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            Y_pred = fold.model_y.predict(controls_test)
            T_pred = fold.model_t.predict(controls_test)

        # Compute residuals
        Y_res = outcome_test - Y_pred
        T_res = treatment_test - T_pred

        Y_residuals_list.append(Y_res)
        T_residuals_list.append(T_res)

    if not Y_residuals_list:
        raise RuntimeError(
            f"No residuals computed for day {day_idx}, p={p}. "
            f"active_folds={active_folds}"
        )

    # Concatenate all residuals
    Y_residuals = np.vstack(Y_residuals_list)
    T_residuals = np.vstack(T_residuals_list)

    stats = {
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'total_train_time': total_train_time,
    }

    return Y_residuals, T_residuals, stats


def fit_orvarx_single_day(
    Y: torch.Tensor,
    W: torch.Tensor,
    p: int,
    day_idx: int,
    cache: ModelCache,
    config: GridConfig,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Fit OR-VARX for a single day and lag order using cached models.

    Performs the DML pipeline:
    1. Build DML data (outcome, treatment, controls)
    2. Compute residuals using cached models
    3. Estimate deconfounded coefficients theta
    4. Compute standard errors
    5. Generate prediction

    Args:
        Y: Asset returns, shape (n_days, n_assets)
        W: Confounders, shape (n_days, n_confounders)
        p: Lag order
        day_idx: Current day index
        cache: ModelCache instance for caching models
        config: GridConfig instance
        learner_name: First-stage learner (default: 'xgboost')
        n_jobs: Number of CPU cores (-1 for all, 5 recommended)
        verbose: If True, print training details for cache misses

    Returns:
        forecast: Prediction for day_idx, shape (n_assets,)
        theta: Deconfounded coefficients, shape (n_treatments, n_assets)
        se: Standard errors, shape (n_treatments, n_assets)
        stats: Dict with cache stats ('cache_hits', 'cache_misses', 'total_train_time')
    """
    device = Y.device
    dtype = Y.dtype
    n_assets = Y.shape[1]
    n_confounders = W.shape[1]

    # Convert full data to numpy for sklearn models
    Y_np = Y.cpu().numpy()
    W_np = W.cpu().numpy()

    # Compute residuals using cached models
    Y_residuals_np, T_residuals_np, stats = compute_residuals(
        cache, day_idx, p, Y_np, W_np, config, learner_name, n_jobs, verbose
    )

    # Convert back to torch
    Y_residuals = torch.from_numpy(Y_residuals_np).to(device=device, dtype=dtype)
    T_residuals = torch.from_numpy(T_residuals_np).to(device=device, dtype=dtype)

    # Estimate deconfounded coefficients
    theta = estimate_theta(Y_residuals, T_residuals)

    # Compute standard errors
    se = compute_se_oracle(theta, Y_residuals, T_residuals)

    # Generate prediction for day_idx
    # Build treatment features: [Y_{day_idx-1}, ..., Y_{day_idx-p}]
    indices = torch.arange(day_idx - 1, day_idx - p - 1, -1, device=device)
    treatment_pred = Y[indices, :].reshape(1, n_assets * p)  # (1, n_assets * p)

    # Build control features for prediction (lagged only)
    lagged_indices = torch.arange(day_idx - 1, day_idx - p - 1, -1, device=device)
    controls_pred = W[lagged_indices, :].reshape(1, n_confounders * p)  # (1, n_confounders * p)

    # Get the most recent fold for this day to residualize prediction
    # Use the last active fold (most recent)
    active_folds = get_active_folds_for_day(day_idx, p, config)
    last_fold_idx = max(active_folds)
    last_fold = cache.get_fold(last_fold_idx, p)

    # Residualize treatment using the last fold's model
    controls_pred_np = controls_pred.cpu().numpy()
    treatment_pred_np = treatment_pred.cpu().numpy()

    # Predict (suppress sklearn feature name mismatch warnings)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        T_hat = last_fold.model_t.predict(controls_pred_np)
    T_pred_residual = treatment_pred_np - T_hat
    T_pred_residual_torch = torch.from_numpy(T_pred_residual).to(device=device, dtype=dtype)

    # Compute causal effect: T_residual * theta
    forecast = torch.mm(T_pred_residual_torch, theta).squeeze(0)  # (n_assets,)

    return forecast, theta, se, stats


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


def precompute_all_residuals(
    cache: ModelCache,
    p: int,
    Y: np.ndarray,
    W: np.ndarray,
    config: GridConfig,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Pre-compute residuals for all test windows at a given lag order.

    For each fold that has been trained, compute residuals for its ENTIRE
    test window. Store residuals indexed by absolute row position.

    Args:
        cache: ModelCache with pre-trained models
        p: Lag order
        Y: Full Y array, shape (n_days, n_assets)
        W: Full W array, shape (n_days, n_confounders)
        config: GridConfig instance
        learner_name: Learner name (for training if needed)
        n_jobs: Number of CPU cores
        verbose: Print progress

    Returns:
        Tuple of (R_Y, R_T, first_row, n_rows) where:
        - R_Y: Residualized outcomes, shape (n_residual_rows, n_assets)
        - R_T: Residualized treatments, shape (n_residual_rows, n_treatments)
        - first_row: Absolute row index of first residual row
        - n_rows: Number of residual rows

    Notes:
        - Only computes residuals from test windows (DML cross-fitting)
        - Residuals are stored contiguously; use first_row to map to absolute indices
        - The first train_size rows have no residuals (training portion)
    """
    n_days = Y.shape[0]
    n_assets = Y.shape[1]
    n_confounders = W.shape[1]
    n_treatments = n_assets * p

    # Get all trained folds for this p
    all_grid_indices = cache.get_all_trained_folds(p)

    if not all_grid_indices:
        raise RuntimeError(f"No trained folds found for p={p}")

    # Collect residuals from all folds
    residual_chunks = []
    first_row_global = None

    for grid_idx in sorted(all_grid_indices):
        fold = cache.get_fold(grid_idx, p)
        if fold is None:
            continue

        # Compute fold's test window rows (accounting for lags)
        # Outcome rows start at test_start + p (need p rows for lags)
        fold_test_row_start = fold.test_start + p
        fold_test_row_end = min(fold.test_end, n_days)

        if fold_test_row_start >= fold_test_row_end:
            continue  # No valid test rows

        # Track the first row across all folds
        if first_row_global is None:
            first_row_global = fold_test_row_start
        else:
            first_row_global = min(first_row_global, fold_test_row_start)

        # Build lagged features for entire test window
        outcome_test, treatment_test, controls_test = _build_lagged_features(
            Y, W, p, fold.test_start, fold_test_row_end
        )

        # Predict (suppress sklearn warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            Y_pred = fold.model_y.predict(controls_test)
            T_pred = fold.model_t.predict(controls_test)

        # Compute residuals
        Y_res = outcome_test - Y_pred
        T_res = treatment_test - T_pred

        # Store with row indices
        residual_chunks.append({
            'start_row': fold_test_row_start,
            'end_row': fold_test_row_end,
            'Y_res': Y_res,
            'T_res': T_res,
        })

    if not residual_chunks:
        raise RuntimeError(f"No residuals computed for p={p}")

    # Determine global row range
    first_row = min(chunk['start_row'] for chunk in residual_chunks)
    last_row = max(chunk['end_row'] for chunk in residual_chunks)
    n_rows = last_row - first_row

    # Allocate contiguous arrays
    R_Y = np.zeros((n_rows, n_assets), dtype=np.float32)
    R_T = np.zeros((n_rows, n_treatments), dtype=np.float32)

    # Fill in residuals (some positions may be overwritten if folds overlap)
    for chunk in residual_chunks:
        local_start = chunk['start_row'] - first_row
        local_end = chunk['end_row'] - first_row
        R_Y[local_start:local_end] = chunk['Y_res']
        R_T[local_start:local_end] = chunk['T_res']

    return R_Y, R_T, first_row, n_rows


def _fit_orvarx_core(
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
    1. Pre-training all required grid folds
    2. Pre-computing residuals for all test windows
    3. Batched OLS for all days and all lags

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

    n_total_test_days = n_days - lookback

    device = Y.device
    dtype = Y.dtype

    # Convert to numpy for sklearn models
    Y_np = Y.cpu().numpy()
    W_np = W.cpu().numpy()

    # Create model cache
    cache = ModelCache(n_assets=n_assets, n_confounders=n_confounders, p_max=p_max)

    n_cpus = get_physical_cpu_count()
    n_jobs_display = n_cpus if n_jobs == -1 else n_jobs
    print(f"  Core DML: {n_total_test_days} test days, p_max={p_max}, learner={learner_name}")
    print(f"  Grid config: train_size={config.train_size}, test_size={config.test_size}, lookback={lookback}")
    print(f"  Using {n_jobs_display} CPU cores ({n_cpus} physical cores available)")

    # =========================================================================
    # Step 1: Determine and pre-train all required folds
    # =========================================================================
    print("    Step 1: Pre-training all required folds...")
    all_folds = get_all_required_folds(n_days, p_max, config)
    print(f"      Total folds needed: {len(all_folds)}")

    total_train_time = 0.0
    for grid_idx in all_folds:
        for p in range(1, p_max + 1):
            _, was_hit, train_time = ensure_fold_trained(
                cache, grid_idx, p, Y_np, W_np, config, learner_name, n_jobs, verbose
            )
            total_train_time += train_time

    print(f"      Training completed in {total_train_time:.2f}s")

    # Initialize storage for ALL test days (no trimming)
    forecasts_all = torch.zeros(n_total_test_days, n_assets, p_max, device=device, dtype=dtype)
    coefficients = torch.zeros(n_total_test_days, p_max, n_assets, n_assets, device=device, dtype=dtype)
    standard_errors = torch.zeros(n_total_test_days, p_max, n_assets, n_assets, device=device, dtype=dtype)

    # =========================================================================
    # Step 2 & 3: For each p, pre-compute residuals and run batched OLS
    # =========================================================================
    print("    Step 2-3: Pre-computing residuals and running batched OLS...")

    ols_window = config.ols_window

    for p in range(1, p_max + 1):
        n_treatments = n_assets * p

        # Pre-compute all residuals for this p
        R_Y_np, R_T_np, first_row, n_residual_rows = precompute_all_residuals(
            cache, p, Y_np, W_np, config, learner_name, n_jobs, verbose
        )

        # Convert to torch
        R_Y = torch.from_numpy(R_Y_np).to(device=device, dtype=dtype)
        R_T = torch.from_numpy(R_T_np).to(device=device, dtype=dtype)

        # =====================================================================
        # Batched OLS: Use unfold() to create sliding windows, then batched_ols()
        # =====================================================================
        theta_all = None
        offset = first_row + ols_window - lookback
        n_windows = 0

        if n_residual_rows >= ols_window:
            # Create sliding windows using unfold
            T_windows = R_T.unfold(0, ols_window, 1).transpose(1, 2)  # (n_windows, ols_window, n_treatments)
            Y_windows = R_Y.unfold(0, ols_window, 1).transpose(1, 2)  # (n_windows, ols_window, n_assets)
            n_windows = T_windows.shape[0]

            try:
                theta_all = batched_ols(T_windows, Y_windows, chunk_size=config.batch_chunk_size)
                # theta_all shape: (n_windows, n_treatments, n_assets)

                # Batched SE computation (always compute - needed by both models)
                # Compute (T'T)^{-1} diagonal elements for all windows
                TtT = torch.bmm(T_windows.transpose(1, 2), T_windows)  # (n_windows, n_treatments, n_treatments)
                TtT_inv = torch.linalg.inv(TtT)  # (n_windows, n_treatments, n_treatments)
                TtT_inv_diag = torch.diagonal(TtT_inv, dim1=-2, dim2=-1)  # (n_windows, n_treatments)

                # Compute residuals and sigma^2 for each window
                Y_pred = torch.bmm(T_windows, theta_all)  # (n_windows, ols_window, n_assets)
                residuals = Y_windows - Y_pred  # (n_windows, ols_window, n_assets)
                RSS = (residuals ** 2).sum(dim=1)  # (n_windows, n_assets)
                df = ols_window - n_treatments  # Degrees of freedom
                sigma_sq = RSS / df  # (n_windows, n_assets)

                # SE[i,j,k] = sqrt(TtT_inv_diag[i,j] * sigma_sq[i,k])
                se_all = torch.sqrt(TtT_inv_diag.unsqueeze(-1) * sigma_sq.unsqueeze(1))
                # se_all shape: (n_windows, n_treatments, n_assets)

                # Store coefficients and SEs for all valid days at once
                for i in range(n_windows):
                    day_rel_idx = i + offset
                    if 0 <= day_rel_idx < n_total_test_days:
                        theta_reshaped = theta_all[i].view(p, n_assets, n_assets)
                        coefficients[day_rel_idx, :p, :, :] = theta_reshaped

                        se_reshaped = se_all[i].view(p, n_assets, n_assets)
                        standard_errors[day_rel_idx, :p, :, :] = se_reshaped

            except RuntimeError:
                # Singular matrix encountered - this shouldn't happen with proper residualization
                theta_all = None

        # =====================================================================
        # Forecast generation loop (must stay sequential - model_t.predict dependency)
        # =====================================================================
        for day_rel_idx in range(n_total_test_days):
            day_idx = lookback + day_rel_idx

            # Get theta for this day from batched result
            window_idx = day_rel_idx - offset
            if theta_all is None or window_idx < 0 or window_idx >= n_windows:
                # No valid theta for this day (early days or batched OLS failed)
                continue

            theta = theta_all[window_idx]  # (n_treatments, n_assets)

            # Generate forecast
            indices = torch.arange(day_idx - 1, day_idx - p - 1, -1, device=device)
            treatment_pred = Y[indices, :].reshape(1, n_treatments)

            # Build control features for prediction (lagged only)
            lagged_indices = torch.arange(day_idx - 1, day_idx - p - 1, -1, device=device)
            controls_pred = W[lagged_indices, :].reshape(1, n_confounders * p)

            # Get the most recent fold for residualizing prediction
            active_folds = get_active_folds_for_day(day_idx, p, config)
            last_fold_idx = max(active_folds)
            last_fold = cache.get_fold(last_fold_idx, p)

            if last_fold is None:
                continue

            # Residualize treatment using the last fold's model
            controls_pred_np = controls_pred.cpu().numpy()
            treatment_pred_np = treatment_pred.cpu().numpy()

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                T_hat = last_fold.model_t.predict(controls_pred_np)

            T_pred_residual = treatment_pred_np - T_hat
            T_pred_residual_torch = torch.from_numpy(T_pred_residual).to(device=device, dtype=dtype)

            # Compute forecast: T_residual * theta
            forecast = torch.mm(T_pred_residual_torch, theta).squeeze(0)
            forecasts_all[day_rel_idx, :, p - 1] = forecast

        if verbose:
            print(f"      p={p}: completed")

    print("    Step 2-3: Complete")

    # Actuals for ALL test days (no trimming)
    actuals = Y[lookback:, :]  # (n_total_test_days, n_assets)

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
) -> Union[VARXResult, Tuple[VARXResult, torch.Tensor]]:
    """Fit OR-VARX model using vectorized/batched operations.

    This function uses _fit_orvarx_core() for DML computation, then applies
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
    forecasts_all, coefficients, standard_errors, actuals = _fit_orvarx_core(
        Y=Y,
        W=W,
        p_max=p_max,
        config=config,
        learner_name=learner_name,
        n_jobs=n_jobs,
        verbose=verbose,
    )

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

    # Return tuple with SEs if requested, otherwise just the result
    if return_se:
        return result, standard_errors_output
    else:
        return result
