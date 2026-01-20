"""Batched VAR implementation using PyTorch for GPU acceleration.

This module implements vector autoregression (VAR) models using batched matrix
operations for efficient computation on GPUs. The key insight is to compute all
(day, p) combinations in a single batch using torch.bmm for batched matrix
multiplication.

Coefficient Convention:
    VAR(p) model: Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + ε_t

    Coefficient matrix A_k has shape (n_assets, n_assets) where:
        A_k[i, j] = effect of asset j at lag k on asset i

    Example: A_1[2, 3] = effect of asset 3 at lag 1 on asset 2

    The returned coefficients tensor has shape (n_days, p_max, n_assets, n_assets)
    where coefficients[d, k-1, i, j] gives the effect of asset j at lag k on
    asset i for day d.

Key functions:
- build_var_design_batch: Constructs design matrices for VAR(p) estimation
- build_pred_features: Builds prediction features from lagged observations
- batch_var_all_days: Main batched VAR estimation across all days and lags
- select_optimal_p: Selects optimal lag order based on validation RMSE
- fit_var: High-level API for VAR model fitting
"""

import torch
from typing import Tuple, List, Optional

from src.modules.grid_config import GridConfig
from src.modules.batch_utils import batched_ols
from src.results import VARXResult


def build_var_design_batch(
    windows: torch.Tensor,
    p: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build design matrices for VAR(p) from batched windows.

    For VAR(p): Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + ε

    The design matrix X contains:
    - Intercept (column of 1s)
    - Lagged values: Y_{t-1}, Y_{t-2}, ..., Y_{t-p} flattened

    Example for p=2, n_assets=3:
        X row at time t: [1, y1_{t-1}, y2_{t-1}, y3_{t-1}, y1_{t-2}, y2_{t-2}, y3_{t-2}]

    Args:
        windows: Batched time windows, shape (batch, window_len, n_assets)
        p: Lag order

    Returns:
        X_batch: Design matrices, shape (batch, T-p, 1 + n_assets*p)
        Y_batch: Response matrices, shape (batch, T-p, n_assets)

    Raises:
        ValueError: If p is invalid or windows too short
    """
    batch_size, window_len, n_assets = windows.shape

    if p < 1:
        raise ValueError(f"Lag order p must be >= 1, got {p}")
    if window_len <= p:
        raise ValueError(f"Window length {window_len} must be > p={p}")

    T = window_len - p  # Number of observations for regression
    n_features = 1 + n_assets * p  # Intercept + p lags of n_assets

    device = windows.device
    dtype = windows.dtype

    # Initialize design matrix and response
    X_batch = torch.zeros(batch_size, T, n_features, device=device, dtype=dtype)
    Y_batch = torch.zeros(batch_size, T, n_assets, device=device, dtype=dtype)

    # Fill intercept column
    X_batch[:, :, 0] = 1.0

    # Fill response: Y_t for t = p, p+1, ..., window_len-1
    Y_batch = windows[:, p:, :]  # Shape: (batch, T, n_assets)

    # Fill lagged values: Y_{t-1}, Y_{t-2}, ..., Y_{t-p} (vectorized)
    # For each lag k in 1..p, we want Y_{t-k} for t in [p, window_len-1]
    # This corresponds to windows[:, p-k:window_len-k, :] for each k
    #
    # Build all lags at once using advanced indexing:
    # lag_indices[k] gives the time indices for lag k+1: [p-(k+1), p-(k+1)+1, ..., window_len-(k+1)-1]
    # which is equivalent to [p-k-1, ..., window_len-k-2] for k in 0..p-1
    #
    # For lag=1: indices are [p-1, p, ..., window_len-2] (T values)
    # For lag=2: indices are [p-2, p-1, ..., window_len-3] (T values)
    # ...
    # For lag=p: indices are [0, 1, ..., T-1] (T values)

    # Create time indices for all lags: shape (p, T)
    # For lag k (1-indexed), start_idx = p - k, and we take T consecutive values
    base_indices = torch.arange(T, device=device)  # [0, 1, ..., T-1]
    lag_offsets = torch.arange(p - 1, -1, -1, device=device)  # [p-1, p-2, ..., 0] for lags 1, 2, ..., p

    # time_indices[k, t] = (p - (k+1)) + t = p - k - 1 + t
    # For lag=1 (k=0): p - 1 + t -> indices [p-1, p, ..., p-1+T-1] = [p-1, ..., window_len-2]
    time_indices = lag_offsets.unsqueeze(1) + base_indices.unsqueeze(0)  # Shape: (p, T)

    # Gather lagged values for all lags at once
    # windows shape: (batch, window_len, n_assets)
    # We want to gather along dim=1 using time_indices
    # Expand time_indices to (batch, p, T) and gather
    time_indices_expanded = time_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, p, T)

    # We need to gather windows[:, time_indices[k, t], :] for each batch, lag k, time t
    # Use advanced indexing: for each (b, k, t), get windows[b, time_indices[k, t], :]
    # Reshape for gather: expand time_indices to (batch, p, T, n_assets)
    time_indices_for_gather = time_indices_expanded.unsqueeze(-1).expand(-1, -1, -1, n_assets)  # (batch, p, T, n_assets)

    # Expand windows for gather: (batch, 1, window_len, n_assets) -> gather along dim 2
    windows_expanded = windows.unsqueeze(1).expand(-1, p, -1, -1)  # (batch, p, window_len, n_assets)

    # Gather: result[b, k, t, a] = windows[b, time_indices[k, t], a]
    lagged_all = torch.gather(windows_expanded, 2, time_indices_for_gather)  # Shape: (batch, p, T, n_assets)

    # Reshape lagged_all to (batch, T, p * n_assets) to fill X_batch columns 1:1+p*n_assets
    # Current shape: (batch, p, T, n_assets)
    # Permute to (batch, T, p, n_assets) then reshape to (batch, T, p * n_assets)
    lagged_all = lagged_all.permute(0, 2, 1, 3)  # Shape: (batch, T, p, n_assets)
    lagged_all = lagged_all.reshape(batch_size, T, p * n_assets)  # Shape: (batch, T, p * n_assets)

    # Fill the lagged columns in X_batch (columns 1 to 1 + p*n_assets)
    X_batch[:, :, 1:1 + p * n_assets] = lagged_all

    return X_batch, Y_batch


def build_pred_features(
    windows: torch.Tensor,
    p: int
) -> torch.Tensor:
    """Build prediction features from the last p observations.

    For predicting Y_{T+1}, we need [1, Y_T, Y_{T-1}, ..., Y_{T-p+1}]

    Args:
        windows: Batched time windows, shape (batch, window_len, n_assets)
        p: Lag order

    Returns:
        X_pred: Prediction features, shape (batch, 1, 1 + n_assets*p)

    Raises:
        ValueError: If p is invalid or windows too short
    """
    batch_size, window_len, n_assets = windows.shape

    if p < 1:
        raise ValueError(f"Lag order p must be >= 1, got {p}")
    if window_len < p:
        raise ValueError(f"Window length {window_len} must be >= p={p}")

    n_features = 1 + n_assets * p
    device = windows.device
    dtype = windows.dtype

    # Initialize prediction features
    X_pred = torch.zeros(batch_size, 1, n_features, device=device, dtype=dtype)

    # Fill intercept
    X_pred[:, 0, 0] = 1.0

    # Fill lagged values: Y_T, Y_{T-1}, ..., Y_{T-p+1} (vectorized)
    # These are the last p observations in windows in reverse order
    # For lag=1: index = window_len - 1 (Y_T)
    # For lag=2: index = window_len - 2 (Y_{T-1})
    # ...
    # For lag=p: index = window_len - p (Y_{T-p+1})
    #
    # Gather the last p time steps and flatten them
    # time_indices = [window_len - 1, window_len - 2, ..., window_len - p]
    time_indices = torch.arange(window_len - 1, window_len - p - 1, -1, device=device)  # Shape: (p,)

    # Gather lagged values: windows[:, time_indices, :]
    # Result shape: (batch, p, n_assets)
    lagged_values = windows[:, time_indices, :]  # Shape: (batch, p, n_assets)

    # Reshape to (batch, p * n_assets) and place in X_pred
    lagged_values = lagged_values.reshape(batch_size, p * n_assets)  # Shape: (batch, p * n_assets)

    # Fill columns 1 to 1 + p*n_assets
    X_pred[:, 0, 1:1 + p * n_assets] = lagged_values

    return X_pred


def batch_var_all_days(
    Y: torch.Tensor,
    p_max: int,
    lookback: int,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched VAR estimation for all days and all lag orders.

    This is the KEY FUNCTION for GPU acceleration. It computes:
    - VAR(1), VAR(2), ..., VAR(p_max) for all test days in ONE batch

    Algorithm:
    1. Stack sliding windows for all test days
    2. For each p in 1..p_max:
       - Build batched design matrices
       - Solve batched OLS: β = (X'X)^{-1} X'Y using torch.linalg.solve
       - Extract coefficient matrices A_1, ..., A_p
       - Generate predictions

    Args:
        Y: Full return series, shape (n_days, n_assets)
        p_max: Maximum lag order to consider
        lookback: Training window size
        chunk_size: If set, process batched OLS in chunks to manage GPU memory.
                   Default None means no chunking.

    Returns:
        forecasts: Predictions, shape (n_test_days, n_assets, p_max)
                  forecasts[d, :, p-1] = prediction for day d using VAR(p)
        coefficients: VAR coefficients, shape (n_test_days, p_max, n_assets, n_assets)
                     coefficients[d, k, i, j] = effect of asset j at lag k+1 on asset i

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If OLS solve fails (singular matrix)
    """
    n_days, n_assets = Y.shape

    if lookback < p_max:
        raise ValueError(f"Lookback {lookback} must be >= p_max={p_max}")
    if n_days <= lookback:
        raise ValueError(f"Need at least {lookback + 1} days, got {n_days}")

    n_test_days = n_days - lookback
    device = Y.device
    dtype = Y.dtype

    forecasts = torch.zeros(n_test_days, n_assets, p_max, device=device, dtype=dtype)
    coefficients = torch.zeros(n_test_days, p_max, n_assets, n_assets, device=device, dtype=dtype)

    # Build sliding windows for all test days using unfold (vectorized)
    # Y.unfold(0, lookback, 1) creates sliding windows along dimension 0
    # Y shape: (n_days, n_assets)
    # unfold(dim=0, size=lookback, step=1) returns shape: (n_windows, n_assets, lookback)
    # where n_windows = n_days - lookback + 1
    # We need shape (n_windows, lookback, n_assets), so transpose last two dims
    # windows[i] = Y[i:i+lookback], so windows[0] = Y[0:lookback] (for predicting day lookback)
    # We need n_test_days windows: windows[0:n_test_days]
    windows = Y.unfold(0, lookback, 1)  # Shape: (n_days - lookback + 1, n_assets, lookback)
    windows = windows.transpose(1, 2)  # Shape: (n_days - lookback + 1, lookback, n_assets)
    windows = windows[:n_test_days]  # Shape: (n_test_days, lookback, n_assets)

    for p in range(1, p_max + 1):

        # Build design matrices
        X_batch, Y_batch = build_var_design_batch(windows, p)  # (batch, T-p, n_features), (batch, T-p, n_assets)

        # Batched OLS: β = (X'X)^{-1} X'Y
        try:
            beta = batched_ols(X_batch, Y_batch, chunk_size=chunk_size)  # (batch, n_features, n_assets)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed VAR({p}) estimation: {e}. "
                f"Consider reducing p_max or increasing lookback."
            ) from e

        # Extract coefficient matrices (excluding intercept)
        # beta shape: (batch, 1 + n_assets*p, n_assets)
        # beta[:, :, i] gives coefficients for predicting asset i
        # We want coefficients[d, k, i, j] = effect of asset j at lag k+1 on asset i
        beta_no_intercept = beta[:, 1:, :]  # (batch, n_assets*p, n_assets)

        # Reshape to (batch, p, n_assets, n_assets) and transpose last two dims
        # First reshape: (batch, p, n_assets, n_assets) where [:, k, j, i] = effect of j on i
        A_matrices = beta_no_intercept.view(n_test_days, p, n_assets, n_assets)
        # Transpose to get [:, k, i, j] = effect of j on i
        A_matrices = A_matrices.transpose(-2, -1)

        # Store coefficients
        coefficients[:, :p, :, :] = A_matrices

        # Build prediction features and generate forecasts
        X_pred = build_pred_features(windows, p)  # (batch, 1, n_features)
        forecasts[:, :, p - 1] = torch.bmm(X_pred, beta).squeeze(1)

    return forecasts, coefficients


def select_optimal_p(
    forecasts_all: torch.Tensor,
    actuals: torch.Tensor,
    validation_days: int,
) -> torch.Tensor:
    """Select optimal p for each output day using fixed-size rolling validation.

    For each output day d (where d >= validation_days), selects optimal lag order
    based on RMSE over the trailing validation window [d - validation_days, d).

    Args:
        forecasts_all: All predictions, shape (n_test_days, n_assets, p_max)
        actuals: Actual returns, shape (n_test_days, n_assets)
        validation_days: Size of rolling validation window

    Returns:
        p_optimal: Optimal lag for each output day (1-indexed), shape (n_test_days - validation_days,)

    Raises:
        ValueError: If n_test_days <= validation_days
    """
    n_test_days, n_assets, p_max = forecasts_all.shape
    device = forecasts_all.device
    dtype = forecasts_all.dtype

    if n_test_days <= validation_days:
        raise ValueError(
            f"Need n_test_days > validation_days, got {n_test_days} <= {validation_days}"
        )

    # Compute squared errors and average over assets
    squared_errors = (forecasts_all - actuals.unsqueeze(2)) ** 2
    mse_per_day_p = squared_errors.mean(dim=1)  # (n_test_days, p_max)

    # Cumsum for efficient rolling computation
    cumsum = mse_per_day_p.cumsum(dim=0)  # (n_test_days, p_max)
    cumsum_padded = torch.cat([
        torch.zeros(1, p_max, device=device, dtype=dtype),
        cumsum
    ], dim=0)  # (n_test_days + 1, p_max)

    # Only compute for days >= validation_days
    n_output_days = n_test_days - validation_days
    d_indices = torch.arange(validation_days, n_test_days, device=device)
    val_start_indices = d_indices - validation_days

    # Gather cumsum values for rolling window
    cumsum_at_d = cumsum_padded[d_indices]  # (n_output_days, p_max)
    cumsum_at_start = cumsum_padded[val_start_indices]  # (n_output_days, p_max)

    # Rolling mean (window size is always validation_days)
    rolling_mse = (cumsum_at_d - cumsum_at_start) / validation_days  # (n_output_days, p_max)

    # Select optimal p for each output day (1-indexed)
    p_optimal = torch.argmin(rolling_mse, dim=1) + 1  # (n_output_days,)

    return p_optimal


def fit_var(
    Y: torch.Tensor,
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    validation_days: int = 20,
    asset_names: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
) -> VARXResult:
    """Fit plain VAR model with optimal p selection.

    This is the main entry point for VAR fitting. It performs:
    1. Batched VAR estimation for all test days and lag orders 1..p_max
    2. Optimal lag selection based on validation RMSE
    3. Extraction of forecasts at optimal lag

    The validation period uses the first `validation_days` of the test period
    to select the optimal lag order, which is then applied to all test days.

    Args:
        Y: Asset returns, shape (n_days, n_assets)
        p_max: Maximum lag order to consider (default: 10)
        config: GridConfig for lookback settings (default: None, uses GridConfig())
                The lookback is derived from config.lookback_var (514 by default)
        validation_days: Number of days to use for validation (default: 20)
        asset_names: Names of assets (default: None, uses A1, A2, ...)
        dates: Date strings for forecast days (default: None, uses indices)

    Returns:
        VARXResult with forecasts, coefficients, and metadata

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If VAR estimation fails

    Example:
        >>> Y = torch.randn(600, 5)  # 600 days, 5 assets
        >>> result = fit_var(Y, p_max=10)  # Uses GridConfig() default (lookback_var=514)
        >>> print(result.forecasts.shape)  # (5, 66) = (n_assets, n_output_days)
        >>> print(result.p_optimal)  # Optimal lag for each day
    """
    # Use default GridConfig if not provided
    if config is None:
        config = GridConfig()
    lookback = config.lookback_var
    n_days, n_assets = Y.shape

    # Validate inputs
    if p_max < 1:
        raise ValueError(f"p_max must be >= 1, got {p_max}")
    if lookback < p_max:
        raise ValueError(f"lookback {lookback} must be >= p_max {p_max}")
    if n_days <= lookback:
        raise ValueError(f"Need at least {lookback + 1} days, got {n_days}")
    if validation_days < 1:
        raise ValueError(f"validation_days must be >= 1, got {validation_days}")

    n_test_days = n_days - lookback
    if n_test_days <= validation_days:
        raise ValueError(
            f"Insufficient data: need > {lookback + validation_days} days, got {n_days}. "
            f"n_test_days ({n_test_days}) must be > validation_days ({validation_days})."
        )
    n_output_days = n_test_days - validation_days

    # Set default asset names
    if asset_names is None:
        asset_names = [f"A{i+1}" for i in range(n_assets)]
    elif len(asset_names) != n_assets:
        raise ValueError(f"Expected {n_assets} asset names, got {len(asset_names)}")

    # Set default dates
    if dates is None:
        dates = [f"D{lookback + i}" for i in range(n_test_days)]
    elif len(dates) != n_test_days:
        raise ValueError(f"Expected {n_test_days} dates, got {len(dates)}")

    # Step 1: Batched VAR estimation
    forecasts_all, coefficients = batch_var_all_days(Y, p_max, lookback)
    # forecasts_all: (n_test_days, n_assets, p_max)
    # coefficients: (n_test_days, p_max, n_assets, n_assets)

    # Step 2: Extract actual returns for test period
    actuals = Y[lookback:, :]  # (n_test_days, n_assets)

    # Step 3: Select optimal p based on validation RMSE
    p_optimal = select_optimal_p(forecasts_all, actuals, validation_days)

    # Step 4: Extract forecasts at optimal p for each day
    # Slice forecasts_all to output days only
    forecasts_all_output = forecasts_all[validation_days:, :, :]  # (n_output_days, n_assets, p_max)

    # Extract forecast at each day's optimal p using gather
    p_indices = (p_optimal - 1).unsqueeze(1).unsqueeze(2)  # (n_output_days, 1, 1)
    p_indices = p_indices.expand(-1, n_assets, -1)  # (n_output_days, n_assets, 1)
    forecasts = torch.gather(forecasts_all_output, dim=2, index=p_indices).squeeze(2)  # (n_output_days, n_assets)

    # Step 5: Trim dates to output days
    dates_output = dates[validation_days:]

    # Step 6: Trim coefficients to output days
    coefficients_output = coefficients[validation_days:, :, :, :]

    # Step 7: Transpose to match VARXResult expected shape (n_assets, n_days)
    forecasts = forecasts.T  # (n_assets, n_output_days)
    forecasts_all = forecasts_all.transpose(0, 1)[:, validation_days:, :]  # (n_assets, n_output_days, p_max)

    return VARXResult(
        forecasts=forecasts,
        forecasts_all=forecasts_all,
        p_optimal=p_optimal,
        p_max=p_max,
        coefficients=coefficients_output,
        asset_names=asset_names,
        confounder_names=[],  # Empty for plain VAR
        dates=dates_output,
    )
