"""ACLE-VARX: VAR with significance-based p-selection + alpha-selection.

This module implements ACLE-VARX, which applies the same significance-testing
methodology as ORACLE-VARX but to plain VAR models (without DML/orthogonalization).

Architecture:
    batch_var_all_days_with_se()  <- Batched VAR with standard errors
        |
        +---> fit_aclevarx()      <- p-selection via significance + alpha-selection

Key Concept:
    For each alpha in alpha_grid:
    1. Run VAR to get coefficients and SE (standard errors)
    2. Compute z-statistics and p-values
    3. Apply Benjamini-Hochberg FDR correction
    4. Keep lags where corrected p-values are significant
    5. This gives p_alpha (lag order determined by alpha)

    Then select optimal alpha via validation RMSE.

Algorithm (fit_aclevarx):
    Phase 1: Batched VAR Estimation with SE
    - Call batch_var_all_days_with_se() to get coefficients and SEs for all p in [1, p_max]
    - Returns ALL test days (no trimming)

    Phase 2: Batched Significance Testing
    - For each alpha in alpha_grid:
      - For each day d (batched across all days):
        - Start with p_selected = 1
        - For p in [2, p_max]:
          - Test if new lag-p coefficients are significant using BH FDR
          - If significant: p_selected = p, continue
          - Else: break (stop at p-1)
      - Store p_alpha[d, alpha_idx] = p_selected

    Phase 3: alpha Selection via Validation (the ONLY validation)
    - For each alpha: compute validation RMSE using forecasts at p_alpha
    - Select alpha_optimal = argmin(RMSE)

    Phase 4: Forecast Retrieval and Trimming
    - For each output day: extract forecast at p_alpha[d, alpha_optimal]
    - n_output_days = n_total_test_days - validation_days (CLEAN formula)
"""

import torch
import numpy as np
from scipy import stats
from typing import Tuple, List, Optional

from src.results import ACLEVARXResult
from src.modules.grid_config import GridConfig
from src.modules.batch_utils import batched_ols, batched_benjamini_hochberg
from src.models.var_pytorch import build_var_design_batch, build_pred_features, rolling_alpha_selection


def batch_var_all_days_with_se(
    Y: torch.Tensor,
    p_max: int,
    lookback: int,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched VAR estimation with standard errors for all days and all lag orders.

    Extends batch_var_all_days() to also compute OLS standard errors for
    significance testing. This is the KEY FUNCTION for ACLE-VARX.

    Algorithm:
    1. Stack sliding windows for all test days
    2. For each p in 1..p_max:
       - Build batched design matrices
       - Solve batched OLS: beta = (X'X)^{-1} X'Y
       - Compute residuals and estimate sigma^2
       - Compute SEs from (X'X)^{-1} and sigma^2
       - Extract coefficient matrices A_1, ..., A_p
       - Generate predictions

    Standard Error Computation:
        For VAR(p): Y = X @ beta + epsilon
        beta = (X'X)^{-1} X'Y
        XtX_inv = (X'X)^{-1}
        residuals = Y - X @ beta
        sigma_sq[j] = sum(residuals[:, j]^2) / (T - k)  where k = 1 + n_assets * p
        SE[i, j] = sqrt(diag(XtX_inv)[i] * sigma_sq[j])

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
        standard_errors: Standard errors for coefficients, shape (n_test_days, p_max, n_assets, n_assets)
                        standard_errors[d, k, i, j] = SE of coefficient[d, k, i, j]
        actuals: Actual returns for test days, shape (n_test_days, n_assets)

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
    standard_errors = torch.zeros(n_test_days, p_max, n_assets, n_assets, device=device, dtype=dtype)

    # Extract actual returns for test period
    actuals = Y[lookback:, :]  # (n_test_days, n_assets)

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
        batch_size, T, n_features = X_batch.shape

        # k = number of parameters = 1 (intercept) + n_assets * p (lagged coefficients)
        k = n_features  # Already equals 1 + n_assets * p

        # Compute X'X for SE calculation
        # X_batch: (batch, T, n_features)
        # XtX: (batch, n_features, n_features)
        XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)

        # Compute XtX_inv using batched inverse
        # Add small regularization for numerical stability
        eye = torch.eye(n_features, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        XtX_reg = XtX + 1e-6 * eye
        try:
            XtX_inv = torch.linalg.inv(XtX_reg)  # (batch, n_features, n_features)
        except torch.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Failed to invert X'X in VAR({p}) estimation: {e}. "
                f"Consider reducing p_max or increasing lookback."
            ) from e

        # Batched OLS: beta = (X'X)^{-1} X'Y
        try:
            beta = batched_ols(X_batch, Y_batch, chunk_size=chunk_size)  # (batch, n_features, n_assets)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed VAR({p}) estimation: {e}. "
                f"Consider reducing p_max or increasing lookback."
            ) from e

        # Compute residuals: Y - X @ beta
        # X_batch: (batch, T, n_features)
        # beta: (batch, n_features, n_assets)
        # Y_batch: (batch, T, n_assets)
        Y_pred_train = torch.bmm(X_batch, beta)  # (batch, T, n_assets)
        residuals = Y_batch - Y_pred_train  # (batch, T, n_assets)

        # Estimate sigma^2 for each asset: sum(residuals^2) / (T - k)
        # residuals: (batch, T, n_assets)
        residual_ss = (residuals ** 2).sum(dim=1)  # (batch, n_assets)
        dof = T - k  # degrees of freedom
        if dof <= 0:
            raise ValueError(
                f"Insufficient degrees of freedom for VAR({p}): T={T}, k={k}. "
                f"Need T > k. Consider increasing lookback or reducing p_max."
            )
        sigma_sq = residual_ss / dof  # (batch, n_assets)

        # Compute standard errors for each coefficient
        # SE[i, j] = sqrt(diag(XtX_inv)[i] * sigma_sq[j])
        # diag(XtX_inv): (batch, n_features)
        diag_XtX_inv = torch.diagonal(XtX_inv, dim1=1, dim2=2)  # (batch, n_features)

        # SE for all features and all targets: sqrt(diag_XtX_inv[:, :, None] * sigma_sq[:, None, :])
        # Shape: (batch, n_features, n_assets)
        SE_all = torch.sqrt(diag_XtX_inv.unsqueeze(2) * sigma_sq.unsqueeze(1))  # (batch, n_features, n_assets)

        # Extract coefficient matrices and SEs (excluding intercept)
        # beta shape: (batch, 1 + n_assets*p, n_assets)
        # beta[:, :, i] gives coefficients for predicting asset i
        # We want coefficients[d, k, i, j] = effect of asset j at lag k+1 on asset i
        beta_no_intercept = beta[:, 1:, :]  # (batch, n_assets*p, n_assets)
        SE_no_intercept = SE_all[:, 1:, :]  # (batch, n_assets*p, n_assets)

        # Reshape and transpose to match coefficient convention:
        # beta_no_intercept[:, :, i] contains coeffs predicting asset i
        # After reshape: (batch, p, n_assets, n_assets) where [:, k, :, i] = coeffs for asset i
        # After transpose: [:, k, i, j] = effect of asset j at lag k+1 on asset i
        A_matrices = beta_no_intercept.view(n_test_days, p, n_assets, n_assets)
        SE_matrices = SE_no_intercept.view(n_test_days, p, n_assets, n_assets)
        A_matrices = A_matrices.transpose(-2, -1)
        SE_matrices = SE_matrices.transpose(-2, -1)

        # Store coefficients and SEs
        coefficients[:, :p, :, :] = A_matrices
        standard_errors[:, :p, :, :] = SE_matrices

        # Build prediction features and generate forecasts
        X_pred = build_pred_features(windows, p)  # (batch, 1, n_features)
        forecasts[:, :, p - 1] = torch.bmm(X_pred, beta).squeeze(1)

    return forecasts, coefficients, standard_errors, actuals


def fit_aclevarx(
    Y: torch.Tensor,
    alpha_grid: List[float] = None,
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    validation_days: int = 21,
    asset_names: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    verbose: bool = False,
) -> ACLEVARXResult:
    """Fit ACLE-VARX model: VAR with significance-based p-selection + alpha-selection.

    ACLE-VARX applies the same significance-testing methodology as ORACLE-VARX
    but to plain VAR models (without DML/orthogonalization).

    Key difference from plain VAR:
    - VAR: p-selection via validation RMSE
    - ACLE-VARX: p-selection via significance tests, alpha-selection via validation RMSE

    Uses Benjamini-Hochberg FDR correction for significance testing.

    Algorithm:
        Phase 1: Batched VAR Estimation with SE
        - Call batch_var_all_days_with_se() to get coefficients and SEs for all p in [1, p_max]
        - Returns ALL test days (no trimming)

        Phase 2: Batched Significance Testing
        - For each alpha in alpha_grid:
          - For each day d (batched across all days):
            - Start with p_selected = 1
            - For p in [2, p_max]:
              - Test if new lag-p coefficients are significant using BH FDR
              - If significant: p_selected = p, continue
              - Else: break (stop at p-1)
          - Store p_alpha[d, alpha_idx] = p_selected

        Phase 3: alpha Selection via Validation (the ONLY validation)
        - For each alpha: compute validation RMSE using forecasts at p_alpha
        - Select alpha_optimal = argmin(RMSE)

        Phase 4: Forecast Retrieval and Trimming
        - For each output day: extract forecast at p_alpha[d, alpha_optimal]

    Args:
        Y: Asset returns, shape (n_days, n_assets)
        alpha_grid: Significance levels (default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        p_max: Maximum lag order to consider (default: 10)
        config: GridConfig instance (default: None, uses default GridConfig)
                Uses config.lookback_var (514) for the lookback window.
        validation_days: Days for alpha-selection. Must be >= 1.
                         This is the ONLY validation - p is selected via significance tests.
        asset_names: Names of assets (default: None, uses A1, A2, ...)
        dates: Date strings for forecast days (default: None, uses indices)
        verbose: If True, print detailed progress

    Returns:
        ACLEVARXResult with forecasts, alpha_optimal, p_optimal

    Output Shape:
        n_output_days = (n_days - lookback_var) - validation_days  (CLEAN formula!)

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If VAR estimation fails

    Example:
        >>> Y = torch.randn(600, 5)  # 600 days, 5 assets
        >>> result = fit_aclevarx(Y, p_max=10)
        >>> print(result.forecasts.shape)  # (5, n_output_days)
        >>> print(result.method)  # 'ACLE-VARX'
    """
    # Default alpha grid
    if alpha_grid is None:
        alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # Use default config if not provided
    if config is None:
        config = GridConfig()

    lookback = config.lookback_var  # Use VAR lookback (514), not OR-VARX lookback (1018)
    n_days, n_assets = Y.shape
    n_alphas = len(alpha_grid)

    # Validation
    if p_max < 1:
        raise ValueError(f"p_max must be >= 1, got {p_max}")
    if lookback < p_max:
        raise ValueError(f"lookback {lookback} must be >= p_max {p_max}")
    if n_days <= lookback:
        raise ValueError(f"Need at least {lookback + 1} days, got {n_days}")
    if validation_days < 1:
        raise ValueError(f"validation_days must be >= 1, got {validation_days}")
    if n_alphas < 1:
        raise ValueError(f"alpha_grid must have at least 1 value, got {n_alphas}")

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

    if dates is None:
        dates = [f"D{lookback + validation_days + i}" for i in range(n_output_days)]
    elif len(dates) != n_output_days:
        raise ValueError(f"Expected {n_output_days} dates, got {len(dates)}")

    device = Y.device
    dtype = Y.dtype

    print(f"Fitting ACLE-VARX for {n_total_test_days} total test days, "
          f"{n_alphas} alphas, p_max={p_max}")
    print(f"Grid config: lookback_var={lookback}")

    # =========================================================================
    # Phase 1: Batched VAR Estimation with SE
    # =========================================================================
    print("  Phase 1: Running batched VAR with SE computation...")

    forecasts_all_batched_raw, theta_all, SE_all, actuals = batch_var_all_days_with_se(
        Y=Y,
        p_max=p_max,
        lookback=lookback,
        chunk_size=config.batch_chunk_size,
    )

    # Shapes from batch_var_all_days_with_se:
    # forecasts_all_batched_raw: (n_total_test_days, n_assets, p_max)
    # theta_all: (n_total_test_days, p_max, n_assets, n_assets)
    # SE_all: (n_total_test_days, p_max, n_assets, n_assets)
    # actuals: (n_total_test_days, n_assets)

    # Transpose forecasts to match expected shape for later processing
    # (n_assets, n_total_test_days, p_max)
    forecasts_all_batched = forecasts_all_batched_raw.transpose(0, 1)

    print(f"  Phase 1: Complete. Got coefficients and SE for {n_total_test_days} days and {p_max} lags.")

    # =========================================================================
    # Phase 2: Batched Significance Testing
    # =========================================================================
    print("  Phase 2: Running batched significance tests for each alpha...")

    # p_alpha_all: (n_total_test_days, n_alphas) - stores selected p for each (day, alpha)
    p_alpha_all = torch.zeros(n_total_test_days, n_alphas, device=device, dtype=torch.long)

    for alpha_idx, alpha in enumerate(alpha_grid):
        # For each day, iteratively test lags starting from p=2
        # Start with p_selected = 1 for all days
        p_selected = torch.ones(n_total_test_days, device=device, dtype=torch.long)

        # Test each lag p in [2, p_max]
        for p in range(2, p_max + 1):
            # For each day, check if lag p is significant
            # Extract new lag's coefficients and SEs
            # Shape: (n_total_test_days, n_assets, n_assets)
            theta_new = theta_all[:, p - 1, :, :]  # Lag p coefficients
            SE_new = SE_all[:, p - 1, :, :]

            # Compute z-statistics: z = |theta / SE|
            # Add small epsilon to avoid division by zero
            z_new = torch.abs(theta_new / (SE_new + 1e-10))  # (n_total_test_days, n_assets, n_assets)

            # Compute two-tailed p-values using normal distribution
            # p_val = 2 * (1 - Phi(|z|))
            # Use torch.distributions.Normal for CDF
            normal_dist = torch.distributions.Normal(0, 1)
            p_vals = 2 * (1 - normal_dist.cdf(z_new))  # (n_total_test_days, n_assets, n_assets)

            # Flatten p-values for each day: (n_total_test_days, n_assets * n_assets)
            p_vals_flat = p_vals.reshape(n_total_test_days, n_assets * n_assets)

            # Apply Benjamini-Hochberg FDR correction (batched over days)
            # reject: (n_total_test_days, n_assets * n_assets)
            reject = batched_benjamini_hochberg(p_vals_flat, alpha)

            # Check if any coefficient is significant for each day
            # is_significant: (n_total_test_days,)
            is_significant = reject.any(dim=1)

            # Update p_selected: if significant, set to p; else keep current value and stop
            # For vectorized stopping, we use a mask
            # If is_significant[d] and p_selected[d] == (p-1), then p_selected[d] = p
            # We need to track which days are still "active" (haven't stopped yet)
            # A day stops when it encounters the first non-significant lag

            # Update p_selected where significant AND haven't stopped yet
            # A day has stopped if p_selected[d] < p-1 (i.e., stopped at earlier lag)
            still_active = (p_selected == p - 1)  # Days that selected p-1 in previous iteration
            should_update = is_significant & still_active
            p_selected = torch.where(should_update, torch.tensor(p, device=device, dtype=torch.long), p_selected)

        # Store selected p for this alpha
        p_alpha_all[:, alpha_idx] = p_selected

        if verbose:
            p_stats = p_selected.cpu().numpy()
            print(f"    alpha={alpha:.3f}: p selected - mean={p_stats.mean():.2f}, "
                  f"min={p_stats.min()}, max={p_stats.max()}, "
                  f"median={int(np.median(p_stats))}")

    print("  Phase 2: Complete.")

    # =========================================================================
    # Phase 3: Rolling alpha Selection via Validation (the ONLY validation)
    # =========================================================================
    print("  Phase 3: Selecting optimal alpha via ROLLING validation RMSE...")

    alpha_optimal, alpha_counts, alpha_percentages, p_optimal_all_days = rolling_alpha_selection(
        forecasts_all_batched=forecasts_all_batched,
        p_alpha_all=p_alpha_all,
        actuals=actuals,
        validation_days=validation_days,
        alpha_grid=alpha_grid,
        verbose=True,
        use_greek_symbol=False,  # Use "alpha" instead of "α"
    )

    # Print summary statistics for selected p values
    p_optimal_np = p_optimal_all_days.cpu().numpy()
    print(f"\n  Selected p statistics (rolling alpha selection):")
    print(f"    Mean: {p_optimal_np.mean():.2f}")
    print(f"    Median: {int(np.median(p_optimal_np))}")
    print(f"    Min: {p_optimal_np.min()}, Max: {p_optimal_np.max()}")
    print(f"    Mode: {int(stats.mode(p_optimal_np, keepdims=False).mode)}")

    # =========================================================================
    # Phase 4: Forecast Retrieval and Trimming
    # =========================================================================
    print("  Phase 4: Retrieving forecasts at optimal (alpha, p)...")

    # n_output_days = n_total_test_days - validation_days (CLEAN formula!)
    # Already computed and validated at the start of the function

    # Extract forecasts for output days using per-day optimal alpha (vectorized)
    # For each output day d, use alpha_optimal[d] to get p, then retrieve forecast

    # Slice forecasts to output days only: (n_assets, n_output_days, p_max)
    forecasts_sliced = forecasts_all_batched[:, validation_days:, :]

    # Build gather indices: p_optimal_all_days - 1 (convert to 0-indexed)
    # Shape: (n_assets, n_output_days, 1) for gathering along p_max dimension
    p_indices = (p_optimal_all_days - 1).unsqueeze(0).unsqueeze(-1).expand(n_assets, -1, 1)

    # Gather forecasts at optimal p for each day
    # forecasts_final shape: (n_assets, n_output_days)
    forecasts_final = torch.gather(forecasts_sliced, dim=2, index=p_indices).squeeze(-1)

    # =========================================================================
    # Construct forecasts_all for ACLEVARXResult
    # =========================================================================
    # forecasts_all should have shape (n_assets, n_output_days, n_alphas)
    # For each (day, alpha), we need the forecast at p_alpha[day, alpha]
    forecasts_all = torch.zeros(n_assets, n_output_days, n_alphas, device=device, dtype=dtype)

    # Pre-slice forecasts to output days: (n_assets, n_output_days, p_max)
    forecasts_sliced = forecasts_all_batched[:, validation_days:, :]
    # Get p values for output days: (n_output_days, n_alphas)
    p_alpha_output = p_alpha_all[validation_days:, :]

    # Vectorize over days, loop over alphas (only 7 alphas vs 1000+ days)
    for alpha_idx in range(n_alphas):
        # p indices for this alpha across all days: (n_output_days,) -> expand to (n_assets, n_output_days, 1)
        p_idx = (p_alpha_output[:, alpha_idx] - 1).unsqueeze(0).unsqueeze(-1).expand(n_assets, -1, 1)
        forecasts_all[:, :, alpha_idx] = torch.gather(forecasts_sliced, dim=2, index=p_idx).squeeze(-1)

    # Trim arrays to output days
    p_optimal_all_output = p_alpha_all[validation_days:, :]  # (n_output_days, n_alphas)
    # alpha_optimal is already (n_output_days,) from Phase 3
    alpha_optimal_output = alpha_optimal  # (n_output_days,)
    # p_optimal_all_days is already (n_output_days,) from Phase 3
    p_optimal_output = p_optimal_all_days  # (n_output_days,)

    # Store SE_all (full test period) for optional future use
    SE_all_output = SE_all  # Keep full array for reference

    print("  Phase 4: Complete.")

    return ACLEVARXResult(
        forecasts=forecasts_final,
        forecasts_all=forecasts_all,
        p_optimal_all=p_optimal_all_output,
        alpha_optimal=alpha_optimal_output,
        p_optimal=p_optimal_output,
        alpha_grid=alpha_grid,
        asset_names=asset_names,
        confounder_names=[],  # No confounders in base ACLE-VARX (added in post-processing if needed)
        dates=dates,
        SE_all=SE_all_output,
    )
