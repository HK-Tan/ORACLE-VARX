"""ORACLE-VARX: Significance-based lag selection for VAR models.

This module implements ORACLE-VARX, which extends OR-VARX by selecting lag order p
via significance testing rather than direct validation. The significance level α is
then selected via validation RMSE.

Key Concept:
    For each α in alpha_grid:
    1. Run DML to get θ (deconfounded coefficients) and SE (standard errors)
    2. Compute z-statistics and p-values
    3. Apply Benjamini-Hochberg FDR correction
    4. Keep lags where corrected p-values are significant
    5. This gives p_α (lag order determined by α)

    Then select optimal α via validation RMSE.

Algorithm (fit_oracle_varx_cached):
    Phase 1: Batched DML Estimation
    - Call fit_orvarx_batched() to get coefficients and SEs for all p ∈ [1, p_max]

    Phase 2: Batched Significance Testing
    - For each α in alpha_grid:
      - For each day d (batched across all days):
        - Start with p_selected = 1
        - For p in [2, p_max]:
          - Test if new lag-p coefficients are significant using BH FDR
          - If significant: p_selected = p, continue
          - Else: break (stop at p-1)
      - Store p_α[d, α_idx] = p_selected

    Phase 3: α Selection via Validation
    - For each α: compute validation RMSE using forecasts at p_α
    - Select α_optimal = argmin(RMSE)

    Phase 4: Forecast Retrieval
    - For each output day: extract forecast at p_α[d, α_optimal]
"""

import torch
import numpy as np
from scipy import stats
from typing import List, Optional

from src.results import ORACLEVARXResult
from src.modules.grid_config import GridConfig
from src.models.dml_pytorch import fit_orvarx_batched
from src.modules.batch_utils import batched_benjamini_hochberg


def fit_oracle_varx_cached(
    Y: torch.Tensor,
    W: torch.Tensor,
    alpha_grid: List[float] = None,
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    validation_days: int = 21,
    asset_names: Optional[List[str]] = None,
    confounder_names: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
) -> ORACLEVARXResult:
    """Fit ORACLE-VARX model using cached batched approach.

    This function fits ORACLE-VARX by:
    1. Calling fit_orvarx_batched(..., return_se=True) once to get θ_all, SE_all, forecasts_all
    2. Running batched significance tests for each α (vectorized over days)
    3. Selecting optimal α via validation RMSE
    4. Returning ORACLEVARXResult with retrieved forecasts

    Uses Benjamini-Hochberg FDR correction for significance testing.

    Algorithm:
        Phase 1: Batched DML Estimation
        - Call fit_orvarx_batched() to get coefficients and SEs for all p ∈ [1, p_max]

        Phase 2: Batched Significance Testing
        - For each α in alpha_grid:
          - For each day d (batched across all days):
            - Start with p_selected = 1
            - For p in [2, p_max]:
              - Test if new lag-p coefficients are significant using BH FDR
              - If significant: p_selected = p, continue
              - Else: break (stop at p-1)
          - Store p_α[d, α_idx] = p_selected

        Phase 3: α Selection via Validation
        - For each α: compute validation RMSE using forecasts at p_α
        - Select α_optimal = argmin(RMSE)

        Phase 4: Forecast Retrieval
        - For each output day: extract forecast at p_α[d, α_optimal]

    Args:
        Y: Asset returns, shape (n_days, n_assets)
        W: Confounders, shape (n_days, n_confounders)
        alpha_grid: Significance levels (default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        p_max: Maximum lag order to consider (default: 10)
        config: GridConfig instance (default: None, uses default GridConfig)
        validation_days: Number of days to use for validation (default: 21)
        asset_names: Names of assets (default: None, uses A1, A2, ...)
        confounder_names: Names of confounders (default: None, uses W1, W2, ...)
        dates: Date strings for forecast days (default: None, uses indices)
        learner_name: First-stage learner ('xgboost', 'lgbm', 'rf', 'extra_trees')
        n_jobs: Number of CPU cores (-1 for all, 5 recommended)
        verbose: If True, print detailed progress

    Returns:
        ORACLEVARXResult with forecasts, coefficients_all, alpha_optimal, p_optimal

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If DML estimation fails

    Example:
        >>> Y = torch.randn(1100, 5)  # 1100 days, 5 assets
        >>> W = torch.randn(1100, 3)  # 1100 days, 3 confounders
        >>> result = fit_oracle_varx_cached(Y, W, p_max=10)
        >>> print(result.forecasts.shape)  # (5, n_output_days)
        >>> print(result.method)  # 'ORACLE-VARX'
    """
    # Default alpha grid
    if alpha_grid is None:
        alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # Use default config if not provided
    if config is None:
        config = GridConfig()

    lookback = config.lookback_orvarx
    n_days, n_assets = Y.shape
    n_days_w, n_confounders = W.shape
    n_alphas = len(alpha_grid)

    # Validation
    if n_days != n_days_w:
        raise ValueError(f"Y and W must have same number of days: {n_days} vs {n_days_w}")
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

    if confounder_names is None:
        confounder_names = [f"W{i+1}" for i in range(n_confounders)]
    elif len(confounder_names) != n_confounders:
        raise ValueError(f"Expected {n_confounders} confounder names, got {len(confounder_names)}")

    if dates is None:
        dates = [f"D{lookback + validation_days + i}" for i in range(n_output_days)]
    elif len(dates) != n_output_days:
        raise ValueError(f"Expected {n_output_days} dates, got {len(dates)}")

    device = Y.device
    dtype = Y.dtype

    print(f"Fitting ORACLE-VARX (cached) for {n_total_test_days} total test days, "
          f"{n_alphas} alphas, p_max={p_max}, learner={learner_name}")
    print(f"Grid config: train_size={config.train_size}, test_size={config.test_size}, lookback={lookback}")

    # =========================================================================
    # Phase 1: Batched DML Estimation
    # =========================================================================
    print("  Phase 1: Running batched DML to get θ_all, SE_all, forecasts_all...")

    # Call fit_orvarx_batched with return_se=True
    # IMPORTANT: fit_orvarx_batched uses validation_days for p-selection and trims output.
    # We pass validation_days=1 (minimum) to get maximum data, then we'll handle
    # our own validation period for alpha selection.
    # This gives us (n_total_test_days - 1) days of data.
    varx_result, SE_all_raw = fit_orvarx_batched(
        Y=Y,
        W=W,
        p_max=p_max,
        config=config,
        validation_days=1,  # Minimal trimming - we handle validation ourselves
        asset_names=asset_names,
        confounder_names=confounder_names,
        dates=None,  # We'll handle dates ourselves
        learner_name=learner_name,
        n_jobs=n_jobs,
        verbose=verbose,
        return_se=True,
    )

    # Extract arrays from VARXResult
    # NOTE: Because we passed validation_days=1, the returned arrays have shape:
    # varx_result.coefficients: (n_total_test_days-1, p_max, n_assets, n_assets)
    # varx_result.forecasts_all: (n_assets, n_total_test_days-1, p_max)
    # SE_all_raw: (n_total_test_days-1, p_max, n_assets, n_assets)
    # We'll treat these as the full available dataset for alpha selection.

    theta_all = varx_result.coefficients  # (n_total_test_days-1, p_max, n_assets, n_assets)
    SE_all = SE_all_raw  # (n_total_test_days-1, p_max, n_assets, n_assets)
    forecasts_all_batched = varx_result.forecasts_all  # (n_assets, n_total_test_days-1, p_max)

    # Adjust n_total_test_days to match actual data we got
    n_total_test_days_actual = theta_all.shape[0]

    print(f"  Phase 1: Complete. Got θ and SE for {n_total_test_days_actual} days and {p_max} lags.")

    # =========================================================================
    # Phase 2: Batched Significance Testing
    # =========================================================================
    print("  Phase 2: Running batched significance tests for each α...")

    # p_alpha_all: (n_total_test_days_actual, n_alphas) - stores selected p for each (day, α)
    p_alpha_all = torch.zeros(n_total_test_days_actual, n_alphas, device=device, dtype=torch.long)

    for alpha_idx, alpha in enumerate(alpha_grid):
        # For each day, iteratively test lags starting from p=2
        # Start with p_selected = 1 for all days
        p_selected = torch.ones(n_total_test_days_actual, device=device, dtype=torch.long)

        # Test each lag p in [2, p_max]
        for p in range(2, p_max + 1):
            # For each day, check if lag p is significant
            # Extract coefficients for lag p (the new lag being tested)
            # theta_all[d, p-1, :, :] contains all coefficients for lags [1, ..., p]
            # The new lag's coefficients are at positions [(p-1)*n_assets : p*n_assets]
            # But theta_all is reshaped to (p_max, n_assets, n_assets)
            # So we need the (p-1)-th lag index

            # Extract new lag's coefficients and SEs
            # Shape: (n_total_test_days_actual, n_assets, n_assets)
            theta_new = theta_all[:, p - 1, :, :]  # Lag p coefficients
            SE_new = SE_all[:, p - 1, :, :]

            # Compute z-statistics: z = |θ / SE|
            # Add small epsilon to avoid division by zero
            z_new = torch.abs(theta_new / (SE_new + 1e-10))  # (n_total_test_days_actual, n_assets, n_assets)

            # Compute two-tailed p-values using normal distribution
            # p_val = 2 * (1 - Φ(|z|))
            # Use torch.distributions.Normal for CDF
            normal_dist = torch.distributions.Normal(0, 1)
            p_vals = 2 * (1 - normal_dist.cdf(z_new))  # (n_total_test_days_actual, n_assets, n_assets)

            # Flatten p-values for each day: (n_total_test_days_actual, n_assets * n_assets)
            p_vals_flat = p_vals.reshape(n_total_test_days_actual, n_assets * n_assets)

            # Apply Benjamini-Hochberg FDR correction (batched over days)
            # reject: (n_total_test_days_actual, n_assets * n_assets)
            reject = batched_benjamini_hochberg(p_vals_flat, alpha)

            # Check if any coefficient is significant for each day
            # is_significant: (n_total_test_days_actual,)
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

        # Store selected p for this α
        p_alpha_all[:, alpha_idx] = p_selected

        if verbose:
            p_stats = p_selected.cpu().numpy()
            print(f"    α={alpha:.3f}: p selected - mean={p_stats.mean():.2f}, "
                  f"min={p_stats.min()}, max={p_stats.max()}, "
                  f"median={int(np.median(p_stats))}")

    print("  Phase 2: Complete.")

    # =========================================================================
    # Phase 3: α Selection via Validation
    # =========================================================================
    print("  Phase 3: Selecting optimal α via validation RMSE...")

    # For each α, compute validation RMSE using forecasts at selected p
    # forecasts_all_batched: (n_assets, n_total_test_days_actual, p_max)
    # p_alpha_all: (n_total_test_days_actual, n_alphas)

    # Extract validation days
    # Note: We have n_total_test_days_actual days starting from index 1 in the full test period
    # (because we passed validation_days=1 to fit_orvarx_batched)
    val_start = 0
    val_end = min(validation_days, n_total_test_days_actual)

    # Get actual returns for validation period
    # Since fit_orvarx_batched was called with validation_days=1, the data starts at day 1
    # of the full test period. So actual validation corresponds to days [1, 1+validation_days)
    # in the original full test period.
    actuals = Y[lookback + 1:, :]  # (n_total_test_days_actual, n_assets) - skip first validation day
    actuals_val = actuals[val_start:val_end, :]  # (val_days, n_assets)

    # Compute RMSE for each α
    rmse_per_alpha = torch.zeros(n_alphas, device=device, dtype=dtype)

    for alpha_idx in range(n_alphas):
        # For each validation day, extract forecast at p_α[d, α_idx]
        forecasts_alpha = torch.zeros(val_end - val_start, n_assets, device=device, dtype=dtype)

        for d in range(val_start, val_end):
            p_selected = p_alpha_all[d, alpha_idx].item()
            # forecasts_all_batched[:, d, p_selected-1] gives forecast for day d at lag p_selected
            forecasts_alpha[d - val_start, :] = forecasts_all_batched[:, d, p_selected - 1]

        # Compute RMSE
        errors = forecasts_alpha - actuals_val  # (val_days, n_assets)
        rmse_per_alpha[alpha_idx] = torch.sqrt((errors ** 2).mean())

    # Select α with minimum RMSE
    alpha_optimal_idx = torch.argmin(rmse_per_alpha).item()
    alpha_optimal_value = alpha_grid[alpha_optimal_idx]

    # Create alpha_optimal tensor: (n_total_test_days_actual,) with constant value
    alpha_optimal = torch.full((n_total_test_days_actual,), alpha_optimal_idx, device=device, dtype=torch.long)

    print(f"  Optimal α: {alpha_optimal_value:.3f} (idx={alpha_optimal_idx}, "
          f"RMSE: {rmse_per_alpha[alpha_optimal_idx]:.6f})")

    # Print RMSE for all alphas
    print(f"  RMSE per α:")
    for idx, (alpha_val, rmse_val) in enumerate(zip(alpha_grid, rmse_per_alpha)):
        marker = " *" if idx == alpha_optimal_idx else ""
        print(f"    α={alpha_val:.3f}: RMSE={rmse_val:.6f}{marker}")

    # Extract p_optimal from p_alpha_all using optimal α
    p_optimal_all_days = p_alpha_all[:, alpha_optimal_idx]  # (n_total_test_days,)

    # Print summary statistics for selected p values
    p_optimal_np = p_optimal_all_days.cpu().numpy()
    print(f"  Selected p statistics (at optimal α={alpha_optimal_value:.3f}):")
    print(f"    Mean: {p_optimal_np.mean():.2f}")
    print(f"    Median: {int(np.median(p_optimal_np))}")
    print(f"    Min: {p_optimal_np.min()}, Max: {p_optimal_np.max()}")
    print(f"    Mode: {int(stats.mode(p_optimal_np, keepdims=False).mode)}")

    # =========================================================================
    # Phase 4: Forecast Retrieval
    # =========================================================================
    print("  Phase 4: Retrieving forecasts at optimal (α, p)...")

    # Calculate actual output days
    # We want to skip the validation period and return forecasts for remaining days
    n_output_days_actual = n_total_test_days_actual - validation_days
    if n_output_days_actual < 1:
        raise ValueError(
            f"Insufficient data after validation: n_total_test_days_actual={n_total_test_days_actual}, "
            f"validation_days={validation_days}. Need at least 1 output day."
        )

    # Extract forecasts for output days (skip validation days)
    # For each output day, retrieve forecast at p_optimal_all_days[validation_days + d]
    forecasts_output = torch.zeros(n_output_days_actual, n_assets, device=device, dtype=dtype)

    for d in range(n_output_days_actual):
        day_idx_full = validation_days + d  # Index in available data
        p_selected = p_optimal_all_days[day_idx_full].item()
        # Retrieve forecast from forecasts_all_batched
        forecasts_output[d, :] = forecasts_all_batched[:, day_idx_full, p_selected - 1]

    # Transpose to match ORACLEVARXResult expected shape: (n_assets, n_output_days_actual)
    forecasts_final = forecasts_output.T

    # =========================================================================
    # Construct forecasts_all for ORACLEVARXResult
    # =========================================================================
    # forecasts_all should have shape (n_assets, n_output_days_actual, n_alphas)
    # For each (day, α), we need the forecast at p_α[day, α]
    forecasts_all = torch.zeros(n_assets, n_output_days_actual, n_alphas, device=device, dtype=dtype)

    for alpha_idx in range(n_alphas):
        for d in range(n_output_days_actual):
            day_idx_full = validation_days + d
            p_selected = p_alpha_all[day_idx_full, alpha_idx].item()
            forecasts_all[:, d, alpha_idx] = forecasts_all_batched[:, day_idx_full, p_selected - 1]

    # =========================================================================
    # Construct coefficients_all for ORACLEVARXResult
    # =========================================================================
    # coefficients_all should have shape (n_output_days_actual, n_alphas, p_max, n_assets, n_assets)
    # For each (day, α), we use coefficients for lags [1, ..., p_α[day, α]]
    coefficients_all = torch.zeros(
        n_output_days_actual, n_alphas, p_max, n_assets, n_assets,
        device=device, dtype=dtype
    )

    for alpha_idx in range(n_alphas):
        for d in range(n_output_days_actual):
            day_idx_full = validation_days + d
            p_selected = p_alpha_all[day_idx_full, alpha_idx].item()
            # Copy coefficients for lags [0, ..., p_selected-1] (0-indexed)
            coefficients_all[d, alpha_idx, :p_selected, :, :] = theta_all[day_idx_full, :p_selected, :, :]

    # Trim arrays to output days
    p_optimal_all_output = p_alpha_all[validation_days:, :]  # (n_output_days_actual, n_alphas)
    alpha_optimal_output = alpha_optimal[validation_days:]  # (n_output_days_actual,)
    p_optimal_output = p_optimal_all_days[validation_days:]  # (n_output_days_actual,)

    # Store SE_all (full test period) for optional future use
    SE_all_output = SE_all  # Keep full array for reference

    print("  Phase 4: Complete.")

    return ORACLEVARXResult(
        forecasts=forecasts_final,
        forecasts_all=forecasts_all,
        p_optimal_all=p_optimal_all_output,
        alpha_optimal=alpha_optimal_output,
        p_optimal=p_optimal_output,
        alpha_grid=alpha_grid,
        coefficients_all=coefficients_all,
        asset_names=asset_names,
        confounder_names=confounder_names,
        dates=dates,
        SE_all=SE_all_output,
    )
