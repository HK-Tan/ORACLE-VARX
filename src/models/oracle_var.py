"""ORACLE-VARX: Significance-based lag selection for VAR models.

This module implements ORACLE-VARX, which extends OR-VARX by selecting lag order p
via significance testing rather than direct validation. The significance level α is
then selected via validation RMSE.

Architecture:
    _fit_orvarx_core()           <- Core DML: fold training, residuals, batched OLS
        |
        +---> fit_orvarx_batched()       <- p-selection via validation RMSE
        |
        +---> fit_oraclevarx_batched()   <- p-selection via significance + α-selection

Key Concept:
    For each α in alpha_grid:
    1. Run DML to get θ (deconfounded coefficients) and SE (standard errors)
    2. Compute z-statistics and p-values
    3. Apply Benjamini-Hochberg FDR correction
    4. Keep lags where corrected p-values are significant
    5. This gives p_α (lag order determined by α)

    Then select optimal α via validation RMSE.

Algorithm (fit_oraclevarx_batched):
    Phase 1: Core DML Estimation
    - Call _fit_orvarx_core() to get coefficients and SEs for all p ∈ [1, p_max]
    - Returns ALL test days (no trimming)

    Phase 2: Batched Significance Testing
    - For each α in alpha_grid:
      - For each day d (batched across all days):
        - Start with p_selected = 1
        - For p in [2, p_max]:
          - Test if new lag-p coefficients are significant using BH FDR
          - If significant: p_selected = p, continue
          - Else: break (stop at p-1)
      - Store p_α[d, α_idx] = p_selected

    Phase 3: α Selection via Validation (the ONLY validation)
    - For each α: compute validation RMSE using forecasts at p_α
    - Select α_optimal = argmin(RMSE)

    Phase 4: Forecast Retrieval and Trimming
    - For each output day: extract forecast at p_α[d, α_optimal]
    - n_output_days = n_total_test_days - validation_days (CLEAN formula)
"""

import gc
import torch
import numpy as np
from scipy import stats
from typing import List, Optional

from src.results import ORACLEVARXResult
from src.modules.grid_config import GridConfig
from src.models.dml_pytorch import _fit_orvarx_core
from src.modules.batch_utils import batched_benjamini_hochberg
from src.models.var_pytorch import rolling_alpha_selection


def fit_oraclevarx_batched(
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
    """Fit ORACLE-VARX model using batched operations.

    This function uses _fit_orvarx_core() for DML computation, then applies
    significance-based p-selection and α-selection via validation RMSE.

    Key difference from fit_orvarx_batched():
    - OR-VARX: p-selection via validation RMSE
    - ORACLE-VARX: p-selection via significance tests, α-selection via validation RMSE

    Uses Benjamini-Hochberg FDR correction for significance testing.

    Algorithm:
        Phase 1: Core DML Estimation
        - Call _fit_orvarx_core() to get coefficients and SEs for all p ∈ [1, p_max]
        - Returns ALL test days (no trimming)

        Phase 2: Batched Significance Testing
        - For each α in alpha_grid:
          - For each day d (batched across all days):
            - Start with p_selected = 1
            - For p in [2, p_max]:
              - Test if new lag-p coefficients are significant using BH FDR
              - If significant: p_selected = p, continue
              - Else: break (stop at p-1)
          - Store p_α[d, α_idx] = p_selected

        Phase 3: α Selection via Validation (the ONLY validation)
        - For each α: compute validation RMSE using forecasts at p_α
        - Select α_optimal = argmin(RMSE)

        Phase 4: Forecast Retrieval and Trimming
        - For each output day: extract forecast at p_α[d, α_optimal]

    Args:
        Y: Asset returns, shape (n_days, n_assets)
        W: Confounders, shape (n_days, n_confounders)
        alpha_grid: Significance levels (default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        p_max: Maximum lag order to consider (default: 10)
        config: GridConfig instance (default: None, uses default GridConfig)
        validation_days: Days for α-selection. Must be >= 1.
                         This is the ONLY validation - p is selected via significance tests.
        asset_names: Names of assets (default: None, uses A1, A2, ...)
        confounder_names: Names of confounders (default: None, uses W1, W2, ...)
        dates: Date strings for forecast days (default: None, uses indices)
        learner_name: First-stage learner ('xgboost', 'lgbm', 'rf', 'extra_trees')
        n_jobs: Number of CPU cores (-1 for all, 5 recommended)
        verbose: If True, print detailed progress

    Returns:
        ORACLEVARXResult with forecasts, coefficients_all, alpha_optimal, p_optimal

    Output Shape:
        n_output_days = (n_days - lookback) - validation_days  (CLEAN formula!)

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If DML estimation fails

    Example:
        >>> Y = torch.randn(1100, 5)  # 1100 days, 5 assets
        >>> W = torch.randn(1100, 3)  # 1100 days, 3 confounders
        >>> result = fit_oraclevarx_batched(Y, W, p_max=10)
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

    print(f"Fitting ORACLE-VARX (batched) for {n_total_test_days} total test days, "
          f"{n_alphas} alphas, p_max={p_max}, learner={learner_name}")
    print(f"Grid config: train_size={config.train_size}, test_size={config.test_size}, lookback={lookback}")

    # =========================================================================
    # Phase 1: Core DML Estimation (no validation trimming, no p-selection)
    # =========================================================================
    print("  Phase 1: Running core DML to get θ_all, SE_all, forecasts_all...")

    # Call _fit_orvarx_core() directly - returns ALL test days, no trimming
    forecasts_all_batched_raw, theta_all, SE_all, actuals = _fit_orvarx_core(
        Y=Y,
        W=W,
        p_max=p_max,
        config=config,
        learner_name=learner_name,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Shapes from core:
    # forecasts_all_batched_raw: (n_total_test_days, n_assets, p_max)
    # theta_all: (n_total_test_days, p_max, n_assets, n_assets)
    # SE_all: (n_total_test_days, p_max, n_assets, n_assets)
    # actuals: (n_total_test_days, n_assets)

    # Transpose forecasts to match expected shape for later processing
    # (n_assets, n_total_test_days, p_max)
    forecasts_all_batched = forecasts_all_batched_raw.transpose(0, 1)

    # Memory cleanup after Phase 1
    gc.collect()

    print(f"  Phase 1: Complete. Got θ and SE for {n_total_test_days} days and {p_max} lags.")

    # =========================================================================
    # Phase 2: Batched Significance Testing
    # =========================================================================
    print("  Phase 2: Running batched significance tests for each α...")

    # p_alpha_all: (n_total_test_days, n_alphas) - stores selected p for each (day, α)
    p_alpha_all = torch.zeros(n_total_test_days, n_alphas, device=device, dtype=torch.long)

    for alpha_idx, alpha in enumerate(alpha_grid):
        # For each day, iteratively test lags starting from p=2
        # Start with p_selected = 1 for all days
        p_selected = torch.ones(n_total_test_days, device=device, dtype=torch.long)

        # Test each lag p in [2, p_max]
        for p in range(2, p_max + 1):
            # For each day, check if lag p is significant
            # Extract coefficients for lag p (the new lag being tested)
            # theta_all[d, p-1, :, :] contains all coefficients for lags [1, ..., p]
            # The new lag's coefficients are at positions [(p-1)*n_assets : p*n_assets]
            # But theta_all is reshaped to (p_max, n_assets, n_assets)
            # So we need the (p-1)-th lag index

            # Extract new lag's coefficients and SEs
            # Shape: (n_total_test_days, n_assets, n_assets)
            theta_new = theta_all[:, p - 1, :, :]  # Lag p coefficients
            SE_new = SE_all[:, p - 1, :, :]

            # Compute z-statistics: z = |θ / SE|
            # Add small epsilon to avoid division by zero
            z_new = torch.abs(theta_new / (SE_new + 1e-10))  # (n_total_test_days, n_assets, n_assets)

            # Compute two-tailed p-values using normal distribution
            # p_val = 2 * (1 - Φ(|z|))
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

        # Store selected p for this α
        p_alpha_all[:, alpha_idx] = p_selected

        if verbose:
            p_stats = p_selected.cpu().numpy()
            print(f"    α={alpha:.3f}: p selected - mean={p_stats.mean():.2f}, "
                  f"min={p_stats.min()}, max={p_stats.max()}, "
                  f"median={int(np.median(p_stats))}")

    # Memory cleanup after Phase 2
    gc.collect()

    print("  Phase 2: Complete.")

    # =========================================================================
    # Phase 3: Rolling α Selection via Validation
    # =========================================================================
    print("  Phase 3: Selecting optimal α via ROLLING validation RMSE...")

    # Use rolling_alpha_selection for per-day alpha selection
    # This matches ACLEVARX behavior - each output day has its own optimal α
    # based on trailing validation window [d, d + validation_days)
    alpha_optimal, alpha_counts, alpha_percentages, p_optimal_all_days = rolling_alpha_selection(
        forecasts_all_batched=forecasts_all_batched,
        p_alpha_all=p_alpha_all,
        actuals=actuals,
        validation_days=validation_days,
        alpha_grid=alpha_grid,
        verbose=True,
        use_greek_symbol=True,
    )

    # Print summary statistics for selected p values
    p_optimal_np = p_optimal_all_days.cpu().numpy()
    print(f"  Selected p statistics (rolling α selection):")
    print(f"    Mean: {p_optimal_np.mean():.2f}")
    print(f"    Median: {int(np.median(p_optimal_np))}")
    print(f"    Min: {p_optimal_np.min()}, Max: {p_optimal_np.max()}")
    print(f"    Mode: {int(stats.mode(p_optimal_np, keepdims=False).mode)}")

    # =========================================================================
    # Phase 4: Forecast Retrieval and Trimming
    # =========================================================================
    print("  Phase 4: Retrieving forecasts at optimal (α, p)...")

    # n_output_days = n_total_test_days - validation_days (CLEAN formula!)
    # Already computed and validated at the start of the function

    # Extract forecasts for output days (vectorized)
    # For each output day d, retrieve forecast at (α_optimal[d], p_optimal[d])
    # alpha_optimal and p_optimal_all_days are already indexed for output days (n_output_days,)

    # Slice forecasts to output days only: (n_assets, n_output_days, p_max)
    forecasts_sliced = forecasts_all_batched[:, validation_days:, :]

    # Build gather indices: p_optimal_all_days - 1 (convert to 0-indexed)
    # Shape: (n_assets, n_output_days, 1) for gathering along p_max dimension
    p_indices = (p_optimal_all_days - 1).unsqueeze(0).unsqueeze(-1).expand(n_assets, -1, 1)

    # Gather forecasts at optimal p for each day
    # forecasts_final shape: (n_assets, n_output_days)
    forecasts_final = torch.gather(forecasts_sliced, dim=2, index=p_indices).squeeze(-1)

    # =========================================================================
    # Construct forecasts_all for ORACLEVARXResult
    # =========================================================================
    # forecasts_all should have shape (n_assets, n_output_days, n_alphas)
    # For each (day, α), we need the forecast at p_α[day, α]
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

    # Memory cleanup after forecasts_all construction
    gc.collect()

    # =========================================================================
    # Construct coefficients_all for ORACLEVARXResult
    # =========================================================================
    # coefficients_all should have shape (n_output_days, n_alphas, p_max, n_assets, n_assets)
    # For each (day, α), we use coefficients for lags [1, ..., p_α[day, α]]
    coefficients_all = torch.zeros(
        n_output_days, n_alphas, p_max, n_assets, n_assets,
        device=device, dtype=dtype
    )

    for alpha_idx in range(n_alphas):
        for d in range(n_output_days):
            day_idx_full = validation_days + d
            p_selected = p_alpha_all[day_idx_full, alpha_idx].item()
            # Copy coefficients for lags [0, ..., p_selected-1] (0-indexed)
            coefficients_all[d, alpha_idx, :p_selected, :, :] = theta_all[day_idx_full, :p_selected, :, :]

    # Memory cleanup after coefficients_all construction
    del theta_all
    gc.collect()

    # Prepare arrays for result
    # p_alpha_all needs trimming: (n_total_test_days, n_alphas) -> (n_output_days, n_alphas)
    p_optimal_all_output = p_alpha_all[validation_days:, :]  # (n_output_days, n_alphas)
    # alpha_optimal and p_optimal_all_days are already (n_output_days,) from rolling_alpha_selection
    alpha_optimal_output = alpha_optimal  # (n_output_days,)
    p_optimal_output = p_optimal_all_days  # (n_output_days,)

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
