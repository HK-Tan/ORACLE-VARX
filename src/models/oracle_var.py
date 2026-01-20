"""ORACLE-VARX: Significance-based lag selection for VAR models.

This module implements ORACLE-VARX, which extends OR-VARX by selecting lag order p
via significance testing rather than direct validation. The significance level α is
then selected via validation RMSE.

Key Concept:
    For each α in alpha_grid:
    1. Run DML to get θ (deconfounded coefficients) and SE (standard errors)
    2. Compute z-statistics: z = θ / SE
    3. Keep lags where |z| > z_critical(α)
    4. This gives p_α (lag order determined by α)

    Then select optimal α via validation RMSE.

Algorithm:
    1. For each test day d:
       For each α in alpha_grid:
         a) Fit DML with p_max lags to get theta and SE
         b) Run significance test with α → get p_α
         c) Re-fit OR-VARX with p_α and generate forecast
         d) Store forecast in 3D tensor [:, d, α_idx]

    2. Select optimal α via validation RMSE
    3. Extract forecasts at optimal α

References:
    See plans/implementation-plan.md Section 9 for α selection logic
"""

import torch
import numpy as np
from scipy import stats
from typing import Tuple, List, Optional

from src.results import ORACLEVARXResult
from src.modules.grid_config import GridConfig
from src.modules.model_cache import ModelCache
from src.models.dml_pytorch import fit_orvarx_single_day


def oracle_significance_test(
    theta: torch.Tensor,
    se: torch.Tensor,
    alpha: float,
    p_max: int,
    n_assets: int,
) -> int:
    """Determine lag order p via significance testing.

    For each lag k, check if any coefficient at that lag is significant.
    A lag is significant if |z_ijk| > z_critical for any (i,j) pair.

    The test iterates from lag 1 to p_max. The selected p is the highest lag
    where at least one coefficient is significant.

    Args:
        theta: Coefficients, shape (n_treatments, n_assets) = (n_assets*p_max, n_assets)
        se: Standard errors, same shape as theta
        alpha: Significance level (e.g., 0.05)
        p_max: Maximum lag order
        n_assets: Number of assets

    Returns:
        p: Selected lag order (at least 1)

    Raises:
        ValueError: If theta/se shapes are inconsistent with p_max and n_assets

    Example:
        >>> theta = torch.randn(20, 5)  # 4 lags * 5 assets = 20 treatments, 5 outcomes
        >>> se = torch.abs(torch.randn(20, 5)) + 0.1
        >>> p = oracle_significance_test(theta, se, alpha=0.05, p_max=4, n_assets=5)
        >>> print(p)  # Selected lag order (1-4)
    """
    # Validation
    if theta.shape != se.shape:
        raise ValueError(f"theta and se must have same shape, got {theta.shape} vs {se.shape}")

    n_treatments = theta.shape[0]
    expected_treatments = n_assets * p_max

    if n_treatments != expected_treatments:
        raise ValueError(
            f"theta has {n_treatments} treatments but expected {expected_treatments} "
            f"(n_assets={n_assets} * p_max={p_max})"
        )

    # Compute z-statistics
    z_critical = stats.norm.ppf(1 - alpha / 2)  # Two-tailed test
    z_stats = torch.abs(theta / (se + 1e-10))  # Add small epsilon to avoid division by zero

    # Vectorized significance test across all lags
    # Reshape z_stats from (n_assets * p_max, n_assets) to (p_max, n_assets, n_assets)
    # Each (n_assets, n_assets) slice represents the z-statistics for lag k
    z_reshaped = z_stats.view(p_max, n_assets, n_assets)

    # Check if any coefficient at each lag is significant
    # sig_per_lag[k] is True if any coefficient at lag k+1 exceeds z_critical
    sig_per_lag = (z_reshaped > z_critical).any(dim=(1, 2))  # (p_max,) boolean

    # Find the highest lag with at least one significant coefficient
    # We want p = max{k : at least one |z_ijk| > z_critical}
    if sig_per_lag.any():
        # Get indices where significant, take the last one, add 1 for 1-indexing
        # nonzero returns 0-indexed positions, so +1 converts to lag number
        p = (sig_per_lag.nonzero(as_tuple=True)[0][-1] + 1).item()
    else:
        p = 1  # Fallback to lag 1 if nothing significant

    return p


def fit_oracle_varx(
    Y: torch.Tensor,
    W: torch.Tensor,
    alpha_grid: List[float] = None,
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    validation_days: int = 20,
    asset_names: Optional[List[str]] = None,
    confounder_names: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    learner_name: str = 'xgboost',
    n_jobs: int = -1,
    verbose: bool = False,
) -> ORACLEVARXResult:
    """Fit ORACLE-VARX model with significance-based lag selection.

    This is the main entry point for ORACLE-VARX estimation. It performs:
    1. For each test day:
       - For each α in alpha_grid:
         - Fit DML with p_max lags to get theta and SE
         - Run significance test with α → get p_α
         - Re-fit OR-VARX with p_α and generate forecast
         - Store forecast in 3D tensor [:, d, α_idx]

    2. Select optimal α via validation RMSE
    3. Extract forecasts at optimal α

    The model uses Double Machine Learning to remove confounding effects,
    then applies significance testing to select lag order for each α.

    Args:
        Y: Asset returns, shape (n_days, n_assets)
        W: Confounders, shape (n_days, n_confounders)
        alpha_grid: Significance levels (default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        p_max: Maximum lag order to consider (default: 10)
        config: GridConfig instance (default: None, uses default GridConfig)
                The lookback is derived from config.lookback_orvarx (1018 by default)
        validation_days: Number of days to use for validation (default: 20)
        asset_names: Names of assets (default: None, uses A1, A2, ...)
        confounder_names: Names of confounders (default: None, uses W1, W2, ...)
        dates: Date strings for forecast days (default: None, uses indices)
        learner_name: First-stage learner ('xgboost', 'lgbm', 'rf', 'extra_trees')
        n_jobs: Number of CPU cores (-1 for all, 5 recommended based on benchmarks)
        verbose: If True, print detailed progress (default: False)

    Returns:
        ORACLEVARXResult with forecasts, coefficients_all, alpha_optimal, p_optimal

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If DML estimation fails

    Example:
        >>> Y = torch.randn(1100, 5)  # 1100 days, 5 assets (~4+ years)
        >>> W = torch.randn(1100, 3)  # 1100 days, 3 confounders
        >>> result = fit_oracle_varx(Y, W, p_max=10)  # Uses GridConfig() default (lookback_orvarx=1018)
        >>> print(result.forecasts.shape)  # (5, n_test_days)
        >>> print(result.method)  # 'ORACLE-VARX'
        >>> print(result.alpha_optimal)  # Selected α indices for each day
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

    n_test_days = n_days - lookback
    if validation_days > n_test_days:
        raise ValueError(f"validation_days {validation_days} exceeds test days {n_test_days}")

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
        dates = [f"D{lookback + i}" for i in range(n_test_days)]
    elif len(dates) != n_test_days:
        raise ValueError(f"Expected {n_test_days} dates, got {len(dates)}")

    device = Y.device
    dtype = Y.dtype

    # Create model cache
    cache = ModelCache(n_assets=n_assets, n_confounders=n_confounders, p_max=p_max)

    # Convert to numpy for sklearn models
    Y_np = Y.cpu().numpy()
    W_np = W.cpu().numpy()

    # Initialize storage
    # forecasts_all: (n_assets, n_test_days, n_alphas)
    forecasts_all = torch.zeros(n_assets, n_test_days, n_alphas, device=device, dtype=dtype)

    # p_optimal_all: (n_test_days, n_alphas) - stores selected p for each (day, α)
    p_optimal_all = torch.zeros(n_test_days, n_alphas, device=device, dtype=torch.long)

    # coefficients_all: (n_test_days, n_alphas, p_max, n_assets, n_assets)
    coefficients_all = torch.zeros(
        n_test_days, n_alphas, p_max, n_assets, n_assets,
        device=device, dtype=dtype
    )

    # Main loop: for each test day and each alpha
    print(f"Fitting ORACLE-VARX for {n_test_days} days, {n_alphas} alphas, p_max={p_max}, learner={learner_name}")
    print(f"Grid config: train_size={config.train_size}, test_size={config.test_size}, lookback={lookback}")

    for day_rel_idx in range(n_test_days):
        day_idx = lookback + day_rel_idx

        if (day_rel_idx + 1) % 10 == 0 or day_rel_idx == 0:
            print(f"  Processing day {day_rel_idx + 1}/{n_test_days} (absolute idx: {day_idx})")

        for alpha_idx, alpha in enumerate(alpha_grid):
            try:
                # Step 1: Fit DML with p_max lags to get theta and SE
                forecast_pmax, theta_pmax, se_pmax, _ = fit_orvarx_single_day(
                    Y, W, p_max, day_idx, cache, config, learner_name, n_jobs, verbose
                )

                # Step 2: Run ORACLE significance test to get p_α
                p_alpha = oracle_significance_test(
                    theta_pmax, se_pmax, alpha, p_max, n_assets
                )

                # Store selected p for this (day, α)
                p_optimal_all[day_rel_idx, alpha_idx] = p_alpha

                # Step 3: Re-fit OR-VARX with p_α to get final forecast and coefficients
                if p_alpha == p_max:
                    # If p_α equals p_max, we can reuse the forecast from step 1
                    forecast = forecast_pmax
                    theta = theta_pmax
                else:
                    # Otherwise, re-fit with p_α
                    forecast, theta, _, _ = fit_orvarx_single_day(
                        Y, W, p_alpha, day_idx, cache, config, learner_name, n_jobs, verbose
                    )

                # Store forecast
                forecasts_all[:, day_rel_idx, alpha_idx] = forecast

                # Store coefficients
                # theta has shape (n_treatments, n_assets) = (n_assets * p_alpha, n_assets)
                # Reshape to (p_alpha, n_assets, n_assets)
                theta_reshaped = theta.view(p_alpha, n_assets, n_assets)
                coefficients_all[day_rel_idx, alpha_idx, :p_alpha, :, :] = theta_reshaped

                if verbose and ((day_rel_idx + 1) % 10 == 0 or day_rel_idx == 0):
                    if alpha_idx == 0:
                        print(f"    α={alpha:.3f}: p={p_alpha}")

            except Exception as e:
                print(f"    Warning: Failed to fit α={alpha:.3f} for day {day_rel_idx}: {e}")
                # Use zeros for this (day, α) combination and set p=1 as fallback
                p_optimal_all[day_rel_idx, alpha_idx] = 1

    # Extract actual returns for test period
    actuals = Y[lookback:, :]  # (n_test_days, n_assets)
    actuals = actuals.T  # (n_assets, n_test_days) to match forecasts_all

    # Select optimal α based on validation RMSE
    val_start = 0
    val_end = validation_days

    # Compute RMSE for each α
    # forecasts_all: (n_assets, n_test_days, n_alphas)
    # actuals: (n_assets, n_test_days)
    forecasts_val = forecasts_all[:, val_start:val_end, :]  # (n_assets, val_days, n_alphas)
    actuals_val = actuals[:, val_start:val_end]  # (n_assets, val_days)

    # Compute errors and RMSE per α
    # errors: (n_assets, val_days, n_alphas)
    errors = forecasts_val - actuals_val.unsqueeze(2)
    rmse_per_alpha = torch.sqrt((errors ** 2).mean(dim=(0, 1)))  # (n_alphas,)

    # Select α with minimum RMSE
    alpha_optimal_idx = torch.argmin(rmse_per_alpha).item()
    alpha_optimal_value = alpha_grid[alpha_optimal_idx]

    # Create alpha_optimal tensor: (n_test_days,) with constant value
    alpha_optimal = torch.full((n_test_days,), alpha_optimal_idx, device=device, dtype=torch.long)

    print(f"  Optimal α: {alpha_optimal_value:.3f} (idx={alpha_optimal_idx}, RMSE: {rmse_per_alpha[alpha_optimal_idx]:.6f})")

    # Print RMSE for all alphas for comparison
    print(f"  RMSE per α:")
    for idx, (alpha_val, rmse_val) in enumerate(zip(alpha_grid, rmse_per_alpha)):
        marker = " *" if idx == alpha_optimal_idx else ""
        print(f"    α={alpha_val:.3f}: RMSE={rmse_val:.6f}{marker}")

    # Extract forecasts at optimal α
    forecasts = forecasts_all[:, :, alpha_optimal_idx]  # (n_assets, n_test_days)

    # Extract p_optimal from p_optimal_all using optimal α
    p_optimal = p_optimal_all[:, alpha_optimal_idx]  # (n_test_days,)

    # Print summary statistics for selected p values
    p_optimal_np = p_optimal.cpu().numpy()
    print(f"  Selected p statistics (at optimal α={alpha_optimal_value:.3f}):")
    print(f"    Mean: {p_optimal_np.mean():.2f}")
    print(f"    Median: {int(np.median(p_optimal_np))}")
    print(f"    Min: {p_optimal_np.min()}, Max: {p_optimal_np.max()}")
    print(f"    Mode: {int(stats.mode(p_optimal_np, keepdims=False).mode)}")

    return ORACLEVARXResult(
        forecasts=forecasts,
        forecasts_all=forecasts_all,
        p_optimal_all=p_optimal_all,
        alpha_optimal=alpha_optimal,
        p_optimal=p_optimal,
        alpha_grid=alpha_grid,
        coefficients_all=coefficients_all,
        asset_names=asset_names,
        confounder_names=confounder_names,
        dates=dates,
    )
