"""PnL calculation utilities for trading strategy evaluation.

This module provides functions for calculating profit and loss (PnL) based on
forecasted returns and actual returns, using various trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_pnl(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    pnl_strategy: str = "weighted",
    percentile: float = 0.5,
    contrarian: bool = False,
    market_adjustment: bool = True,
    benchmark: str = "SPY",
    benchmark_returns: pd.Series = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate PnL based on forecasted and actual returns.

    Converts log return forecasts to trading positions and calculates
    cumulative returns based on the chosen strategy.

    Args:
        forecast_df: DataFrame of forecasted returns for each asset.
            Shape: (n_days, n_assets), with DatetimeIndex.
        actual_df: DataFrame of actual returns for each asset (log returns).
            Shape: (n_days, n_assets), with DatetimeIndex.
        pnl_strategy: Strategy for calculating positions:
            - "naive": Go long $1 on positive forecast, short $1 on negative.
            - "weighted": Weight positions by predicted return magnitude.
            - "top": Only trade assets with |forecast| above percentile threshold.
        percentile: For "top" strategy, the percentile threshold (e.g., 0.5 = top 50%).
        contrarian: If True, invert trading signals (bet against forecasts).
        market_adjustment: If True, subtract market (SPY) returns from portfolio returns.
        benchmark: Benchmark ticker for market adjustment (default: "SPY").
        benchmark_returns: Optional Series of benchmark log returns. If not provided,
            will look for benchmark column in actual_df.

    Returns:
        Tuple of (cumulative_returns, daily_returns_pct, positions):
            - cumulative_returns: Cumulative portfolio returns starting from 0
            - daily_returns_pct: Daily percentage returns
            - positions: Daily total position exposure (sum of absolute positions)

    Raises:
        KeyError: If benchmark not found and benchmark_returns not provided.

    Example:
        >>> forecasts = pd.DataFrame({"XLF": [0.01, -0.02], "XLE": [0.005, 0.01]})
        >>> actuals = pd.DataFrame({"XLF": [0.008, -0.015], "XLE": [0.003, 0.012]})
        >>> cumulative, daily, positions = calculate_pnl(forecasts, actuals, "naive")
    """
    # Convert log returns to simple returns
    # Simple return = exp(log_return) - 1
    simple_returns = np.exp(actual_df) - 1

    # Set trading direction: -1 for contrarian, 1 for normal
    direction = -1 if contrarian else 1

    if pnl_strategy == "naive":
        # Go long on positive forecasts, short on negative
        raw_positions = direction * np.sign(forecast_df)
        # Normalize so absolute positions sum to 1 each day
        row_abs_sum = raw_positions.abs().sum(axis=1).replace(0, 1)
        positions = raw_positions.div(row_abs_sum, axis=0)

    elif pnl_strategy == "weighted":
        # Weight by forecast magnitude
        row_abs_sum = forecast_df.abs().sum(axis=1).replace(0, 1)
        positions = direction * forecast_df.div(row_abs_sum, axis=0)

    elif pnl_strategy == "top":
        # Only trade assets above percentile threshold
        positions = pd.DataFrame(0, index=forecast_df.index, columns=forecast_df.columns)
        n_rows = forecast_df.shape[0]

        for i in range(n_rows):
            abs_val = forecast_df.iloc[i, :].abs()
            sorted_vals = abs_val.sort_values(ascending=False)
            cutoff_idx = int(abs_val.shape[0] * (1 - percentile))
            threshold = sorted_vals.iloc[cutoff_idx] if cutoff_idx < len(sorted_vals) else 0

            # Go long where forecast > threshold, short where forecast < -threshold
            positions.iloc[i, forecast_df.iloc[i, :] > threshold] = direction
            positions.iloc[i, forecast_df.iloc[i, :] < -threshold] = -direction

        # Normalize positions
        row_sums = positions.abs().sum(axis=1).replace(0, 1)
        positions = positions.div(row_sums, axis=0)

    else:
        raise ValueError(f"Unknown pnl_strategy: {pnl_strategy}. Use 'naive', 'weighted', or 'top'.")

    # Calculate daily portfolio returns
    daily_pnl = positions * simple_returns
    daily_portfolio_returns_pct = daily_pnl.sum(axis=1)

    # Market adjustment: subtract benchmark returns
    if market_adjustment:
        # Get benchmark returns
        if benchmark_returns is not None:
            benchmark_simple_returns = np.exp(benchmark_returns) - 1
        elif benchmark in simple_returns.columns:
            benchmark_simple_returns = simple_returns[benchmark]
        else:
            raise KeyError(
                f"Benchmark '{benchmark}' not found in returns data and benchmark_returns not provided. "
                f"Available columns: {list(simple_returns.columns)}"
            )

        # Subtract benchmark returns weighted by position exposure
        position_exposure = positions.sum(axis=1)
        daily_portfolio_returns_pct = daily_portfolio_returns_pct - position_exposure * benchmark_simple_returns

    # Calculate cumulative returns
    daily_portfolio_returns = daily_portfolio_returns_pct + 1
    cumulative_returns = daily_portfolio_returns.cumprod() - 1

    # Return cumulative returns, percentage returns, and total position exposure
    return cumulative_returns, daily_portfolio_returns_pct, positions.sum(axis=1)


def calculate_spy_returns(
    actual_df: pd.DataFrame,
    benchmark: str = "SPY",
    benchmark_returns: pd.Series = None,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate SPY benchmark returns for comparison.

    Args:
        actual_df: DataFrame of actual returns (log returns).
        benchmark: Benchmark ticker (default: "SPY").
        benchmark_returns: Optional Series of benchmark log returns.

    Returns:
        Tuple of (cumulative_returns, daily_returns_pct).

    Raises:
        KeyError: If benchmark not found and benchmark_returns not provided.
    """
    # Get benchmark returns
    if benchmark_returns is not None:
        benchmark_log = benchmark_returns
    elif benchmark in actual_df.columns:
        benchmark_log = actual_df[benchmark]
    else:
        raise KeyError(
            f"Benchmark '{benchmark}' not found in returns data. "
            f"Available columns: {list(actual_df.columns)}"
        )

    # Convert to simple returns
    benchmark_simple = np.exp(benchmark_log) - 1

    # Calculate cumulative returns
    cumulative = (1 + benchmark_simple).cumprod() - 1

    return cumulative, benchmark_simple
