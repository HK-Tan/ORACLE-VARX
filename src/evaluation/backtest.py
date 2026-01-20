"""Backtesting utilities for VAR and OR-VARX model results.

This module provides functions to convert model results to PnL calculations
and run comprehensive backtests across multiple strategies.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from src.results import VARXResult, ORACLEVARXResult
from src.evaluation.pnl import calculate_pnl, calculate_spy_returns


def run_backtest(
    result: Union[VARXResult, ORACLEVARXResult],
    actual_returns: pd.DataFrame,
    strategies: List[str] = None,
    market_adjustment: bool = True,
    benchmark: str = "SPY",
    benchmark_returns: pd.Series = None,
    include_spy: bool = True,
) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series]]:
    """Convert model forecasts to PnL for multiple strategies.

    This function bridges the gap between VARXResult/ORACLEVARXResult objects
    and PnL calculation, handling date alignment and format conversion.

    Args:
        result: VARXResult or ORACLEVARXResult from model fitting.
        actual_returns: DataFrame of actual log returns with DatetimeIndex.
            Must contain columns for all assets in result.asset_names.
            Should also contain benchmark column if market_adjustment=True.
        strategies: List of strategies to evaluate. Options:
            - "naive": Direction-based trading
            - "weighted": Magnitude-weighted positions
            - "top_50": Trade top 50% by |forecast|
            - "top_25": Trade top 25% by |forecast|
            - "top_75": Trade top 75% by |forecast|
            Default: ["naive", "weighted", "top_50", "top_25", "top_75"]
        market_adjustment: Whether to subtract benchmark returns.
        benchmark: Benchmark ticker (default: "SPY").
        benchmark_returns: Optional benchmark returns Series.
        include_spy: Whether to include SPY buy-and-hold as a strategy.

    Returns:
        Dictionary mapping strategy names to tuples of:
        (cumulative_returns, daily_returns_pct, positions)

    Raises:
        ValueError: If forecast dates don't match actual returns dates.
        KeyError: If required assets are missing from actual_returns.

    Example:
        >>> result = fit_var(Y, p_max=10, ...)
        >>> actual_df = pd.DataFrame(Y.numpy(), index=dates, columns=tickers)
        >>> pnl_results = run_backtest(result, actual_df)
        >>> cumulative_naive, daily_naive, _ = pnl_results["naive"]
    """
    if strategies is None:
        strategies = ["naive", "weighted", "top_50", "top_25", "top_75"]

    # Convert result forecasts to DataFrame
    forecast_df = result.to_dataframe()

    # Ensure index is DatetimeIndex
    if not isinstance(forecast_df.index, pd.DatetimeIndex):
        forecast_df.index = pd.to_datetime(forecast_df.index)

    # Verify all required assets are present
    missing_assets = set(result.asset_names) - set(actual_returns.columns)
    if missing_assets:
        raise KeyError(
            f"Missing assets in actual_returns: {missing_assets}. "
            f"Available columns: {list(actual_returns.columns)}"
        )

    # Align actual returns to forecast dates
    actual_aligned = actual_returns.loc[forecast_df.index, result.asset_names]

    # Get benchmark returns aligned to forecast dates
    if benchmark_returns is not None:
        benchmark_aligned = benchmark_returns.loc[forecast_df.index]
    elif benchmark in actual_returns.columns:
        benchmark_aligned = actual_returns.loc[forecast_df.index, benchmark]
    else:
        benchmark_aligned = None

    # Calculate PnL for each strategy
    pnl_results = {}

    strategy_configs = {
        "naive": {"pnl_strategy": "naive"},
        "weighted": {"pnl_strategy": "weighted"},
        "top_50": {"pnl_strategy": "top", "percentile": 0.5},
        "top_25": {"pnl_strategy": "top", "percentile": 0.25},
        "top_75": {"pnl_strategy": "top", "percentile": 0.75},
    }

    for strategy_name in strategies:
        if strategy_name not in strategy_configs:
            raise ValueError(f"Unknown strategy: {strategy_name}. Options: {list(strategy_configs.keys())}")

        config = strategy_configs[strategy_name]
        cumulative, daily_pct, positions = calculate_pnl(
            forecast_df=forecast_df,
            actual_df=actual_aligned,
            market_adjustment=market_adjustment,
            benchmark=benchmark,
            benchmark_returns=benchmark_aligned,
            **config,
        )
        pnl_results[strategy_name] = (cumulative, daily_pct, positions)

    # Add SPY buy-and-hold if requested
    if include_spy and not market_adjustment:
        if benchmark_aligned is not None:
            spy_cumulative, spy_daily = calculate_spy_returns(
                actual_df=actual_aligned,
                benchmark=benchmark,
                benchmark_returns=benchmark_aligned,
            )
            # Create dummy positions series (always 1 for buy-and-hold)
            spy_positions = pd.Series(1.0, index=forecast_df.index)
            pnl_results["spy"] = (spy_cumulative, spy_daily, spy_positions)

    return pnl_results


def save_experiment_results(
    result: Union[VARXResult, ORACLEVARXResult],
    pnl_results: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    performance_df: pd.DataFrame,
    output_dir: Union[str, Path],
    experiment_name: str,
) -> Dict[str, Path]:
    """Save experiment results to disk in a per-experiment subfolder.

    Creates a subfolder named after the experiment and saves model results,
    PnL data, and performance metrics inside it.

    Args:
        result: VARXResult or ORACLEVARXResult from model fitting.
        pnl_results: Dictionary of PnL results from run_backtest.
        performance_df: DataFrame of performance metrics from print_performance_summary.
        output_dir: Base directory for results (e.g., "results/var").
        experiment_name: Name for the experiment (used as subfolder name).

    Returns:
        Dictionary mapping file types to their paths:
        - "experiment_dir": Path to experiment subfolder
        - "model_results": Path to saved VARXResult (.pt file)
        - "pnl_results": Path to saved PnL results (.pkl file)
        - "performance": Path to saved performance CSV

    Example:
        >>> paths = save_experiment_results(result, pnl_results, perf_df, "results/var", "var_20260119_123456")
        >>> print(f"Saved to: {paths['experiment_dir']}")
        # Creates: results/var/var_20260119_123456/results.pt, etc.
    """
    output_dir = Path(output_dir)

    # Create experiment subfolder
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    paths = {"experiment_dir": experiment_dir}

    # Save model results using built-in save method (no prefix)
    model_path = experiment_dir / "results.pt"
    result.save(str(model_path))
    paths["model_results"] = model_path

    # Save PnL results as pickle (no prefix)
    pnl_path = experiment_dir / "pnl_results.pkl"
    with open(pnl_path, "wb") as f:
        pickle.dump(pnl_results, f)
    paths["pnl_results"] = pnl_path

    # Save performance metrics as CSV (no prefix)
    perf_path = experiment_dir / "performance.csv"
    performance_df.to_csv(perf_path, index=False)
    paths["performance"] = perf_path

    return paths


def load_experiment_results(
    output_dir: Union[str, Path],
    experiment_name: str,
) -> Tuple[Union[VARXResult, ORACLEVARXResult], Dict, pd.DataFrame]:
    """Load experiment results from disk.

    Loads results from the per-experiment subfolder structure.

    Args:
        output_dir: Base directory containing experiment subfolders (e.g., "results/var").
        experiment_name: Name of the experiment (subfolder name).

    Returns:
        Tuple of (result, pnl_results, performance_df).

    Raises:
        FileNotFoundError: If any required files are missing.

    Example:
        >>> result, pnl, perf = load_experiment_results("results/var", "var_20260119_123456")
        # Loads from: results/var/var_20260119_123456/results.pt, etc.
    """
    output_dir = Path(output_dir)
    experiment_dir = output_dir / experiment_name

    # Load model results (no prefix in filename)
    model_path = experiment_dir / "results.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model results not found: {model_path}")

    # Try loading as VARXResult first, then ORACLEVARXResult
    try:
        result = VARXResult.load(str(model_path))
    except ValueError:
        result = ORACLEVARXResult.load(str(model_path))

    # Load PnL results (no prefix in filename)
    pnl_path = experiment_dir / "pnl_results.pkl"
    if not pnl_path.exists():
        raise FileNotFoundError(f"PnL results not found: {pnl_path}")
    with open(pnl_path, "rb") as f:
        pnl_results = pickle.load(f)

    # Load performance metrics (no prefix in filename)
    perf_path = experiment_dir / "performance.csv"
    if not perf_path.exists():
        raise FileNotFoundError(f"Performance metrics not found: {perf_path}")
    performance_df = pd.read_csv(perf_path)

    return result, pnl_results, performance_df
