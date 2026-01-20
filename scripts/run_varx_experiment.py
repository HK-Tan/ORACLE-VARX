#!/usr/bin/env python3
"""Run VARX experiment on ETF returns with VIX as exogenous input.

This script runs a VAR model with 10 variables (9 sector ETFs + VIX) but only
trades the 9 ETFs. VIX is included in the VAR dynamics to help forecast ETF
returns but is NOT a tradeable asset.

VIX is converted to log returns: ln(VIX_t / VIX_{t-1}) * 100

This differs from:
- Plain VAR: which doesn't include VIX at all
- OR-VARX: which orthogonalizes ETF returns with respect to VIX using DML

Steps:
1. Loads ETF returns + VIX log returns from data files
2. Runs fit_var() with 10 assets (9 ETFs + VIX)
3. Filters out VIX from forecasts (only 9 ETFs are tradeable)
4. Converts forecasts to PnL using multiple strategies
5. Saves results and generates plots

Parameters:
    - lookback_var: 514 days (504 ols_window + 10 p_max_offset via GridConfig)
    - validation_days: 21 days
    - p_max: 10
    - Model assets: 9 ETFs + VIX (10 variables in VAR)
    - Tradeable assets: 9 ETFs only (VIX excluded from trading)
"""

import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime

from src.data.loader import load_opcl_with_vix
from src.data.constants import ETFS_WITH_VIX
from src.models import fit_var, GridConfig
from src.evaluation import (
    run_backtest,
    plot_strategy_comparison,
    plot_lag_analysis,
)
from src.evaluation.backtest import save_experiment_results
from src.evaluation.plotting import print_performance_summary


def main(
    n_days: int = None,
    validation_days: int = 21,
    p_max: int = 10,
    device: str = "cpu",
    output_dir: str = "results/varx",
    experiment_name: str = None,
    show_plots: bool = True,
):
    """Run VARX experiment with VIX as exogenous input (not tradeable).

    Args:
        n_days: Number of days to load. If None, loads all available data.
        validation_days: Number of days for optimal p selection.
        p_max: Maximum lag order to consider.
        device: Device to run on ("cpu" or "cuda").
        output_dir: Directory to save results.
        experiment_name: Name for the experiment. If None, auto-generated.
        show_plots: Whether to display plots interactively.
    """
    print("=" * 80)
    print("VARX EXPERIMENT (9 ETFs + VIX as Exogenous Input)")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"varx_{timestamp}"

    print(f"\nExperiment: {experiment_name}")
    print(f"Output directory: {output_dir}")

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Loading Data")
    print("-" * 40)

    # Load ETF returns + VIX log returns + SPY for benchmark
    # First load without SPY to get our tradeable assets
    etf_df = load_opcl_with_vix(n_days=n_days)

    # Also load SPY separately for benchmark
    from src.data.loader import load_opcl_data
    spy_df = load_opcl_data(tickers=["SPY"], n_days=n_days + 1 if n_days else None)

    # Align SPY with ETF dates
    common_dates = etf_df.index.intersection(spy_df.index)
    etf_df = etf_df.loc[common_dates]
    spy_df = spy_df.loc[common_dates]

    # Create combined DataFrame with SPY
    combined_df = etf_df.copy()
    combined_df["SPY"] = spy_df["SPY"]

    # Extract dates and tickers
    dates = combined_df.index.strftime("%Y-%m-%d").tolist()
    all_tickers = combined_df.columns.tolist()
    asset_tickers = [t for t in all_tickers if t != "SPY"]  # 9 ETFs + VIX

    print(f"  Total days loaded: {len(dates)}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Tradeable assets: {asset_tickers}")
    print(f"  Shape: ({len(dates)}, {len(asset_tickers)})")

    # Convert to tensors
    Y_all = torch.from_numpy(combined_df.values.astype("float32")).to(device)

    # Extract SPY and remove from Y
    spy_idx = all_tickers.index("SPY")
    Y = torch.cat([Y_all[:, :spy_idx], Y_all[:, spy_idx + 1:]], dim=1)

    print(f"  Y shape (tradeable): {Y.shape}")

    # =========================================================================
    # Step 2: Run VARX Model
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Running VARX Model")
    print("-" * 40)

    config = GridConfig()
    lookback = config.lookback_var
    n_test_days = len(dates) - lookback - validation_days

    print(f"  Configuration:")
    print(f"    lookback: {lookback}")
    print(f"    validation_days: {validation_days}")
    print(f"    p_max: {p_max}")
    print(f"    Expected output days: {n_test_days}")

    start_time = time.perf_counter()
    result = fit_var(
        Y=Y,
        p_max=p_max,
        config=config,
        validation_days=validation_days,
        asset_names=asset_tickers,
        dates=dates[lookback:],  # Offset dates for output period
    )
    elapsed = time.perf_counter() - start_time

    print(f"\n  Results (full model with VIX):")
    print(f"    Method: {result.method}")
    print(f"    Forecast shape: {result.forecasts.shape}")
    print(f"    Output days: {len(result.dates)}")
    print(f"    Computation time: {elapsed:.2f}s")

    # -------------------------------------------------------------------------
    # Filter out VIX from result for backtesting (VIX is not tradeable)
    # -------------------------------------------------------------------------
    vix_idx = asset_tickers.index("VIX")
    etf_indices = [i for i in range(len(asset_tickers)) if i != vix_idx]
    etf_only_tickers = [asset_tickers[i] for i in etf_indices]

    # Create filtered result for backtesting (exclude VIX)
    from src.results import VARXResult
    result_for_backtest = VARXResult(
        forecasts=result.forecasts[etf_indices, :],
        forecasts_all=result.forecasts_all[etf_indices, :, :],
        p_optimal=result.p_optimal,
        p_max=result.p_max,
        coefficients=result.coefficients[:, :, etf_indices, :][:, :, :, etf_indices],
        asset_names=etf_only_tickers,
        confounder_names=["VIX"],  # Mark VIX as confounder since it's not traded
        dates=result.dates,
    )

    print(f"\n  Filtered for backtest (VIX excluded from trading):")
    print(f"    Tradeable assets: {etf_only_tickers}")
    print(f"    Forecast shape: {result_for_backtest.forecasts.shape}")

    # p_optimal statistics
    p_optimal_np = result.p_optimal.cpu().numpy()
    print(f"\n  p_optimal statistics:")
    print(f"    Mean: {p_optimal_np.mean():.2f}")
    print(f"    Median: {int(np.median(p_optimal_np))}")
    print(f"    Min: {p_optimal_np.min()}, Max: {p_optimal_np.max()}")
    print(f"    Unique values: {np.unique(p_optimal_np)}")

    # =========================================================================
    # Step 3: Calculate PnL
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 3: Calculating PnL")
    print("-" * 40)

    # Create actual returns DataFrame for the forecast period
    actual_returns = pd.DataFrame(
        combined_df.values,
        index=pd.to_datetime(dates),
        columns=all_tickers,
    )

    # Run backtest for all strategies (using filtered result without VIX)
    pnl_results = run_backtest(
        result=result_for_backtest,
        actual_returns=actual_returns,
        strategies=["naive", "weighted", "top_50", "top_25", "top_75"],
        market_adjustment=True,
        benchmark="SPY",
        include_spy=False,  # Not for market-adjusted results
    )

    # Also calculate non-market-adjusted results with SPY
    pnl_results_raw = run_backtest(
        result=result_for_backtest,
        actual_returns=actual_returns,
        strategies=["naive", "weighted", "top_50", "top_25", "top_75"],
        market_adjustment=False,
        benchmark="SPY",
        include_spy=True,
    )

    print("\n  Market-Adjusted Performance:")
    performance_df = print_performance_summary(pnl_results)

    print("\n  Raw Performance (with SPY Benchmark):")
    performance_df_raw = print_performance_summary(pnl_results_raw)

    # =========================================================================
    # Step 4: Generate Plots
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 4: Generating Plots")
    print("-" * 40)

    # Get experiment directory from save_experiment_results
    # Create it early so we can save plots there
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Lag analysis
    lag_plot_path = experiment_dir / "lag_analysis.png"
    plot_lag_analysis(
        p_optimal=result.p_optimal.cpu().numpy(),
        dates=pd.to_datetime(result.dates),
        title=f"VARX: Optimal Lag (p) Over Time\np_max={p_max}, validation={validation_days} days",
        save_path=str(lag_plot_path),
        show_plot=show_plots,
    )
    print(f"  Saved: {lag_plot_path}")

    # Plot 2: Strategy comparison (market-adjusted)
    strategy_plot_path = experiment_dir / "strategy_comparison.png"
    plot_strategy_comparison(
        pnl_results=pnl_results,
        market_adjusted=True,
        title=f"VARX: Market-Adjusted Strategy Comparison\n{result.dates[0]} to {result.dates[-1]}",
        save_path=str(strategy_plot_path),
        show_plot=show_plots,
    )
    print(f"  Saved: {strategy_plot_path}")

    # Plot 3: Strategy comparison with SPY (raw)
    strategy_plot_raw_path = experiment_dir / "strategy_comparison_raw.png"
    plot_strategy_comparison(
        pnl_results=pnl_results_raw,
        include_spy=True,
        market_adjusted=False,
        title=f"VARX: Strategy Comparison vs SPY\n{result.dates[0]} to {result.dates[-1]}",
        save_path=str(strategy_plot_raw_path),
        show_plot=show_plots,
    )
    print(f"  Saved: {strategy_plot_raw_path}")

    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 5: Saving Results")
    print("-" * 40)

    paths = save_experiment_results(
        result=result_for_backtest,  # Save filtered result (VIX excluded)
        pnl_results=pnl_results,
        performance_df=performance_df,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    for file_type, path in paths.items():
        print(f"  {file_type}: {path}")

    # Also save raw performance
    raw_perf_path = experiment_dir / "performance_raw.csv"
    performance_df_raw.to_csv(raw_perf_path, index=False)
    print(f"  performance_raw: {raw_perf_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return result_for_backtest, pnl_results, performance_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run VARX experiment (9 ETFs with VIX as exogenous input)")
    parser.add_argument("--n-days", type=int, default=None, help="Number of days to load (default: all)")
    parser.add_argument("--validation-days", type=int, default=21, help="Validation period (default: 21)")
    parser.add_argument("--p-max", type=int, default=10, help="Maximum lag order (default: 10)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="results/varx", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")

    args = parser.parse_args()

    main(
        n_days=args.n_days,
        validation_days=args.validation_days,
        p_max=args.p_max,
        device=args.device,
        output_dir=args.output_dir,
        experiment_name=args.name,
        show_plots=not args.no_show,
    )
