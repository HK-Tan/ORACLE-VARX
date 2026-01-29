#!/usr/bin/env python3
"""Run OR-VARX experiment on ETF returns with VIX confounder.

This script:
1. Loads ETF returns + VIX confounder from data files
2. Runs fit_orvarx() with specified parameters and learner
3. Converts forecasts to PnL using multiple strategies
4. Saves results and generates plots

Parameters:
    - lookback_orvarx: 1018 days (504 tree_train + 504 ols_window + 10 p_max_offset via GridConfig)
    - validation_days: 21 days
    - p_max: 10
    - Assets: 9 ETFs (XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU)
    - Confounder: VIX only

Example:
    Run with 10 CPU cores and verbose output to see training progress:

        python3 scripts/run_orvarx_experiment.py --n-jobs 10 --verbose

    The --verbose flag shows per-fold training times, which is useful since
    the pre-training step (Step 1) trains ~2000+ models and can take 10-30 minutes.

    For LGBM, use one less job than your physical core count (e.g., 10 jobs on
    an 11-core machine) to leave headroom for LGBM's internal threading.
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

from src.data.loader import prepare_tensors
from src.data.constants import ETFS
from src.models import fit_orvarx_batched, GridConfig
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
    learner_name: str = "lgbm",
    n_jobs: int = 10, # -1 for all cores minus 1 (avoids LightGBM issues)
    device: str = "cpu",
    output_dir: str = "results/orvarx",
    experiment_name: str = None,
    show_plots: bool = True,
    verbose: bool = False,
):
    """Run OR-VARX experiment.

    Args:
        n_days: Number of days to load. If None, loads all available data.
        validation_days: Number of days for optimal p selection.
        p_max: Maximum lag order to consider.
        learner_name: First-stage learner ('xgboost', 'lgbm', 'rf', 'extra_trees').
        n_jobs: Number of CPU cores for parallel processing (-1 for all cores minus 1, avoids LightGBM issues).
        device: Device to run on ("cpu" or "cuda").
        output_dir: Directory to save results.
        experiment_name: Name for the experiment. If None, auto-generated.
        show_plots: Whether to display plots interactively.
        verbose: Whether to print detailed progress.
    """
    print("=" * 80)
    print("OR-VARX EXPERIMENT")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"orvarx_{learner_name}_{timestamp}"

    print(f"\nExperiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    print(f"Learner: {learner_name}")

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Loading Data")
    print("-" * 40)

    # Load ETF returns with SPY for benchmark, and VIX as confounder
    tickers = ETFS + ["SPY"]

    Y, W, dates, loaded_tickers = prepare_tensors(
        tickers=tickers,
        confounder_names=["VIX"],
        n_days=n_days,
        device=device,
    )

    # Extract SPY and remove from Y
    spy_idx = loaded_tickers.index("SPY")
    Y_etf = torch.cat([Y[:, :spy_idx], Y[:, spy_idx + 1:]], dim=1)
    etf_tickers = [t for t in loaded_tickers if t != "SPY"]

    print(f"  Total days loaded: {len(dates)}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Assets: {etf_tickers}")
    print(f"  Y shape: {Y_etf.shape}")
    print(f"  W shape (VIX): {W.shape}")

    # =========================================================================
    # Step 2: Run OR-VARX Model
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Running OR-VARX Model")
    print("-" * 40)

    config = GridConfig()
    lookback = config.lookback_orvarx
    n_test_days = len(dates) - lookback - validation_days

    print(f"  Configuration:")
    print(f"    lookback: {lookback}")
    print(f"    validation_days: {validation_days}")
    print(f"    p_max: {p_max}")
    print(f"    learner: {learner_name}")
    print(f"    n_jobs: {n_jobs}")
    print(f"    Expected output days: {n_test_days}")

    start_time = time.perf_counter()
    result = fit_orvarx_batched(
        Y=Y_etf,
        W=W,
        p_max=p_max,
        config=config,
        validation_days=validation_days,
        asset_names=etf_tickers,
        confounder_names=["VIX"],
        dates=dates[lookback + validation_days:],  # Offset dates for output period (after validation)
        learner_name=learner_name,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start_time

    print(f"\n  Results:")
    print(f"    Method: {result.method}")
    print(f"    Is orthogonalized: {result.is_orthogonalized}")
    print(f"    Forecast shape: {result.forecasts.shape}")
    print(f"    Output days: {len(result.dates)}")
    print(f"    Computation time: {elapsed:.2f}s ({elapsed/60:.1f} min)")

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
    # Need to include SPY for market adjustment
    actual_returns = pd.DataFrame(
        Y.cpu().numpy(),
        index=pd.to_datetime(dates),
        columns=loaded_tickers,
    )

    # Run backtest for all strategies
    pnl_results = run_backtest(
        result=result,
        actual_returns=actual_returns,
        strategies=["naive", "weighted", "top_50", "top_25", "top_75"],
        market_adjustment=True,
        benchmark="SPY",
        include_spy=False,  # Not for market-adjusted results
    )

    # Also calculate non-market-adjusted results with SPY
    pnl_results_raw = run_backtest(
        result=result,
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

    # Create experiment directory for plots (same as save_experiment_results will use)
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Lag analysis
    lag_plot_path = experiment_dir / "lag_analysis.png"
    plot_lag_analysis(
        p_optimal=result.p_optimal.cpu().numpy(),
        dates=pd.to_datetime(result.dates),
        title=f"OR-VARX ({learner_name}): Optimal Lag (p) Over Time\np_max={p_max}, validation={validation_days} days",
        save_path=str(lag_plot_path),
        show_plot=show_plots,
    )
    print(f"  Saved: {lag_plot_path}")

    # Plot 2: Strategy comparison (market-adjusted)
    strategy_plot_path = experiment_dir / "strategy_comparison.png"
    plot_strategy_comparison(
        pnl_results=pnl_results,
        market_adjusted=True,
        title=f"OR-VARX ({learner_name}): Market-Adjusted Strategy Comparison\n{result.dates[0]} to {result.dates[-1]}",
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
        title=f"OR-VARX ({learner_name}): Strategy Comparison vs SPY\n{result.dates[0]} to {result.dates[-1]}",
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
        result=result,
        pnl_results=pnl_results,
        performance_df=performance_df,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    for file_type, path in paths.items():
        print(f"  {file_type}: {path}")

    # Also save raw performance (in experiment subfolder)
    raw_perf_path = experiment_dir / "performance_raw.csv"
    performance_df_raw.to_csv(raw_perf_path, index=False)
    print(f"  performance_raw: {raw_perf_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return result, pnl_results, performance_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OR-VARX experiment on ETF returns")
    parser.add_argument("--n-days", type=int, default=None, help="Number of days to load (default: all)")
    parser.add_argument("--validation-days", type=int, default=21, help="Validation period (default: 21)")
    parser.add_argument("--p-max", type=int, default=10, help="Maximum lag order (default: 10)")
    parser.add_argument("--learner", type=str, default="lgbm", choices=["xgboost", "lgbm", "rf", "extra_trees"],
                        help="First-stage learner (default: lgbm)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU cores (-1 for all cores minus 1 (default, avoids LightGBM issues))")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="results/orvarx", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    main(
        n_days=args.n_days,
        validation_days=args.validation_days,
        p_max=args.p_max,
        learner_name=args.learner,
        n_jobs=args.n_jobs,
        device=args.device,
        output_dir=args.output_dir,
        experiment_name=args.name,
        show_plots=not args.no_show,
        verbose=args.verbose,
    )
