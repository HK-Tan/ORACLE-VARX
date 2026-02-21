#!/usr/bin/env python3
"""Run ACLE-VARX experiment on ETF returns.

This script:
1. Loads ETF returns from data files
2. Runs fit_aclevarx() with specified parameters
3. Converts forecasts to PnL using multiple strategies
4. Saves results and generates plots

ACLE-VARX applies the same significance-testing methodology as ORACLE-VARX but
to plain VAR models (without DML/orthogonalization). It uses:
- Significance-based p-selection (like ORACLE-VARX)
- α-selection via validation RMSE (like ORACLE-VARX)
- Plain OLS coefficients (like VAR, no confounders)

Parameters:
    - lookback_var: 514 days (504 ols_window + 10 p_max_offset via GridConfig)
    - validation_days: 21 days
    - p_max: 10
    - alpha_grid: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    - Assets: 9 ETFs (XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU)
    - No confounders (plain VAR)

Example:
    Run with verbose output:

        python3 scripts/run_aclevarx_experiment.py --verbose

    Run with smaller dataset for quick testing:

        python3 scripts/run_aclevarx_experiment.py --n-days 600 --verbose
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

from src.data.loader import load_opcl_with_vix, load_opcl_data
from src.data.constants import ETFS
from src.models import GridConfig
from src.models.acle_var import fit_aclevarx
from src.results import ACLEVARXResult
from src.evaluation import (
    run_backtest,
    plot_strategy_comparison,
    plot_lag_analysis,
)
from src.models.coefficient_refit import refit_var_coefficients_for_day, get_target_days
from src.evaluation.plotting import plot_coefficient_evolution_per_p
from src.evaluation.backtest import save_experiment_results
from src.evaluation.plotting import print_performance_summary


def main(
    n_days: int = None,
    validation_days: int = 21,
    p_max: int = 10,
    alpha_grid: list = None,
    device: str = "cpu",
    output_dir: str = "results/aclevarx",
    experiment_name: str = None,
    show_plots: bool = True,
    verbose: bool = False,
):
    """Run ACLE-VARX experiment.

    Args:
        n_days: Number of days to load. If None, loads all available data.
        validation_days: Number of days for optimal alpha selection.
        p_max: Maximum lag order to consider.
        alpha_grid: Significance levels for p-selection (default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]).
        device: Device to run on ("cpu" or "cuda").
        output_dir: Directory to save results.
        experiment_name: Name for the experiment. If None, auto-generated.
        show_plots: Whether to display plots interactively.
        verbose: Whether to print detailed progress.
    """
    # Default alpha grid
    if alpha_grid is None:
        alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("=" * 80)
    print("ACLE-VARX EXPERIMENT")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"aclevarx_{timestamp}"

    print(f"\nExperiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    print(f"Alpha grid: {alpha_grid}")

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Loading Data")
    print("-" * 40)

    # Load ETF returns + VIX log returns (VIX included as endogenous variable)
    etf_df = load_opcl_with_vix(n_days=n_days)

    # Also load SPY separately for benchmark
    spy_df = load_opcl_data(tickers=["SPY"], n_days=n_days + 1 if n_days else None)

    # Align SPY with ETF dates
    common_dates = etf_df.index.intersection(spy_df.index)
    etf_df = etf_df.loc[common_dates]
    spy_df = spy_df.loc[common_dates]

    # Create combined DataFrame with SPY (for benchmarking only)
    combined_df = etf_df.copy()
    combined_df["SPY"] = spy_df["SPY"]

    # Extract dates and tickers
    dates = combined_df.index.strftime("%Y-%m-%d").tolist()
    all_tickers = combined_df.columns.tolist()
    model_tickers = [t for t in all_tickers if t != "SPY"]  # 9 ETFs + VIX (10 variables in VAR)

    print(f"  Total days loaded: {len(dates)}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Model assets (including VIX): {model_tickers}")
    print(f"  Shape: ({len(dates)}, {len(model_tickers)})")

    # Convert to tensors
    Y_all = torch.from_numpy(combined_df.values.astype("float32")).to(device)

    # Extract SPY and remove from Y (keep VIX in Y for VAR)
    spy_idx = all_tickers.index("SPY")
    Y_model = torch.cat([Y_all[:, :spy_idx], Y_all[:, spy_idx + 1:]], dim=1)

    print(f"  Y_model shape: {Y_model.shape}")

    # =========================================================================
    # Step 2: Run ACLE-VARX Model
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Running ACLE-VARX Model")
    print("-" * 40)

    config = GridConfig()
    lookback = config.lookback_var  # Uses VAR lookback (514), not OR-VARX lookback (1018)
    n_test_days = len(dates) - lookback - validation_days

    print(f"  Configuration:")
    print(f"    lookback: {lookback}")
    print(f"    validation_days: {validation_days}")
    print(f"    p_max: {p_max}")
    print(f"    alpha_grid: {alpha_grid}")
    print(f"    Expected output days: {n_test_days}")

    start_time = time.perf_counter()
    result = fit_aclevarx(
        Y=Y_model,  # 10 variables (9 ETFs + VIX)
        alpha_grid=alpha_grid,
        p_max=p_max,
        config=config,
        validation_days=validation_days,
        asset_names=model_tickers,
        dates=dates[lookback + validation_days:],  # Offset dates for output period (after validation)
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start_time

    print(f"\n  Results (full model with VIX):")
    print(f"    Method: {result.method}")
    print(f"    Forecast shape: {result.forecasts.shape}")
    print(f"    Output days: {len(result.dates)}")
    print(f"    Computation time: {elapsed:.2f}s ({elapsed/60:.1f} min)")

    # -------------------------------------------------------------------------
    # Filter out VIX from result for backtesting (VIX is not tradeable)
    # -------------------------------------------------------------------------
    vix_idx = model_tickers.index("VIX")
    etf_indices = [i for i in range(len(model_tickers)) if i != vix_idx]
    etf_only_tickers = [model_tickers[i] for i in etf_indices]

    # Create filtered result for backtesting (exclude VIX)
    result_for_backtest = ACLEVARXResult(
        forecasts=result.forecasts[etf_indices, :],
        forecasts_all=result.forecasts_all[etf_indices, :, :],
        p_optimal_all=result.p_optimal_all,
        alpha_optimal=result.alpha_optimal,
        p_optimal=result.p_optimal,
        alpha_grid=result.alpha_grid,
        asset_names=etf_only_tickers,
        confounder_names=["VIX"],  # VIX included in VAR but not tradeable
        dates=result.dates,
        SE_all=result.SE_all,
    )

    print(f"\n  Filtered for backtest (VIX excluded from trading):")
    print(f"    Tradeable assets: {etf_only_tickers}")
    print(f"    Forecast shape: {result_for_backtest.forecasts.shape}")
    print(f"    Confounder names: {result_for_backtest.confounder_names}")

    # alpha_optimal statistics
    alpha_optimal_np = result_for_backtest.alpha_optimal.cpu().numpy()
    alpha_values = np.array(alpha_grid)[alpha_optimal_np]
    print(f"\n  alpha_optimal statistics:")
    print(f"    Mean: {alpha_values.mean():.3f}")
    print(f"    Median: {np.median(alpha_values):.3f}")
    print(f"    Min: {alpha_values.min():.3f}, Max: {alpha_values.max():.3f}")
    unique, counts = np.unique(alpha_optimal_np, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"    α={alpha_grid[idx]:.2f}: {count} days ({100*count/len(alpha_optimal_np):.1f}%)")

    # p_optimal statistics
    p_optimal_np = result_for_backtest.p_optimal.cpu().numpy()
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

    # Create experiment directory for plots (same as save_experiment_results will use)
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Lag analysis
    lag_plot_path = experiment_dir / "lag_analysis.png"
    plot_lag_analysis(
        p_optimal=result_for_backtest.p_optimal.cpu().numpy(),
        dates=pd.to_datetime(result_for_backtest.dates),
        title=f"ACLE-VARX: Optimal Lag (p) Over Time\np_max={p_max}, validation={validation_days} days",
        save_path=str(lag_plot_path),
        show_plot=show_plots,
    )
    print(f"  Saved: {lag_plot_path}")

    # Plot 2: Strategy comparison (market-adjusted)
    strategy_plot_path = experiment_dir / "strategy_comparison.png"
    plot_strategy_comparison(
        pnl_results=pnl_results,
        market_adjusted=True,
        title=f"ACLE-VARX: Market-Adjusted Strategy Comparison\n{result_for_backtest.dates[0]} to {result_for_backtest.dates[-1]}",
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
        title=f"ACLE-VARX: Strategy Comparison vs SPY\n{result_for_backtest.dates[0]} to {result_for_backtest.dates[-1]}",
        save_path=str(strategy_plot_raw_path),
        show_plot=show_plots,
    )
    print(f"  Saved: {strategy_plot_raw_path}")

    # Plot 4+: Per-p coefficient evolution for target days
    targets = get_target_days(result)
    for label, result_day_idx in targets:
        abs_day_idx = lookback + validation_days + result_day_idx
        p_star = int(result.p_optimal[result_day_idx].item())
        date_str = result.dates[result_day_idx]

        per_p_coefs = refit_var_coefficients_for_day(
            Y=Y_model, day_idx=abs_day_idx, p_star=p_star,
            lookback=lookback, asset_names=model_tickers, date=date_str,
        )

        save_path = experiment_dir / f"coefficient_evolution_{label}.png"
        plot_coefficient_evolution_per_p(
            per_p_coefs,
            title=f"ACLE-VARX: Coefficient Evolution p*={p_star} ({date_str})",
            save_path=str(save_path), show_plot=show_plots,
        )
        print(f"  Saved: {save_path}")

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

    # Also save raw performance (in experiment subfolder)
    raw_perf_path = experiment_dir / "performance_raw.csv"
    performance_df_raw.to_csv(raw_perf_path, index=False)
    print(f"  performance_raw: {raw_perf_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return result_for_backtest, pnl_results, performance_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ACLE-VARX experiment on ETF returns")
    parser.add_argument("--n-days", type=int, default=None, help="Number of days to load (default: all)")
    parser.add_argument("--validation-days", type=int, default=21, help="Validation period (default: 21)")
    parser.add_argument("--p-max", type=int, default=10, help="Maximum lag order (default: 10)")
    parser.add_argument("--alpha-grid", type=str, default=None,
                        help="Comma-separated alpha values (default: 0.01,0.05,0.10,0.15,0.20,0.25,0.30)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="results/aclevarx", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    # Parse alpha grid if provided
    alpha_grid = None
    if args.alpha_grid:
        alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",")]

    main(
        n_days=args.n_days,
        validation_days=args.validation_days,
        p_max=args.p_max,
        alpha_grid=alpha_grid,
        device=args.device,
        output_dir=args.output_dir,
        experiment_name=args.name,
        show_plots=not args.no_show,
        verbose=args.verbose,
    )
