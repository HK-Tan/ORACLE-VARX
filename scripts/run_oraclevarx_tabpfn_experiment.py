#!/usr/bin/env python3
"""Run ORACLE-VARX experiment using TabPFN for nuisance function estimation.

TabPFN is a transformer-based model for tabular data that requires no hyperparameter
tuning. It works best with datasets under 10k samples (our training window is 504),
making it ideal for this use case.

Requirements:
    - GPU with CUDA support (TabPFN is optimized for GPU)
    - HuggingFace account with accepted TabPFN terms:
      https://huggingface.co/Prior-Labs/TabPFN
    - HF_TOKEN environment variable set with your HuggingFace token

This script:
1. Validates GPU availability and HF_TOKEN
2. Loads ETF returns + VIX confounder from data files
3. Runs fit_oraclevarx_batched() with TabPFN as the learner
4. Converts forecasts to PnL using multiple strategies
5. Saves results and generates plots

Parameters:
    - lookback_orvarx: 1018 days (504 tree_train + 504 ols_window + 10 p_max_offset via GridConfig)
    - validation_days: 21 days
    - p_max: 10
    - alpha_grid: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    - Assets: 9 ETFs (XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU)
    - Confounder: VIX only

Example:
    # Set HuggingFace token
    export HF_TOKEN=your_huggingface_token

    # Run experiment
    python3 scripts/run_oraclevarx_tabpfn_experiment.py --verbose

    # Run with limited data for testing
    python3 scripts/run_oraclevarx_tabpfn_experiment.py --n-days 1500 --verbose
"""

import os
import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable TabPFN's posthog telemetry (~200ms latency per fit() call)
os.environ['TABPFN_NO_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

class _FakePosthog:
    """Mock posthog to prevent any network calls."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['posthog'] = _FakePosthog()


def check_requirements():
    """Check that GPU and HuggingFace token are available."""
    import torch

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: TabPFN requires GPU but CUDA is not available.")
        print("Please run on a machine with GPU support.")
        sys.exit(1)

    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    # Check HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\nERROR: HF_TOKEN environment variable not set.")
        print("\nTo use TabPFN, you need to:")
        print("1. Create a HuggingFace account at https://huggingface.co/join")
        print("2. Accept TabPFN terms at https://huggingface.co/Prior-Labs/TabPFN")
        print("3. Create an access token at https://huggingface.co/settings/tokens")
        print("4. Set the token: export HF_TOKEN=your_token")
        sys.exit(1)

    print("HF_TOKEN environment variable is set")


def main(
    n_days: int = None,
    validation_days: int = 21,
    p_max: int = 10,
    alpha_grid: list = None,
    output_dir: str = "results/oraclevarx_tabpfn",
    experiment_name: str = None,
    show_plots: bool = True,
    verbose: bool = False,
):
    """Run ORACLE-VARX experiment with TabPFN.

    Args:
        n_days: Number of days to load. If None, loads all available data.
        validation_days: Number of days for optimal alpha selection.
        p_max: Maximum lag order to consider.
        alpha_grid: Significance levels for p-selection (default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]).
        output_dir: Directory to save results.
        experiment_name: Name for the experiment. If None, auto-generated.
        show_plots: Whether to display plots interactively.
        verbose: Whether to print detailed progress.
    """
    import pandas as pd
    import numpy as np
    import torch
    import time
    from datetime import datetime

    from src.data.loader import prepare_tensors
    from src.data.constants import ETFS
    from src.models import GridConfig
    from src.models.oracle_var_tabpfn import fit_oraclevarx_tabpfn
    from src.evaluation import (
        run_backtest,
        plot_strategy_comparison,
        plot_lag_analysis,
    )
    from src.models.coefficient_refit import refit_dml_coefficients_for_day_tabpfn, get_target_days
    from src.evaluation.plotting import plot_coefficient_evolution_per_p
    from src.evaluation.backtest import save_experiment_results
    from src.evaluation.plotting import print_performance_summary

    # Default alpha grid
    if alpha_grid is None:
        alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("=" * 80)
    print("ORACLE-VARX EXPERIMENT (TabPFN)")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"oraclevarx_tabpfn_{timestamp}"

    print(f"\nExperiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    print(f"Learner: TabPFN (GPU)")
    print(f"Alpha grid: {alpha_grid}")

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Loading Data")
    print("-" * 40)

    # Load ETF returns with SPY for benchmark, and VIX as confounder
    # Use CUDA device for tensor operations
    tickers = ETFS + ["SPY"]

    Y, W, dates, loaded_tickers = prepare_tensors(
        tickers=tickers,
        confounder_names=["VIX"],
        n_days=n_days,
        device="cuda",
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
    # Step 2: Run ORACLE-VARX Model
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Running ORACLE-VARX Model")
    print("-" * 40)

    config = GridConfig()
    lookback = config.lookback_orvarx
    n_test_days = len(dates) - lookback - validation_days

    print(f"  Configuration:")
    print(f"    lookback: {lookback}")
    print(f"    validation_days: {validation_days}")
    print(f"    p_max: {p_max}")
    print(f"    alpha_grid: {alpha_grid}")
    print(f"    learner: TabPFN (batched GPU)")
    print(f"    Expected output days: {n_test_days}")

    start_time = time.perf_counter()
    result = fit_oraclevarx_tabpfn(
        Y=Y_etf,
        W=W,
        alpha_grid=alpha_grid,
        p_max=p_max,
        config=config,
        validation_days=validation_days,
        asset_names=etf_tickers,
        confounder_names=["VIX"],
        dates=dates[lookback + validation_days:],  # Offset dates for output period (after validation)
        n_estimators=8,
        device='cuda',
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start_time

    print(f"\n  Results:")
    print(f"    Method: {result.method}")
    print(f"    Forecast shape: {result.forecasts.shape}")
    print(f"    Output days: {len(result.dates)}")
    print(f"    Computation time: {elapsed:.2f}s ({elapsed/60:.1f} min)")

    # alpha_optimal statistics
    alpha_optimal_np = result.alpha_optimal.cpu().numpy()
    alpha_values = np.array(alpha_grid)[alpha_optimal_np]
    print(f"\n  alpha_optimal statistics:")
    print(f"    Mean: {alpha_values.mean():.3f}")
    print(f"    Median: {np.median(alpha_values):.3f}")
    print(f"    Min: {alpha_values.min():.3f}, Max: {alpha_values.max():.3f}")
    unique, counts = np.unique(alpha_optimal_np, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"    α={alpha_grid[idx]:.2f}: {count} days ({100*count/len(alpha_optimal_np):.1f}%)")

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
        title=f"ORACLE-VARX (TabPFN): Optimal Lag (p) Over Time\np_max={p_max}, validation={validation_days} days",
        save_path=str(lag_plot_path),
        show_plot=show_plots,
    )
    print(f"  Saved: {lag_plot_path}")

    # Plot 2: Strategy comparison (market-adjusted)
    strategy_plot_path = experiment_dir / "strategy_comparison.png"
    plot_strategy_comparison(
        pnl_results=pnl_results,
        market_adjusted=True,
        title=f"ORACLE-VARX (TabPFN): Market-Adjusted Strategy Comparison\n{result.dates[0]} to {result.dates[-1]}",
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
        title=f"ORACLE-VARX (TabPFN): Strategy Comparison vs SPY\n{result.dates[0]} to {result.dates[-1]}",
        save_path=str(strategy_plot_raw_path),
        show_plot=show_plots,
    )
    print(f"  Saved: {strategy_plot_raw_path}")

    # Plot 4+: Per-p coefficient evolution for target days (TabPFN refit)
    targets = get_target_days(result)
    for label, result_day_idx in targets:
        abs_day_idx = lookback + validation_days + result_day_idx
        p_star = int(result.p_optimal[result_day_idx].item())
        date_str = result.dates[result_day_idx]

        per_p_coefs = refit_dml_coefficients_for_day_tabpfn(
            Y=Y_etf, W=W, day_idx=abs_day_idx, p_star=p_star,
            lookback=lookback, asset_names=etf_tickers, date=date_str,
            config=config, device='cuda',
        )

        save_path = experiment_dir / f"coefficient_evolution_{label}.png"
        plot_coefficient_evolution_per_p(
            per_p_coefs,
            title=f"ORACLE-VARX (TabPFN): Coefficient Evolution p*={p_star} ({date_str})",
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

    parser = argparse.ArgumentParser(
        description="Run ORACLE-VARX experiment with TabPFN (requires GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
  - GPU with CUDA support
  - HuggingFace token: export HF_TOKEN=your_token
  - Accept TabPFN terms: https://huggingface.co/Prior-Labs/TabPFN

Example:
  export HF_TOKEN=your_huggingface_token
  python scripts/run_oraclevarx_tabpfn_experiment.py --verbose
        """
    )
    parser.add_argument("--n-days", type=int, default=None, help="Number of days to load (default: all)")
    parser.add_argument("--validation-days", type=int, default=21, help="Validation period (default: 21)")
    parser.add_argument("--p-max", type=int, default=10, help="Maximum lag order (default: 10)")
    parser.add_argument("--alpha-grid", type=str, default=None,
                        help="Comma-separated alpha values (default: 0.01,0.05,0.10,0.15,0.20,0.25,0.30)")
    parser.add_argument("--output-dir", type=str, default="results/oraclevarx_tabpfn", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    # Check requirements before running
    print("Checking requirements...")
    check_requirements()
    print()

    # Parse alpha grid if provided
    alpha_grid = None
    if args.alpha_grid:
        alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",")]

    main(
        n_days=args.n_days,
        validation_days=args.validation_days,
        p_max=args.p_max,
        alpha_grid=alpha_grid,
        output_dir=args.output_dir,
        experiment_name=args.name,
        show_plots=not args.no_show,
        verbose=args.verbose,
    )
