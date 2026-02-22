#!/usr/bin/env python3
"""Combined experiment runner.

Three modes of operation:

  1. --no-confounders: VAR + ACLE-VAR baseline (OLS, no confounders)
  2. --confounders <preset> --ols-only: VARX + ACLE-VARX (OLS with confounders)
  3. --confounders <preset> --learner <learner>: OR-VARX + ORACLE-VARX (DML)

Modes 1-2 are used by Phase 0 of the orchestrator. Mode 3 is used by Phases 1-3.
OR-VARX and ORACLE-VARX share the same expensive DML first stage
(fit_orvarx_core), so this script runs it once and reuses the results.

Usage:
    # Phase 0: OLS baselines
    python scripts/run_combined_experiment.py --no-confounders --no-show
    python scripts/run_combined_experiment.py --confounders vix --ols-only --no-show

    # Phases 1-3: DML methods
    python scripts/run_combined_experiment.py --confounders vix --learner lgbm --no-show

    # Quick smoke test
    python scripts/run_combined_experiment.py --confounders vix --learner lgbm --n-days 1500 --no-show
"""

import sys
import os
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime

from src.data.constants import ETFS, CONFOUNDER_PRESETS
from src.data.loader import (
    load_opcl_data,
    load_opcl_with_confounders,
    prepare_tensors,
)
from src.models import fit_var, fit_orvarx_batched, GridConfig
from src.models.dml_pytorch import fit_orvarx_core, get_physical_cpu_count
from src.models.oracle_var import fit_oraclevarx_batched
from src.models.acle_var import fit_aclevarx
from src.results import VARXResult, ACLEVARXResult
from src.evaluation import (
    run_backtest,
    plot_strategy_comparison,
    plot_lag_analysis,
)
from src.evaluation.backtest import save_experiment_results
from src.evaluation.plotting import print_performance_summary


def resolve_confounders(confounders_arg: str) -> list[str]:
    """Resolve confounder argument to list of names."""
    if confounders_arg in CONFOUNDER_PRESETS:
        return CONFOUNDER_PRESETS[confounders_arg]
    return [s.strip() for s in confounders_arg.split(",")]



def compute_default_n_jobs() -> int:
    """Compute default n_jobs for parallel tmux panes.

    When running 4 panes simultaneously on the same machine:
    n_jobs = max(1, (physical_cores - 1) // 4)

    This leaves headroom for OS, LightGBM internal threading, and memory bus.
    """
    physical_cores = get_physical_cpu_count()
    n_jobs = max(1, (physical_cores - 1) // 4)
    return n_jobs


def run_backtest_and_save(
    result,
    Y_full,
    dates,
    all_tickers,
    method_name: str,
    conf_label: str,
    learner_label: str,
    output_dir: Path,
    show_plots: bool,
):
    """Run backtest, generate plots, and save results for a single method."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{method_name}_{conf_label}_{learner_label}_{timestamp}"
    method_output_dir = output_dir / method_name

    # Create actual returns DataFrame
    actual_returns = pd.DataFrame(
        Y_full.cpu().numpy() if isinstance(Y_full, torch.Tensor) else Y_full,
        index=pd.to_datetime(dates),
        columns=all_tickers,
    )

    # Market-adjusted backtest
    pnl_results = run_backtest(
        result=result,
        actual_returns=actual_returns,
        strategies=["naive", "weighted", "top_50", "top_25", "top_75"],
        market_adjustment=True,
        benchmark="SPY",
        include_spy=False,
    )

    # Raw backtest with SPY
    pnl_results_raw = run_backtest(
        result=result,
        actual_returns=actual_returns,
        strategies=["naive", "weighted", "top_50", "top_25", "top_75"],
        market_adjustment=False,
        benchmark="SPY",
        include_spy=True,
    )

    print(f"\n  {method_name} Market-Adjusted Performance:")
    performance_df = print_performance_summary(pnl_results)

    print(f"\n  {method_name} Raw Performance (with SPY Benchmark):")
    performance_df_raw = print_performance_summary(pnl_results_raw)

    # Create experiment directory for plots
    experiment_dir = method_output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = f"{method_name.upper()} ({conf_label}/{learner_label})"

    # Plot lag analysis
    lag_plot_path = experiment_dir / "lag_analysis.png"
    plot_lag_analysis(
        p_optimal=result.p_optimal.cpu().numpy(),
        dates=pd.to_datetime(result.dates),
        title=f"{title_prefix}: Optimal Lag (p) Over Time",
        save_path=str(lag_plot_path),
        show_plot=show_plots,
    )

    # Plot strategy comparison (market-adjusted)
    strategy_plot_path = experiment_dir / "strategy_comparison.png"
    plot_strategy_comparison(
        pnl_results=pnl_results,
        market_adjusted=True,
        title=f"{title_prefix}: Market-Adjusted Strategy Comparison\n{result.dates[0]} to {result.dates[-1]}",
        save_path=str(strategy_plot_path),
        show_plot=show_plots,
    )

    # Plot strategy comparison with SPY (raw)
    strategy_plot_raw_path = experiment_dir / "strategy_comparison_raw.png"
    plot_strategy_comparison(
        pnl_results=pnl_results_raw,
        include_spy=True,
        market_adjusted=False,
        title=f"{title_prefix}: Strategy Comparison vs SPY\n{result.dates[0]} to {result.dates[-1]}",
        save_path=str(strategy_plot_raw_path),
        show_plot=show_plots,
    )

    # Save results
    paths = save_experiment_results(
        result=result,
        pnl_results=pnl_results,
        performance_df=performance_df,
        output_dir=method_output_dir,
        experiment_name=experiment_name,
    )

    # Save raw performance
    raw_perf_path = experiment_dir / "performance_raw.csv"
    performance_df_raw.to_csv(raw_perf_path, index=False)

    print(f"  Saved to: {experiment_dir}")
    return result, pnl_results, performance_df


def main(
    n_days: int = None,
    confounders: str = "vix",
    no_confounders: bool = False,
    learner_name: str = "lgbm",
    validation_days: int = 21,
    p_max: int = 10,
    alpha_grid: list = None,
    n_jobs: int = None,
    device: str = "cpu",
    output_dir: str = "results",
    show_plots: bool = True,
    verbose: bool = False,
    ols_only: bool = False,
):
    """Run combined experiment for a given (confounder_config, learner) pair.

    Args:
        n_days: Number of days to load. If None, loads all available data.
        confounders: Confounder config: preset name (vix/macro5/all10) or comma-separated.
        no_confounders: If True, run VAR + ACLE-VAR baseline (no confounders).
        learner_name: First-stage learner for DML methods.
        validation_days: Validation period for p/alpha selection.
        p_max: Maximum lag order.
        alpha_grid: Significance levels for ACLE/ORACLE methods.
        n_jobs: CPU cores per method. If None, auto-computed for 4 parallel panes.
        device: PyTorch device.
        output_dir: Base results directory.
        show_plots: Whether to display plots interactively.
        verbose: Print detailed progress.
    """
    if alpha_grid is None:
        alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # Auto-compute n_jobs if not provided
    if n_jobs is None:
        n_jobs = compute_default_n_jobs()
        print(f"Auto-computed n_jobs={n_jobs} (physical_cores={get_physical_cpu_count()}, "
              f"formula: max(1, (cores-1)//4))")

    output_dir = Path(output_dir)
    config = GridConfig()
    total_start = time.perf_counter()

    if no_confounders:
        # =================================================================
        # NO-CONFOUNDERS MODE: VAR + ACLE-VAR baseline
        # =================================================================
        print("=" * 80)
        print("COMBINED EXPERIMENT: VAR + ACLE-VAR (No Confounders)")
        print("=" * 80)

        conf_label = "none"
        learner_label = "ols"

        # Load data: 9 ETFs + SPY
        tickers = ETFS + ["SPY"]
        Y, _, dates, loaded_tickers = prepare_tensors(
            tickers=tickers,
            confounder_names=None,
            n_days=n_days,
            device=device,
        )

        # Extract SPY
        spy_idx = loaded_tickers.index("SPY")
        Y_etf = torch.cat([Y[:, :spy_idx], Y[:, spy_idx + 1:]], dim=1)
        etf_tickers = [t for t in loaded_tickers if t != "SPY"]

        print(f"  Data: {len(dates)} days, {dates[0]} to {dates[-1]}")
        print(f"  Assets: {etf_tickers}")

        # --- VAR ---
        print("\n" + "=" * 60)
        print("  METHOD 1/2: VAR (Plain)")
        print("=" * 60)

        lookback_var = config.lookback_var
        var_start = time.perf_counter()
        var_result = fit_var(
            Y=Y_etf,
            p_max=p_max,
            config=config,
            validation_days=validation_days,
            asset_names=etf_tickers,
            dates=dates[lookback_var:],
        )
        var_elapsed = time.perf_counter() - var_start
        print(f"  VAR complete: {var_result.forecasts.shape[1]} output days, {var_elapsed:.1f}s")

        run_backtest_and_save(
            var_result, Y, dates, loaded_tickers,
            "var", conf_label, learner_label, output_dir, show_plots,
        )

        # --- ACLE-VAR ---
        print("\n" + "=" * 60)
        print("  METHOD 2/2: ACLE-VAR")
        print("=" * 60)

        acle_start = time.perf_counter()
        acle_result_full = fit_aclevarx(
            Y=Y_etf,
            alpha_grid=alpha_grid,
            p_max=p_max,
            config=config,
            validation_days=validation_days,
            asset_names=etf_tickers,
            dates=dates[lookback_var + validation_days:],
            verbose=verbose,
        )
        acle_elapsed = time.perf_counter() - acle_start
        print(f"  ACLE-VAR complete: {acle_result_full.forecasts.shape[1]} output days, {acle_elapsed:.1f}s")

        run_backtest_and_save(
            acle_result_full, Y, dates, loaded_tickers,
            "aclevar", conf_label, learner_label, output_dir, show_plots,
        )

    else:
        # =================================================================
        # CONFOUNDERS MODE
        # =================================================================
        conf_label = confounders
        confounder_names = resolve_confounders(confounders)
        learner_label = learner_name

        if ols_only:
            # =============================================================
            # OLS-ONLY: VARX + ACLE-VARX (Phase 0)
            # =============================================================
            print("=" * 80)
            print(f"OLS EXPERIMENT: {conf_label} (VARX + ACLE-VARX)")
            print(f"  Confounders: {confounder_names}")
            print("=" * 80)

            # Load endogenous data (ETF returns + confounders as log returns)
            print("\n" + "-" * 40)
            print("Loading Data")
            print("-" * 40)

            endo_df = load_opcl_with_confounders(
                confounder_names=confounder_names,
                etf_tickers=ETFS,
                n_days=n_days,
            )

            spy_df = load_opcl_data(tickers=["SPY"], n_days=n_days + 1 if n_days else None)
            common_dates_endo = endo_df.index.intersection(spy_df.index)
            endo_df = endo_df.loc[common_dates_endo]
            spy_df_endo = spy_df.loc[common_dates_endo]

            endo_combined = endo_df.copy()
            endo_combined["SPY"] = spy_df_endo["SPY"]

            endo_dates = endo_combined.index.strftime("%Y-%m-%d").tolist()
            endo_all_tickers = endo_combined.columns.tolist()
            endo_model_tickers = [t for t in endo_all_tickers if t != "SPY"]

            print(f"  Data: {len(endo_dates)} days, {len(endo_model_tickers)} variables")
            print(f"    Model variables: {endo_model_tickers}")

            Y_endo = torch.from_numpy(endo_combined.values.astype("float32")).to(device)
            spy_idx_endo = endo_all_tickers.index("SPY")
            Y_endo_model = torch.cat([Y_endo[:, :spy_idx_endo], Y_endo[:, spy_idx_endo + 1:]], dim=1)
            lookback_var = config.lookback_var

            # --- VARX ---
            print("\n" + "=" * 60)
            print("  METHOD 1/2: VARX (OLS)")
            print("=" * 60)

            varx_start = time.perf_counter()
            varx_result_full = fit_var(
                Y=Y_endo_model,
                p_max=p_max,
                config=config,
                validation_days=validation_days,
                asset_names=endo_model_tickers,
                dates=endo_dates[lookback_var:],
            )
            varx_elapsed = time.perf_counter() - varx_start

            etf_indices = [i for i, t in enumerate(endo_model_tickers) if t in ETFS]
            conf_indices = [i for i, t in enumerate(endo_model_tickers) if t not in ETFS]
            etf_only_tickers = [endo_model_tickers[i] for i in etf_indices]

            varx_result_bt = VARXResult(
                forecasts=varx_result_full.forecasts[etf_indices, :],
                forecasts_all=varx_result_full.forecasts_all[etf_indices, :, :],
                p_optimal=varx_result_full.p_optimal,
                p_max=varx_result_full.p_max,
                coefficients=varx_result_full.coefficients[:, :, etf_indices, :][:, :, :, etf_indices],
                asset_names=etf_only_tickers,
                confounder_names=[endo_model_tickers[i] for i in conf_indices],
                dates=varx_result_full.dates,
            )

            print(f"  VARX complete: {varx_result_bt.forecasts.shape[1]} output days, {varx_elapsed:.1f}s")

            run_backtest_and_save(
                varx_result_bt, endo_combined.values, endo_dates, endo_all_tickers,
                "varx", conf_label, "ols", output_dir, show_plots,
            )

            # --- ACLE-VARX ---
            print("\n" + "=" * 60)
            print("  METHOD 2/2: ACLE-VARX (OLS)")
            print("=" * 60)

            aclevarx_start = time.perf_counter()
            aclevarx_result_full = fit_aclevarx(
                Y=Y_endo_model,
                alpha_grid=alpha_grid,
                p_max=p_max,
                config=config,
                validation_days=validation_days,
                asset_names=endo_model_tickers,
                dates=endo_dates[lookback_var + validation_days:],
                verbose=verbose,
            )
            aclevarx_elapsed = time.perf_counter() - aclevarx_start

            etf_indices = [i for i, t in enumerate(endo_model_tickers) if t in ETFS]
            conf_indices = [i for i, t in enumerate(endo_model_tickers) if t not in ETFS]
            etf_only_tickers = [endo_model_tickers[i] for i in etf_indices]

            aclevarx_result_bt = ACLEVARXResult(
                forecasts=aclevarx_result_full.forecasts[etf_indices, :],
                forecasts_all=aclevarx_result_full.forecasts_all[etf_indices, :, :],
                p_optimal_all=aclevarx_result_full.p_optimal_all,
                alpha_optimal=aclevarx_result_full.alpha_optimal,
                p_optimal=aclevarx_result_full.p_optimal,
                alpha_grid=aclevarx_result_full.alpha_grid,
                asset_names=etf_only_tickers,
                confounder_names=[endo_model_tickers[i] for i in conf_indices],
                dates=aclevarx_result_full.dates,
                SE_all=aclevarx_result_full.SE_all,
            )

            print(f"  ACLE-VARX complete: {aclevarx_result_bt.forecasts.shape[1]} output days, {aclevarx_elapsed:.1f}s")

            run_backtest_and_save(
                aclevarx_result_bt, endo_combined.values, endo_dates, endo_all_tickers,
                "aclevarx", conf_label, "ols", output_dir, show_plots,
            )

        else:
            # =============================================================
            # DML: OR-VARX + ORACLE-VARX (Phases 1-3)
            # =============================================================
            print("=" * 80)
            print(f"DML EXPERIMENT: {conf_label} / {learner_name}")
            print(f"  Confounders: {confounder_names}")
            print("=" * 80)

            # Load exogenous data for DML methods
            print("\n" + "-" * 40)
            print("Loading Data")
            print("-" * 40)

            tickers_dml = ETFS + ["SPY"]
            Y_dml, W_dml, dml_dates, dml_tickers = prepare_tensors(
                tickers=tickers_dml,
                confounder_names=confounder_names,
                n_days=n_days,
                device=device,
            )

            spy_idx_dml = dml_tickers.index("SPY")
            Y_dml_etf = torch.cat([Y_dml[:, :spy_idx_dml], Y_dml[:, spy_idx_dml + 1:]], dim=1)
            etf_tickers_dml = [t for t in dml_tickers if t != "SPY"]

            print(f"  Data: {len(dml_dates)} days, Y={Y_dml_etf.shape}, W={W_dml.shape}")

            print("\n" + "=" * 60)
            print(f"  DML FIRST STAGE (shared by OR-VARX + ORACLE-VARX)")
            print(f"  Learner: {learner_name}, n_jobs: {n_jobs}")
            print("=" * 60)

            lookback_orvarx = config.lookback_orvarx
            n_output_dates = len(dml_dates) - lookback_orvarx - validation_days

            core_start = time.perf_counter()
            core_results = fit_orvarx_core(
                Y=Y_dml_etf,
                W=W_dml,
                p_max=p_max,
                config=config,
                learner_name=learner_name,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            core_elapsed = time.perf_counter() - core_start
            print(f"  DML first stage complete: {core_elapsed:.1f}s ({core_elapsed/60:.1f} min)")

            # --- OR-VARX ---
            print("\n" + "=" * 60)
            print("  METHOD 1/2: OR-VARX")
            print("=" * 60)

            orvarx_start = time.perf_counter()
            orvarx_result = fit_orvarx_batched(
                Y=Y_dml_etf,
                W=W_dml,
                p_max=p_max,
                config=config,
                validation_days=validation_days,
                asset_names=etf_tickers_dml,
                confounder_names=confounder_names,
                dates=dml_dates[lookback_orvarx + validation_days:],
                learner_name=learner_name,
                n_jobs=n_jobs,
                verbose=verbose,
                core_results=core_results,
            )
            orvarx_elapsed = time.perf_counter() - orvarx_start
            print(f"  OR-VARX complete: {orvarx_result.forecasts.shape[1]} output days, {orvarx_elapsed:.1f}s")

            run_backtest_and_save(
                orvarx_result, Y_dml, dml_dates, dml_tickers,
                "orvarx", conf_label, learner_label, output_dir, show_plots,
            )

            # --- ORACLE-VARX ---
            print("\n" + "=" * 60)
            print("  METHOD 2/2: ORACLE-VARX")
            print("=" * 60)

            oraclevarx_start = time.perf_counter()
            oraclevarx_result = fit_oraclevarx_batched(
                Y=Y_dml_etf,
                W=W_dml,
                alpha_grid=alpha_grid,
                p_max=p_max,
                config=config,
                validation_days=validation_days,
                asset_names=etf_tickers_dml,
                confounder_names=confounder_names,
                dates=dml_dates[lookback_orvarx + validation_days:],
                learner_name=learner_name,
                n_jobs=n_jobs,
                verbose=verbose,
                core_results=core_results,
            )
            oraclevarx_elapsed = time.perf_counter() - oraclevarx_start
            print(f"  ORACLE-VARX complete: {oraclevarx_result.forecasts.shape[1]} output days, {oraclevarx_elapsed:.1f}s")

            run_backtest_and_save(
                oraclevarx_result, Y_dml, dml_dates, dml_tickers,
                "oraclevarx", conf_label, learner_label, output_dir, show_plots,
            )

    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 80)
    print(f"ALL METHODS COMPLETE — Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run combined experiment with amortized DML first stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # VIX confounders with lgbm learner
  python scripts/run_combined_experiment.py --confounders vix --learner lgbm --no-show --verbose

  # No confounders (VAR + ACLE-VAR baseline)
  python scripts/run_combined_experiment.py --no-confounders --no-show

  # Quick smoke test
  python scripts/run_combined_experiment.py --confounders vix --learner lgbm --n-days 1500 --no-show
        """
    )
    parser.add_argument("--n-days", type=int, default=None,
                        help="Number of days to load (default: all)")
    parser.add_argument("--confounders", type=str, default="vix",
                        help="Confounder config: preset name (vix/macro5/all10) or comma-separated (default: vix)")
    parser.add_argument("--no-confounders", action="store_true",
                        help="Run VAR + ACLE-VAR baseline (no confounders)")
    parser.add_argument("--learner", type=str, default="lgbm",
                        choices=["xgboost", "lgbm", "rf", "extra_trees"],
                        help="First-stage learner for DML methods (default: lgbm)")
    parser.add_argument("--validation-days", type=int, default=21,
                        help="Validation period (default: 21)")
    parser.add_argument("--p-max", type=int, default=10,
                        help="Maximum lag order (default: 10)")
    parser.add_argument("--alpha-grid", type=str, default=None,
                        help="Comma-separated alpha values (default: 0.01,0.05,0.10,0.15,0.20,0.25,0.30)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="CPU cores per method (default: auto-computed for 4 parallel panes)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Base results directory (default: results)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    parser.add_argument("--ols-only", action="store_true",
                        help="In confounders mode, run only VARX + ACLE-VARX (skip DML). No effect with --no-confounders.")

    args = parser.parse_args()

    # Parse alpha grid if provided
    alpha_grid = None
    if args.alpha_grid:
        alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",")]

    main(
        n_days=args.n_days,
        confounders=args.confounders,
        no_confounders=args.no_confounders,
        learner_name=args.learner,
        validation_days=args.validation_days,
        p_max=args.p_max,
        alpha_grid=alpha_grid,
        n_jobs=args.n_jobs,
        device=args.device,
        output_dir=args.output_dir,
        show_plots=not args.no_show,
        verbose=args.verbose,
        ols_only=args.ols_only,
    )
