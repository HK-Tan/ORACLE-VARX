#!/usr/bin/env python3
"""Run the 3-variable toy benchmark: 14+ (method, observability) runs.

Phase 0: OLS baselines (VAR, ACLE-VAR, VARX, ACLE-VARX) — sequential, CPU
Phase 1: DML methods (OR-VARX, ORACLE-VARX) per obs level — parallelizable
Phase 2: TabPFN methods (OR-VARX-TabPFN, ORACLE-VARX-TabPFN) per obs level — GPU

Usage:
    python scripts/run_toy_benchmark.py --phase all
    python scripts/run_toy_benchmark.py --phase 0
    python scripts/run_toy_benchmark.py --phase 1 --obs all
    python scripts/run_toy_benchmark.py --phase 1 --obs partial_2 --learner extra_trees
    python scripts/run_toy_benchmark.py --phase 1 --learner all
    python scripts/run_toy_benchmark.py --phase 2 --device cuda --n-estimators 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modules.grid_config import GridConfig
from src.results import VARXResult, ACLEVARXResult, ORACLEVARXResult
from src.models.var_pytorch import fit_var
from src.models.acle_var import fit_aclevarx
from src.models.dml_pytorch import fit_orvarx_core, fit_orvarx_batched, get_physical_cpu_count
from src.models.oracle_var import fit_oraclevarx_batched
from src.models.coefficient_refit import (
    get_target_days,
    refit_var_coefficients_for_day,
    refit_dml_coefficients_for_day,
)
from src.evaluation.plotting import (
    plot_lag_analysis,
    plot_coefficient_heatmap,
    plot_coefficient_evolution_per_p,
)
from src.synthetic.dgp import ToyDGPConfig, get_observed_confounders

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("dataset/toy")
RESULTS_DIR = Path("results-toy")

OBS_LEVELS = ["all", "partial_2", "partial_1"]
ENDO_NAMES = ["X", "Y", "Z"]
N_ENDO = 3
LEARNERS = ["lgbm", "xgboost", "rf", "extra_trees"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_toy_data() -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, dict]:
    """Load pre-generated toy data from dataset/toy/.

    Returns:
        Y: endogenous tensor (T, 3)
        W_full: confounder numpy array (T, 3)
        A_true: ground truth coefficients (T, 3, 3, 3)
        dgp_config: config dict
    """
    Y_df = pd.read_csv(DATA_DIR / "Y.csv")
    W_df = pd.read_csv(DATA_DIR / "W.csv")
    A_true = torch.load(DATA_DIR / "A_true.pt", weights_only=True)

    with open(DATA_DIR / "dgp_config.json") as f:
        dgp_config = json.load(f)

    Y = torch.from_numpy(Y_df.values.astype(np.float32))
    W_full = W_df.values.astype(np.float32)

    print(f"Loaded: Y {tuple(Y.shape)}, W {W_full.shape}, A_true {tuple(A_true.shape)}")
    return Y, W_full, A_true, dgp_config


def make_dates(T: int) -> List[str]:
    """Create synthetic date strings T0, T1, ..., T{T-1}."""
    return [f"T{i}" for i in range(T)]


def parse_time_index(date_str: str) -> int:
    """Parse absolute time index from date string 'T123' -> 123."""
    return int(date_str[1:])


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def extract_per_p_coefficients(
    per_p_coefs: torch.Tensor,
    p_optimal: torch.Tensor,
) -> torch.Tensor:
    """For each day d, extract the full coefficient matrix from VAR(p_optimal[d]).

    Args:
        per_p_coefs: shape (n_days, p_max, p_max, n_assets, n_assets)
        p_optimal: shape (n_days,) — 1-indexed lag for each day

    Returns:
        shape (n_days, p_max, n_assets, n_assets) — coefficients from VAR(p_optimal[d])
    """
    n_days = per_p_coefs.shape[0]
    p_max = per_p_coefs.shape[1]
    result = torch.zeros(n_days, p_max, *per_p_coefs.shape[3:],
                         device=per_p_coefs.device, dtype=per_p_coefs.dtype)
    for d in range(n_days):
        p = int(p_optimal[d].item())
        if 1 <= p <= p_max:
            result[d] = per_p_coefs[d, p - 1]
    return result


def compute_edge_metrics(
    result_coefs: torch.Tensor,
    result_dates: List[str],
    A_true: torch.Tensor,
    p_max_true: int = 3,
) -> Dict:
    """Compute coefficient recovery metrics against ground truth.

    Args:
        result_coefs: shape (n_output_days, p_max_est, n_endo, n_endo)
        result_dates: date strings for each output day
        A_true: shape (T, p_max_true, n_endo, n_endo)
        p_max_true: true maximum lag (3)

    Returns:
        Dict with MAE/MSE on non-zero and zero edges, by regime and overall.
    """
    n_days = result_coefs.shape[0]
    p_compare = min(result_coefs.shape[1], p_max_true)

    est = result_coefs[:, :p_compare, :, :].cpu().numpy()

    # Align to absolute time
    time_indices = np.array([parse_time_index(d) for d in result_dates])
    true = A_true.numpy()[time_indices][:, :p_compare, :, :]

    # Masks for non-zero and zero true edges (per time step)
    nonzero_mask = np.abs(true) > 1e-8
    zero_mask = ~nonzero_mask

    regimes = [(0, 1000), (1000, 2000), (2000, 3000)]
    metrics = {}

    for regime_idx, (r_start, r_end) in enumerate(regimes):
        in_regime = (time_indices >= r_start) & (time_indices < r_end)
        if not in_regime.any():
            continue

        e_r = est[in_regime]
        t_r = true[in_regime]
        nz_r = nonzero_mask[in_regime]
        z_r = zero_mask[in_regime]

        rname = f"regime_{regime_idx + 1}"
        if nz_r.any():
            diff_nz = e_r[nz_r] - t_r[nz_r]
            metrics[f"{rname}_nonzero_mae"] = float(np.mean(np.abs(diff_nz)))
            metrics[f"{rname}_nonzero_mse"] = float(np.mean(diff_nz**2))
        if z_r.any():
            metrics[f"{rname}_zero_mae"] = float(np.mean(np.abs(e_r[z_r])))

    # Overall
    if nonzero_mask.any():
        diff_all = est[nonzero_mask] - true[nonzero_mask]
        metrics["overall_nonzero_mae"] = float(np.mean(np.abs(diff_all)))
        metrics["overall_nonzero_mse"] = float(np.mean(diff_all**2))
    if zero_mask.any():
        metrics["overall_zero_mae"] = float(np.mean(np.abs(est[zero_mask])))

    return metrics


def compute_forecast_metrics(
    forecasts: torch.Tensor,
    Y: torch.Tensor,
    result_dates: List[str],
) -> Dict:
    """Compute forecast error metrics.

    Args:
        forecasts: shape (n_endo, n_output_days)
        Y: full endogenous data (T, n_endo)
        result_dates: date strings for each output day

    Returns:
        Dict with forecast MAE/MSE by regime and overall.
    """
    time_indices = np.array([parse_time_index(d) for d in result_dates])
    actuals = Y[time_indices].cpu().numpy()  # (n_days, n_endo)
    preds = forecasts.cpu().numpy().T  # (n_days, n_endo)

    errors = preds - actuals
    regimes = [(0, 1000), (1000, 2000), (2000, 3000)]
    metrics = {}

    for regime_idx, (r_start, r_end) in enumerate(regimes):
        in_regime = (time_indices >= r_start) & (time_indices < r_end)
        if not in_regime.any():
            continue
        err_r = errors[in_regime]
        rname = f"regime_{regime_idx + 1}"
        metrics[f"{rname}_forecast_mae"] = float(np.mean(np.abs(err_r)))
        metrics[f"{rname}_forecast_mse"] = float(np.mean(err_r**2))

    metrics["overall_forecast_mae"] = float(np.mean(np.abs(errors)))
    metrics["overall_forecast_mse"] = float(np.mean(errors**2))
    return metrics


def compute_lag_metrics(
    p_optimal: np.ndarray,
    result_dates: List[str],
    A_true: torch.Tensor,
    p_max_true: int = 3,
) -> Dict:
    """Compute lag order recovery metrics (RMSE of p_hat vs p_true).

    p_true(t) = max lag k such that any entry in A_true[t, k-1, :, :] is nonzero.

    Args:
        p_optimal: estimated lag order per output day, shape (n_output_days,)
        result_dates: date strings for each output day
        A_true: shape (T, p_max_true, n_endo, n_endo)
        p_max_true: true maximum lag (3)

    Returns:
        Dict with overall_lag_rmse and per-regime lag_rmse.
    """
    time_indices = np.array([parse_time_index(d) for d in result_dates])
    A_np = A_true.numpy()

    # Compute true lag order for each output day
    p_true = np.ones(len(time_indices), dtype=int)
    for i, t in enumerate(time_indices):
        for k in range(p_max_true - 1, -1, -1):  # highest lag first
            if np.any(np.abs(A_np[t, k]) > 1e-8):
                p_true[i] = k + 1
                break

    diff = p_optimal.astype(float) - p_true.astype(float)

    regimes = [(0, 1000), (1000, 2000), (2000, 3000)]
    metrics = {}
    for regime_idx, (r_start, r_end) in enumerate(regimes):
        in_regime = (time_indices >= r_start) & (time_indices < r_end)
        if not in_regime.any():
            continue
        rname = f"regime_{regime_idx + 1}"
        metrics[f"{rname}_lag_rmse"] = float(np.sqrt(np.mean(diff[in_regime] ** 2)))

    metrics["overall_lag_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_forecast_error(
    forecasts: torch.Tensor,
    Y: torch.Tensor,
    result_dates: List[str],
    title: str,
    save_dir: str,
    rolling_window: int = 50,
) -> None:
    """Plot rolling forecast MSE and MAE as separate plots with regime boundaries."""
    time_indices = np.array([parse_time_index(d) for d in result_dates])
    actuals = Y[time_indices].cpu().numpy()
    preds = forecasts.cpu().numpy().T
    errors = preds - actuals

    # Per-timestep aggregate: mean over assets
    mse_t = np.mean(errors**2, axis=1)
    mae_t = np.mean(np.abs(errors), axis=1)

    # Rolling
    mse_roll = pd.Series(mse_t).rolling(rolling_window, min_periods=1).mean().values
    mae_roll = pd.Series(mae_t).rolling(rolling_window, min_periods=1).mean().values

    for metric_name, values in [("MSE", mse_roll), ("MAE", mae_roll)]:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(time_indices, values, label=f"Rolling {metric_name}", linewidth=1.5)

        for boundary in [1000, 2000]:
            ax.axvline(x=boundary, color="gray", linestyle=":", linewidth=1, alpha=0.7)

        ax.set_xlim(0, 3000)
        ax.set_xlabel("Time index")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{title} ({metric_name})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f"forecast_error_{metric_name.lower()}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


def plot_edge_trajectories(
    all_results: Dict[str, Tuple[torch.Tensor, List[str]]],
    A_true: torch.Tensor,
    obs_level: str,
    save_path: str,
) -> None:
    """Plot true vs estimated coefficient trajectories for all 6 edges.

    Args:
        all_results: {method_label: (coefficients, dates)} for each method
        A_true: ground truth (T, 3, 3, 3)
        obs_level: for title
        save_path: output path
    """
    # 6 true edges: (lag_0idx, target_i, source_j, label)
    edges = [
        (0, 0, 2, "Z->X (lag 1)"),
        (0, 1, 0, "X->Y (lag 1)"),
        (0, 2, 1, "Y->Z (lag 1)"),
        (1, 0, 0, "X->X (lag 2)"),
        (1, 1, 1, "Y->Y (lag 2)"),
        (2, 2, 2, "Z->Z (lag 3)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    t_all = np.arange(A_true.shape[0])
    A_np = A_true.numpy()

    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(all_results))))

    for idx, (k, i, j, label) in enumerate(edges):
        ax = axes[idx]

        # True coefficient
        true_vals = A_np[:, k, i, j]
        ax.plot(t_all, true_vals, color="black", linewidth=2.5, label="True", zorder=10)

        # Estimated from each method
        for cidx, (method_label, (coefs, dates)) in enumerate(all_results.items()):
            if coefs is None:
                continue
            t_idx = np.array([parse_time_index(d) for d in dates])
            p_est = coefs.shape[1]
            if k < p_est:
                est_vals = coefs[:, k, i, j].cpu().numpy()
                ax.plot(
                    t_idx, est_vals,
                    color=colors[cidx % len(colors)],
                    linewidth=1.0, alpha=0.8,
                    label=method_label,
                )

        for boundary in [1000, 2000]:
            ax.axvline(x=boundary, color="gray", linestyle=":", linewidth=1, alpha=0.7)

        ax.set_xlim(0, 3000)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("t")
        ax.set_ylabel("Coefficient")

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(8, len(handles)), fontsize=9)
    fig.suptitle(f"Edge Trajectories — obs={obs_level}", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Incremental metrics summary
# ---------------------------------------------------------------------------

def append_metrics_summary(metrics: Dict, output_dir: Path) -> None:
    """Append a single metrics row to the summary CSV, replacing if method+obs already exists."""
    csv_path = output_dir / "metrics_summary.csv"
    summary_cols = ["method", "obs_level",
                    "overall_nonzero_mae", "overall_nonzero_mse", "overall_zero_mae",
                    "overall_forecast_mae", "overall_forecast_mse", "overall_lag_rmse"]
    row = {c: metrics.get(c) for c in summary_cols}

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Remove existing row for same (method, obs_level) to avoid duplicates
        mask = (df["method"] == row["method"]) & (df["obs_level"] == row["obs_level"])
        df = df[~mask]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Per-run evaluation and saving
# ---------------------------------------------------------------------------

def evaluate_and_save(
    result: Union[VARXResult, ACLEVARXResult, ORACLEVARXResult],
    Y: torch.Tensor,
    A_true: torch.Tensor,
    method_name: str,
    obs_label: str,
    output_dir: Path,
    config: GridConfig,
    coef_result: Optional[VARXResult] = None,
    coefficients_override: Optional[torch.Tensor] = None,
    Y_for_refit: Optional[torch.Tensor] = None,
    W_for_refit: Optional[torch.Tensor] = None,
    refit_asset_names: Optional[List[str]] = None,
    learner_name: str = "extra_trees",
    n_jobs: int = 1,
    show_plots: bool = False,
) -> Dict:
    """Evaluate a single result and save all outputs.

    Args:
        result: model result
        Y: full endogenous data (T, n_endo)
        A_true: ground truth (T, p_max_true, n_endo, n_endo)
        method_name: e.g. "VAR", "OR-VARX"
        obs_label: e.g. "none", "all", "partial_2"
        output_dir: base results directory
        config: GridConfig used
        coef_result: VARXResult with .coefficients for heatmaps/plots
        coefficients_override: lag-masked coefficient tensor for edge metrics
                              (used by ACLE/ORACLE methods)
        Y_for_refit: Y tensor for coefficient refitting
        W_for_refit: W tensor for DML refitting (None for OLS methods)
        refit_asset_names: names for refit plots (defaults to ENDO_NAMES)
        learner_name: for DML refit
        n_jobs: for DML refit
        show_plots: display plots interactively

    Returns:
        metrics dict
    """
    if refit_asset_names is None:
        refit_asset_names = ENDO_NAMES

    exp_name = f"{method_name}_{obs_label}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # For edge metrics: use override (lag-masked) if provided, else result's own
    has_coefs = isinstance(result, VARXResult)
    # For heatmaps / per-p plots: need VARXResult methods
    varx_source = result if has_coefs else coef_result

    # --- Edge metrics ---
    edge_metrics = {}
    if coefficients_override is not None:
        edge_metrics = compute_edge_metrics(
            coefficients_override, result.dates, A_true
        )
    elif has_coefs:
        edge_metrics = compute_edge_metrics(
            result.coefficients, result.dates, A_true
        )

    # --- Forecast metrics ---
    forecast_metrics = compute_forecast_metrics(result.forecasts, Y, result.dates)

    # --- Lag order metrics ---
    lag_metrics = compute_lag_metrics(
        result.p_optimal.cpu().numpy(), result.dates, A_true
    )

    # --- Combined metrics ---
    metrics = {**edge_metrics, **forecast_metrics, **lag_metrics, "method": method_name, "obs_level": obs_label}

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Append to incremental metrics summary ---
    append_metrics_summary(metrics, output_dir)

    # --- Save result ---
    result.save(str(exp_dir / "result.pt"))

    # --- Lag analysis plot (with absolute time indices) ---
    time_indices = np.array([parse_time_index(d) for d in result.dates])
    plot_lag_analysis(
        p_optimal=result.p_optimal.cpu().numpy(),
        dates=time_indices,
        title=f"Lag Analysis — {exp_name}",
        save_path=str(exp_dir / "lag_analysis.png"),
        show_plot=show_plots,
    )

    # --- Forecast error plot ---
    plot_forecast_error(
        result.forecasts, Y, result.dates,
        title=f"Forecast Error — {exp_name}",
        save_dir=str(exp_dir),
    )

    # --- Edge trajectory plot (single method, true vs estimated) ---
    # Use masked coefficients for trajectory if available, else base model's
    if coefficients_override is not None:
        single_result = {exp_name: (coefficients_override, result.dates)}
        plot_edge_trajectories(
            single_result, A_true, obs_label,
            save_path=str(exp_dir / "edge_trajectories.png"),
        )
    elif varx_source is not None and hasattr(varx_source, "coefficients"):
        single_result = {exp_name: (varx_source.coefficients, varx_source.dates)}
        plot_edge_trajectories(
            single_result, A_true, obs_label,
            save_path=str(exp_dir / "edge_trajectories.png"),
        )

    # --- Heatmaps (need VARXResult methods) ---
    if varx_source is not None and hasattr(varx_source, "coefficients"):
        target_days = get_target_days(varx_source)
        for day_label, day_idx in target_days:
            heatmap_df = varx_source.get_coefficient_heatmap_matrix(day_idx=day_idx)
            date_str = varx_source.dates[day_idx]
            plot_coefficient_heatmap(
                heatmap_df,
                title=f"{exp_name} — {day_label} (t={parse_time_index(date_str)})",
                save_path=str(exp_dir / f"heatmap_{day_label}.png"),
                show_plot=show_plots,
            )

    # --- Per-p coefficient evolution ---
    if varx_source is not None and Y_for_refit is not None:
        target_days = get_target_days(varx_source)
        is_dml = W_for_refit is not None
        for day_label, day_idx in target_days:
            p_star = int(varx_source.p_optimal[day_idx].item())
            date_str = varx_source.dates[day_idx]
            t_abs = parse_time_index(date_str)

            if p_star < 1:
                continue

            try:
                if is_dml:
                    per_p = refit_dml_coefficients_for_day(
                        Y=Y_for_refit,
                        W=W_for_refit,
                        day_idx=t_abs,
                        p_star=p_star,
                        lookback=config.lookback_orvarx,
                        asset_names=refit_asset_names,
                        date=date_str,
                        config=config,
                        learner_name=learner_name,
                        n_jobs=n_jobs,
                    )
                else:
                    per_p = refit_var_coefficients_for_day(
                        Y=Y_for_refit,
                        day_idx=t_abs,
                        p_star=p_star,
                        lookback=config.lookback_var,
                        asset_names=refit_asset_names,
                        date=date_str,
                    )

                plot_coefficient_evolution_per_p(
                    per_p,
                    title=f"{exp_name} — {day_label} (t={t_abs})",
                    save_path=str(exp_dir / f"coef_evolution_{day_label}.png"),
                    show_plot=show_plots,
                )
            except Exception as e:
                print(f"  Warning: coef evolution plot failed for {day_label}: {e}")

    print(f"  Saved: {exp_dir}")
    return metrics


# ---------------------------------------------------------------------------
# Phase 0: OLS baselines
# ---------------------------------------------------------------------------

def run_phase0(
    Y: torch.Tensor,
    W_full: np.ndarray,
    A_true: torch.Tensor,
    dates: List[str],
    config: GridConfig,
    dgp_config: dict,
    show_plots: bool = False,
    verbose: bool = False,
) -> List[Dict]:
    """Run Phase 0: VAR, ACLE-VAR, VARX, ACLE-VARX for all obs levels."""
    p_max = dgp_config["p_max"]
    validation_days = dgp_config["validation_days"]
    lookback_var = config.lookback_var
    all_metrics = []

    # ---- 1. VAR (no confounders) ----
    print("\n[Phase 0] Fitting VAR (no confounders)...")
    t0 = time.time()
    var_result, per_p_var = fit_var(
        Y=Y, p_max=p_max, config=config,
        validation_days=validation_days,
        asset_names=ENDO_NAMES,
        dates=dates[lookback_var:],
        store_per_p_coefs=True,
    )
    print(f"  VAR done in {time.time() - t0:.1f}s, output days: {len(var_result.dates)}")

    m = evaluate_and_save(
        var_result, Y, A_true, "VAR", "none", RESULTS_DIR, config,
        Y_for_refit=Y, show_plots=show_plots,
    )
    all_metrics.append(m)

    # ---- 2. ACLE-VAR (no confounders) ----
    print("[Phase 0] Fitting ACLE-VAR (no confounders)...")
    t0 = time.time()
    acle_var_result = fit_aclevarx(
        Y=Y, p_max=p_max, config=config,
        validation_days=validation_days,
        asset_names=ENDO_NAMES,
        dates=dates[lookback_var + validation_days:],
        verbose=verbose,
    )
    print(f"  ACLE-VAR done in {time.time() - t0:.1f}s, output days: {len(acle_var_result.dates)}")

    extracted_coefs = extract_per_p_coefficients(per_p_var, acle_var_result.p_optimal)
    m = evaluate_and_save(
        acle_var_result, Y, A_true, "ACLE-VAR", "none", RESULTS_DIR, config,
        coef_result=var_result, coefficients_override=extracted_coefs,
        Y_for_refit=Y, show_plots=show_plots,
    )
    all_metrics.append(m)

    # ---- 3-8. VARX and ACLE-VARX for each obs level ----
    for obs_level in OBS_LEVELS:
        W_obs, obs_names = get_observed_confounders(W_full, obs_level)
        n_conf = W_obs.shape[1]
        W_obs_t = torch.from_numpy(W_obs)

        # Build combined Y matrix: [endo | confounders]
        Y_combined = torch.cat([Y, W_obs_t], dim=1)
        combined_names = ENDO_NAMES + obs_names
        endo_indices = list(range(N_ENDO))

        # ---- VARX ----
        print(f"[Phase 0] Fitting VARX (obs={obs_level}, n_conf={n_conf})...")
        t0 = time.time()
        varx_full, per_p_full = fit_var(
            Y=Y_combined, p_max=p_max, config=config,
            validation_days=validation_days,
            asset_names=combined_names,
            dates=dates[lookback_var:],
            store_per_p_coefs=True,
        )

        # Slice to endogenous-only
        varx_result = VARXResult(
            forecasts=varx_full.forecasts[endo_indices, :],
            forecasts_all=varx_full.forecasts_all[endo_indices, :, :],
            p_optimal=varx_full.p_optimal,
            p_max=varx_full.p_max,
            coefficients=varx_full.coefficients[:, :, endo_indices, :][:, :, :, endo_indices],
            asset_names=ENDO_NAMES,
            confounder_names=obs_names,
            dates=varx_full.dates,
        )
        print(f"  VARX done in {time.time() - t0:.1f}s")

        m = evaluate_and_save(
            varx_result, Y, A_true, "VARX", obs_level, RESULTS_DIR, config,
            Y_for_refit=Y_combined, refit_asset_names=combined_names,
            show_plots=show_plots,
        )
        all_metrics.append(m)

        # ---- ACLE-VARX ----
        print(f"[Phase 0] Fitting ACLE-VARX (obs={obs_level})...")
        t0 = time.time()
        acle_varx_full = fit_aclevarx(
            Y=Y_combined, p_max=p_max, config=config,
            validation_days=validation_days,
            asset_names=combined_names,
            dates=dates[lookback_var + validation_days:],
            verbose=verbose,
        )

        # Slice forecasts to endogenous-only
        acle_varx_result = ACLEVARXResult(
            forecasts=acle_varx_full.forecasts[endo_indices, :],
            forecasts_all=acle_varx_full.forecasts_all[endo_indices, :, :],
            p_optimal_all=acle_varx_full.p_optimal_all,
            alpha_optimal=acle_varx_full.alpha_optimal,
            p_optimal=acle_varx_full.p_optimal,
            alpha_grid=acle_varx_full.alpha_grid,
            asset_names=ENDO_NAMES,
            confounder_names=obs_names,
            dates=acle_varx_full.dates,
        )
        print(f"  ACLE-VARX done in {time.time() - t0:.1f}s")

        # Slice per_p to endo-only (same slicing as varx_result.coefficients)
        per_p_endo = per_p_full[:, :, :, endo_indices, :][:, :, :, :, endo_indices]
        extracted_coefs = extract_per_p_coefficients(per_p_endo, acle_varx_result.p_optimal)
        m = evaluate_and_save(
            acle_varx_result, Y, A_true, "ACLE-VARX", obs_level, RESULTS_DIR, config,
            coef_result=varx_result, coefficients_override=extracted_coefs,
            Y_for_refit=Y_combined,
            refit_asset_names=combined_names, show_plots=show_plots,
        )
        all_metrics.append(m)

    return all_metrics


# ---------------------------------------------------------------------------
# Phase 1: DML methods
# ---------------------------------------------------------------------------

def run_phase1(
    Y: torch.Tensor,
    W_full: np.ndarray,
    A_true: torch.Tensor,
    dates: List[str],
    config: GridConfig,
    dgp_config: dict,
    obs_levels: List[str] = None,
    learner_names: Optional[List[str]] = None,
    n_jobs: int = -1,
    show_plots: bool = False,
    verbose: bool = False,
) -> List[Dict]:
    """Run Phase 1: OR-VARX and ORACLE-VARX for specified obs levels and learners."""
    if obs_levels is None:
        obs_levels = OBS_LEVELS
    if learner_names is None:
        learner_names = ["extra_trees"]

    p_max = dgp_config["p_max"]
    validation_days = dgp_config["validation_days"]
    lookback_orvarx = config.lookback_orvarx

    # Resolve n_jobs
    if n_jobs == -1:
        n_cpus = get_physical_cpu_count()
        n_jobs = max(1, n_cpus - 1)

    all_metrics = []

    for learner_name in learner_names:
        print(f"\n[Phase 1] Using n_jobs={n_jobs}, learner={learner_name}")

        for obs_level in obs_levels:
            W_obs, obs_names = get_observed_confounders(W_full, obs_level)
            W_obs_t = torch.from_numpy(W_obs)

            print(f"\n[Phase 1] DML core for obs={obs_level} (n_conf={W_obs.shape[1]})...")
            t0 = time.time()

            # Shared DML first stage
            orvarx_result, core_results = fit_orvarx_batched(
                Y=Y, W=W_obs_t, p_max=p_max, config=config,
                validation_days=validation_days,
                asset_names=ENDO_NAMES,
                confounder_names=obs_names,
                dates=dates[lookback_orvarx + validation_days:],
                learner_name=learner_name,
                n_jobs=n_jobs,
                verbose=verbose,
                return_core=True,
                store_per_p_coefs=True,
            )
            print(f"  OR-VARX done in {time.time() - t0:.1f}s, output days: {len(orvarx_result.dates)}")

            # Always include learner suffix in directory names
            or_method = f"OR-VARX_{learner_name}"
            oracle_method = f"ORACLE-VARX_{learner_name}"

            m = evaluate_and_save(
                orvarx_result, Y, A_true, or_method, obs_level, RESULTS_DIR, config,
                Y_for_refit=Y, W_for_refit=W_obs_t,
                learner_name=learner_name, n_jobs=n_jobs, show_plots=show_plots,
            )
            all_metrics.append(m)

            # ORACLE-VARX (reuse core_results)
            print(f"[Phase 1] ORACLE-VARX for obs={obs_level}...")
            t0 = time.time()
            oracle_result = fit_oraclevarx_batched(
                Y=Y, W=W_obs_t, p_max=p_max, config=config,
                validation_days=validation_days,
                asset_names=ENDO_NAMES,
                confounder_names=obs_names,
                dates=dates[lookback_orvarx + validation_days:],
                learner_name=learner_name,
                n_jobs=n_jobs,
                verbose=verbose,
                core_results=core_results,
            )
            print(f"  ORACLE-VARX done in {time.time() - t0:.1f}s, output days: {len(oracle_result.dates)}")

            # Extract per_p from core_results (5th element, untrimmed)
            per_p_raw = core_results[4]  # shape (n_total_test_days, p_max, p_max, n_assets, n_assets)
            per_p_trimmed = per_p_raw[validation_days:]  # trim to output days
            extracted_coefs = extract_per_p_coefficients(per_p_trimmed, oracle_result.p_optimal)
            m = evaluate_and_save(
                oracle_result, Y, A_true, oracle_method, obs_level, RESULTS_DIR, config,
                coef_result=orvarx_result, coefficients_override=extracted_coefs,
                Y_for_refit=Y, W_for_refit=W_obs_t,
                learner_name=learner_name, n_jobs=n_jobs, show_plots=show_plots,
            )
            all_metrics.append(m)

    return all_metrics


# ---------------------------------------------------------------------------
# Phase 2: TabPFN methods (GPU)
# ---------------------------------------------------------------------------

def run_phase2(
    Y: torch.Tensor,
    W_full: np.ndarray,
    A_true: torch.Tensor,
    dates: List[str],
    config: GridConfig,
    dgp_config: dict,
    obs_levels: List[str] = None,
    device: str = "cuda",
    n_estimators: int = 8,
    show_plots: bool = False,
    verbose: bool = False,
) -> List[Dict]:
    """Run Phase 2: OR-VARX-TabPFN and ORACLE-VARX-TabPFN for specified obs levels."""
    from src.models.oracle_var_tabpfn import fit_oraclevarx_tabpfn

    if obs_levels is None:
        obs_levels = OBS_LEVELS

    p_max = dgp_config["p_max"]
    validation_days = dgp_config["validation_days"]
    lookback_orvarx = config.lookback_orvarx

    all_metrics = []

    for obs_level in obs_levels:
        W_obs, obs_names = get_observed_confounders(W_full, obs_level)
        W_obs_t = torch.from_numpy(W_obs).to(device)
        Y_dev = Y.to(device)

        print(f"\n[Phase 2] TabPFN for obs={obs_level} (n_conf={W_obs.shape[1]})...")
        t0 = time.time()

        result_tuple = fit_oraclevarx_tabpfn(
            Y=Y_dev, W=W_obs_t,
            p_max=p_max, config=config,
            validation_days=validation_days,
            asset_names=ENDO_NAMES,
            confounder_names=obs_names,
            dates=dates[lookback_orvarx + validation_days:],
            n_estimators=n_estimators,
            device=device,
            verbose=verbose,
            store_per_p_coefs=True,
        )

        if result_tuple is None:
            print(f"  TabPFN returned None for obs={obs_level}, skipping")
            continue

        orvarx_result, oracle_result = result_tuple[0], result_tuple[1]
        per_p_coefs_tabpfn = result_tuple[2] if len(result_tuple) > 2 else None
        print(f"  TabPFN done in {time.time() - t0:.1f}s, output days: {len(orvarx_result.dates)}")

        # Move results back to CPU for evaluation
        Y_cpu = Y.cpu()

        # OR-VARX-TabPFN
        m = evaluate_and_save(
            orvarx_result, Y_cpu, A_true, "OR-VARX-TabPFN", obs_level, RESULTS_DIR, config,
            Y_for_refit=Y_cpu, W_for_refit=torch.from_numpy(W_obs),
            show_plots=show_plots,
        )
        all_metrics.append(m)

        # ORACLE-VARX-TabPFN
        if per_p_coefs_tabpfn is not None:
            extracted_coefs = extract_per_p_coefficients(per_p_coefs_tabpfn.cpu(), oracle_result.p_optimal)
        else:
            extracted_coefs = extract_per_p_coefficients(
                torch.zeros(len(oracle_result.dates), p_max, p_max, N_ENDO, N_ENDO),
                oracle_result.p_optimal,
            )
        m = evaluate_and_save(
            oracle_result, Y_cpu, A_true, "ORACLE-VARX-TabPFN", obs_level, RESULTS_DIR, config,
            coef_result=orvarx_result, coefficients_override=extracted_coefs,
            Y_for_refit=Y_cpu, W_for_refit=torch.from_numpy(W_obs),
            show_plots=show_plots,
        )
        all_metrics.append(m)

    return all_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run toy benchmark experiments")
    parser.add_argument("--phase", type=str, default="all", choices=["0", "1", "2", "all"])
    parser.add_argument("--obs", type=str, default=None,
                        choices=["all", "partial_2", "partial_1"],
                        help="Obs level for Phase 1/2 (default: run all)")
    parser.add_argument("--learner", type=str, default="extra_trees",
                        help="DML learner for Phase 1: lgbm, xgboost, rf, extra_trees, or 'all'")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for Phase 2 TabPFN (default: cpu)")
    parser.add_argument("--n-estimators", type=int, default=8,
                        help="TabPFN ensemble size for Phase 2 (default: 8)")
    parser.add_argument("--noise-scale", type=float, default=None,
                        help="Innovation noise scale (regenerates data if set)")
    parser.add_argument("--confounder-strength", type=float, default=None,
                        help="Confounder nuisance scaling λ (regenerates data if set)")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    show_plots = not args.no_show

    # Resolve learner list
    if args.learner == "all":
        learner_names = LEARNERS
    else:
        learner_names = [args.learner]

    # Regenerate data if alpha or confounder_strength overridden
    if args.noise_scale is not None or args.confounder_strength is not None:
        from src.synthetic.dgp import ToyDGPConfig, generate_toy_data as gen_data

        dgp_cfg = ToyDGPConfig()
        if args.noise_scale is not None:
            dgp_cfg.noise_scale = args.noise_scale
        if args.confounder_strength is not None:
            dgp_cfg.confounder_strength = args.confounder_strength
        print(f"Regenerating data: noise_scale={dgp_cfg.noise_scale}, λ={dgp_cfg.confounder_strength}")
        Y_np, W_np, truth = gen_data(dgp_cfg)

        out_dir = DATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(Y_np, columns=dgp_cfg.endo_names).to_csv(out_dir / "Y.csv", index=False)
        pd.DataFrame(W_np, columns=dgp_cfg.confounder_names).to_csv(out_dir / "W.csv", index=False)
        torch.save(truth.A_true, out_dir / "A_true.pt")
        config_dict = {
            "T": dgp_cfg.T, "n_endo": dgp_cfg.n_endo, "n_confounders": dgp_cfg.n_confounders,
            "burn_in": dgp_cfg.burn_in, "seed": dgp_cfg.seed,
            "noise_scale": dgp_cfg.noise_scale, "confounder_strength": dgp_cfg.confounder_strength,
            "ar_rho": dgp_cfg.ar_rho,
            "cross_effect": dgp_cfg.cross_effect,
            "a_XX_init": dgp_cfg.a_XX_init, "a_YY_init": dgp_cfg.a_YY_init, "a_ZZ_init": dgp_cfg.a_ZZ_init,
            "a_XX_decay_end": dgp_cfg.a_XX_decay_end, "a_YY_decay_end": dgp_cfg.a_YY_decay_end,
            "a_ZZ_decay_end": dgp_cfg.a_ZZ_decay_end,
            "p_max": dgp_cfg.p_max, "ols_window": dgp_cfg.ols_window,
            "tree_train_window": dgp_cfg.tree_train_window, "p_max_offset": dgp_cfg.p_max_offset,
            "test_size": dgp_cfg.test_size, "validation_days": dgp_cfg.validation_days,
            "endo_names": dgp_cfg.endo_names, "confounder_names": dgp_cfg.confounder_names,
        }
        with open(out_dir / "dgp_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"  Y std: {Y_np.std(axis=0)}, W std: {W_np.std(axis=0)}")

    # Load data
    Y, W_full, A_true, dgp_config = load_toy_data()
    T = Y.shape[0]
    dates = make_dates(T)

    # Create GridConfig matching toy parameters
    config = GridConfig(
        ols_window=dgp_config["ols_window"],
        tree_train_window=dgp_config.get("tree_train_window"),
        test_size=dgp_config["test_size"],
        p_max_offset=dgp_config["p_max_offset"],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0
    if args.phase in ("0", "all"):
        print("=" * 60)
        print("PHASE 0: OLS Baselines")
        print("=" * 60)
        t0 = time.time()
        run_phase0(
            Y, W_full, A_true, dates, config, dgp_config,
            show_plots=show_plots, verbose=args.verbose,
        )
        print(f"\nPhase 0 total: {time.time() - t0:.1f}s")

    # Phase 1
    if args.phase in ("1", "all"):
        print("\n" + "=" * 60)
        print("PHASE 1: DML Methods")
        print("=" * 60)
        obs_levels = [args.obs] if args.obs else OBS_LEVELS
        t0 = time.time()
        run_phase1(
            Y, W_full, A_true, dates, config, dgp_config,
            obs_levels=obs_levels,
            learner_names=learner_names,
            n_jobs=args.n_jobs,
            show_plots=show_plots, verbose=args.verbose,
        )
        print(f"\nPhase 1 total: {time.time() - t0:.1f}s")

    # Phase 2
    if args.phase == "2":
        print("\n" + "=" * 60)
        print("PHASE 2: TabPFN Methods (GPU)")
        print("=" * 60)
        obs_levels = [args.obs] if args.obs else OBS_LEVELS
        t0 = time.time()
        run_phase2(
            Y, W_full, A_true, dates, config, dgp_config,
            obs_levels=obs_levels,
            device=args.device,
            n_estimators=args.n_estimators,
            show_plots=show_plots, verbose=args.verbose,
        )
        print(f"\nPhase 2 total: {time.time() - t0:.1f}s")

    print("\nDone!")


if __name__ == "__main__":
    main()
