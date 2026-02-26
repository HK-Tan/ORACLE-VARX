"""Compute Spearman and cross-correlation between p_optimal and SPY realized volatility.

For each experiment in results/, computes:
- Contemporaneous Spearman rho (k=0)
- Cross-correlation for k in [-21, +21] where positive k means vol leads p_optimal
- Peak cross-correlation lag and rho (max |rho| over all k)

Output: results/lag_vol_correlations.csv

Usage:
    python scripts/generate_lag_vol_correlations.py [--verbose]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_opcl_data
from src.results import VARXResult, ORACLEVARXResult, ACLEVARXResult

K_MIN, K_MAX = -21, 21
OUTPUT_FILE = PROJECT_ROOT / "results" / "lag_vol_correlations.csv"

# Map method_dir name → display name (matching toy metrics_summary.csv style)
METHOD_DISPLAY = {
    "var": "VAR",
    "varx": "VARX",
    "aclevar": "ACLE-VAR",
    "aclevarx": "ACLE-VARX",
    "orvarx": "OR-VARX",
    "oraclevarx": "ORACLE-VARX",
    "orvarx_tabpfn": "OR-VARX-TabPFN",
    "oraclevarx_tabpfn": "ORACLE-VARX-TabPFN",
}

TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")


def parse_experiment(method_dir_name: str, exp_name: str) -> tuple[str, str]:
    """Parse experiment name into (method, obs) following toy metrics_summary.csv format.

    Examples:
        ('oraclevarx', 'oraclevarx_vix_lgbm_20260222_103016') → ('ORACLE-VARX_lgbm', 'vix')
        ('orvarx_tabpfn', 'orvarx_tabpfn_all10_20260224_115747') → ('OR-VARX-TabPFN', 'all10')
        ('var', 'var_none_ols_20260222_025117') → ('VAR', 'none')
    """
    base_display = METHOD_DISPLAY[method_dir_name]

    # Strip method_dir prefix and timestamp suffix to get "preset_learner" or "preset"
    middle = exp_name[len(method_dir_name) + 1:]  # strip "{method_dir}_"
    middle = TIMESTAMP_RE.sub("", middle)          # strip "_YYYYMMDD_HHMMSS"

    parts = middle.split("_")
    obs = parts[0]

    if len(parts) > 1:
        learner = "_".join(parts[1:])
        # OLS learner is implicit for base methods — don't append
        if learner != "ols":
            method = f"{base_display}_{learner}"
        else:
            method = base_display
    else:
        method = base_display

    return method, obs


def compute_spy_realized_vol(vol_window: int = 21) -> pd.Series:
    """Load SPY log returns and compute rolling realized volatility."""
    spy_log_returns = load_opcl_data(tickers=["SPY"])["SPY"]
    spy_simple_returns = np.expm1(spy_log_returns).fillna(0.0)
    return spy_simple_returns.rolling(vol_window, min_periods=vol_window).std()


def load_result(results_pt_path: str):
    """Try loading results.pt with each result class."""
    for cls in (VARXResult, ORACLEVARXResult, ACLEVARXResult):
        try:
            return cls.load(results_pt_path, device="cpu")
        except (ValueError, KeyError):
            continue
    return None


def cross_correlate_spearman(
    p_optimal: np.ndarray,
    spy_vol: np.ndarray,
    k_min: int,
    k_max: int,
) -> pd.DataFrame:
    """Compute Spearman correlation at each lag k.

    Positive k: spy_vol[t] vs p_optimal[t+k] — vol leads p_optimal by k days.
    Negative k: spy_vol[t] vs p_optimal[t+k] — p_optimal leads vol by |k| days.
    """
    rows = []
    n = len(p_optimal)
    for k in range(k_min, k_max + 1):
        if k >= 0:
            vol_slice = spy_vol[:n - k] if k > 0 else spy_vol
            p_slice = p_optimal[k:]
        else:
            vol_slice = spy_vol[-k:]
            p_slice = p_optimal[:n + k]

        mask = ~(np.isnan(vol_slice) | np.isnan(p_slice))
        if mask.sum() < 10:
            continue

        rho, _ = stats.spearmanr(p_slice[mask], vol_slice[mask])
        rows.append({"k": k, "rho": rho})

    return pd.DataFrame(rows)


def process_experiment(
    method_dir_name: str,
    exp_dir: Path,
    spy_vol: pd.Series,
    verbose: bool,
) -> dict | None:
    """Process a single experiment. Returns a row dict or None."""
    results_pt = exp_dir / "results.pt"
    if not results_pt.exists():
        return None

    result = load_result(str(results_pt))
    if result is None:
        if verbose:
            print(f"  WARN {exp_dir.name}: could not load results.pt")
        return None

    # Extract p_optimal and dates
    p_optimal = result.p_optimal
    if hasattr(p_optimal, "cpu"):
        p_optimal = p_optimal.cpu().numpy()
    p_optimal = p_optimal.astype(float)

    dates = pd.to_datetime(result.dates)
    spy_vol_aligned = spy_vol.reindex(dates).values

    # Contemporaneous Spearman (k=0)
    mask_0 = ~(np.isnan(spy_vol_aligned) | np.isnan(p_optimal))
    if mask_0.sum() < 10:
        if verbose:
            print(f"  SKIP {exp_dir.name}: too few valid pairs")
        return None

    rho_0, _ = stats.spearmanr(p_optimal[mask_0], spy_vol_aligned[mask_0])

    # Cross-correlation sweep
    xcorr_df = cross_correlate_spearman(p_optimal, spy_vol_aligned, K_MIN, K_MAX)
    peak_idx = xcorr_df["rho"].abs().idxmax()
    peak = xcorr_df.loc[peak_idx]

    method, obs = parse_experiment(method_dir_name, exp_dir.name)

    if verbose:
        print(
            f"  {method} ({obs}): rho={rho_0:+.3f}, "
            f"peak at k={int(peak['k']):+d} rho={peak['rho']:+.3f}"
        )

    return {
        "method": method,
        "obs": obs,
        "spearman_rho": rho_0,
        "xcorr_peak_lag": int(peak["k"]),
        "xcorr_peak_rho": peak["rho"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Spearman and cross-correlation between p_optimal and SPY vol."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    if args.verbose:
        print("Loading SPY realized volatility...")
    spy_vol = compute_spy_realized_vol()

    rows = []
    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir() or method_dir.name not in METHOD_DISPLAY:
            continue
        # Take the most recent experiment per (method_dir, preset, learner)
        exp_dirs = sorted(
            [d for d in method_dir.iterdir() if d.is_dir() and (d / "results.pt").exists()],
            key=lambda d: d.name,
            reverse=True,  # most recent timestamp first
        )
        seen = set()
        for exp_dir in exp_dirs:
            method, obs = parse_experiment(method_dir.name, exp_dir.name)
            key = (method, obs)
            if key in seen:
                if args.verbose:
                    print(f"  SKIP duplicate: {exp_dir.name}")
                continue
            seen.add(key)

            row = process_experiment(method_dir.name, exp_dir, spy_vol, args.verbose)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows, columns=["method", "obs", "spearman_rho", "xcorr_peak_lag", "xcorr_peak_rho"])
    df = df.sort_values(["method", "obs"]).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")
    print(f"\nWrote {len(df)} experiments to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
