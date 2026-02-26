"""Generate lag analysis plots with SPY realized volatility overlay.

Walks all results/{method}/{experiment}/ directories, loads results.pt,
extracts p_optimal and dates, and generates a dual-axis plot overlaying
SPY 21-day realized volatility.

Usage:
    python scripts/generate_lag_vol_plots.py [--force] [--verbose]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_opcl_data
from src.evaluation.plotting import plot_lag_analysis_with_volatility
from src.results import VARXResult, ORACLEVARXResult, ACLEVARXResult


OUTPUT_FILENAME = "lag_analysis_with_realized_volatility.png"


def compute_spy_realized_vol(vol_window: int = 21) -> pd.Series:
    """Load SPY log returns and compute rolling realized volatility."""
    spy_log_returns = load_opcl_data(tickers=["SPY"])["SPY"]
    spy_simple_returns = np.expm1(spy_log_returns).fillna(0.0)
    spy_vol = spy_simple_returns.rolling(vol_window, min_periods=vol_window).std()
    return spy_vol


def load_result(results_pt_path: str):
    """Try loading results.pt with each result class, return (result, class_name) or None."""
    loaders = [
        (VARXResult, "VARXResult"),
        (ORACLEVARXResult, "ORACLEVARXResult"),
        (ACLEVARXResult, "ACLEVARXResult"),
    ]
    for cls, name in loaders:
        try:
            return cls.load(results_pt_path, device="cpu"), name
        except (ValueError, KeyError):
            continue
    return None, None


def process_experiment(
    exp_dir: Path,
    spy_vol: pd.Series,
    force: bool,
    verbose: bool,
) -> bool:
    """Process a single experiment directory. Returns True if plot was generated."""
    output_path = exp_dir / OUTPUT_FILENAME
    results_pt = exp_dir / "results.pt"

    if not results_pt.exists():
        return False

    if output_path.exists() and not force:
        if verbose:
            print(f"  SKIP {exp_dir.relative_to(PROJECT_ROOT)} (already exists)")
        return False

    # Load result
    result, cls_name = load_result(str(results_pt))
    if result is None:
        if verbose:
            print(f"  WARN {exp_dir.relative_to(PROJECT_ROOT)}: could not load results.pt")
        return False

    # Extract p_optimal and dates
    p_optimal = result.p_optimal
    if hasattr(p_optimal, "cpu"):
        p_optimal = p_optimal.cpu().numpy()

    dates_list = result.dates
    dates = pd.to_datetime(dates_list)

    # Align SPY vol to experiment dates
    spy_vol_aligned = spy_vol.reindex(dates)

    # Determine significance level for title (only for ORACLE/ACLE)
    sig_level = None
    if hasattr(result, "alpha_optimal"):
        # Use median alpha as representative significance level
        alpha_opt = result.alpha_optimal
        if hasattr(alpha_opt, "cpu"):
            alpha_opt = alpha_opt.cpu().numpy()
        sig_level = float(np.median(alpha_opt))

    # Build title from experiment directory name
    exp_name = exp_dir.name
    title = f"p_optimal Over Time — {exp_name}"

    # Generate plot
    plot_lag_analysis_with_volatility(
        p_optimal=p_optimal,
        spy_volatility=spy_vol_aligned,
        dates=dates,
        significance_level=sig_level,
        title=title,
        save_path=str(output_path),
        show_plot=False,
    )

    if verbose:
        print(f"  DONE {exp_dir.relative_to(PROJECT_ROOT)} ({cls_name}, {len(dates)} days)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate lag analysis plots with SPY realized volatility overlay."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing plots")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    # Load SPY volatility once
    if args.verbose:
        print("Loading SPY realized volatility...")
    spy_vol = compute_spy_realized_vol()
    if args.verbose:
        print(f"  SPY vol computed: {spy_vol.index.min().date()} to {spy_vol.index.max().date()}")

    # Walk all method/experiment directories
    generated = 0
    skipped = 0
    total = 0

    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        for exp_dir in sorted(method_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if not (exp_dir / "results.pt").exists():
                continue
            total += 1
            if process_experiment(exp_dir, spy_vol, args.force, args.verbose):
                generated += 1
            else:
                skipped += 1

    print(f"\nSummary: {generated} generated, {skipped} skipped, {total} total experiments")


if __name__ == "__main__":
    main()
