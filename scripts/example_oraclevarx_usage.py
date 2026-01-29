"""Simple verification script for ORACLE-VARX model (batched version).

This script verifies that the ORACLE-VARX model works correctly by:
- Using minimal test data (configurable via TOY_EXAMPLE)
- Running with the batched fit_oraclevarx_batched() function
- Timing the full run
- Checking that output shapes are correct

Output Shape Formula:
    n_output_days = (n_days - lookback) - validation_days

This is the SAME formula as OR-VARX. Both models now use a clean, intuitive formula.
"""

import sys
import time
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.oracle_var import fit_oraclevarx_batched
from src.modules.grid_config import GridConfig

# =============================================================================
# Configuration
# =============================================================================
TOY_EXAMPLE = True  # Set to False for full test
VERBOSE = True      # Set to True to see detailed progress (fold training)

# CPU cores for parallel training (5 recommended based on benchmarks, -1 for all)
N_CPU_CORES = 5

# Learner to use (lgbm is fast and reliable)
LEARNER = 'lgbm'


def main():
    # Set parameters based on TOY_EXAMPLE mode
    # With lookback_orvarx=1018, minimum n_days = 1018 + validation_days + 1 = 1040
    if TOY_EXAMPLE:
        n_days, n_assets, p_max = 1061, 5, 3  # ~4+ years data
        validation_days = 21
        n_confounders = 1
        alpha_grid = [0.05, 0.10, 0.20]  # Smaller grid for faster testing
    else:
        n_days, n_assets, p_max = 5000, 10, 10  # ~4.8 years data
        validation_days = 21  # Matches test_size for each fold
        n_confounders = 3
        alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # Full grid

    Y = torch.randn(n_days, n_assets)
    W = torch.randn(n_days, n_confounders)

    config = GridConfig()  # defaults: lookback_orvarx=1018

    print(f"Running ORACLE-VARX verification")
    print(f"  TOY_EXAMPLE: {TOY_EXAMPLE}")
    print(f"  n_days: {n_days}")
    print(f"  n_assets: {n_assets}")
    print(f"  n_confounders: {n_confounders}")
    print(f"  p_max: {p_max}")
    print(f"  lookback: {config.lookback_orvarx}")
    print(f"  validation_days: {validation_days}")
    print(f"  alpha_grid: {alpha_grid}")
    print(f"  n_cpu_cores: {N_CPU_CORES}")
    print(f"  learner: {LEARNER}")
    print()

    # Expected output shapes (after validation burn-in)
    # ORACLE-VARX now uses a CLEAN formula - same as OR-VARX:
    # n_output_days = n_total_test_days - validation_days
    # No more -1 hack from calling fit_orvarx_batched with validation_days=1
    n_total_test_days = n_days - config.lookback_orvarx
    expected_output_days = n_total_test_days - validation_days
    expected_forecasts_shape = (n_assets, expected_output_days)
    expected_forecasts_all_shape = (n_assets, expected_output_days, len(alpha_grid))
    expected_p_optimal_shape = (expected_output_days,)
    expected_alpha_optimal_shape = (expected_output_days,)

    # Run ORACLE-VARX with timing
    if VERBOSE:
        print(f"{'='*60}")
        print(f"Running {LEARNER}...")
        print(f"{'='*60}")
    else:
        print(f"Running {LEARNER}...", end=" ", flush=True)

    start = time.time()
    result = fit_oraclevarx_batched(
        Y, W,
        alpha_grid=alpha_grid,
        p_max=p_max,
        config=config,
        validation_days=validation_days,
        learner_name=LEARNER,
        n_jobs=N_CPU_CORES,
        verbose=VERBOSE,
    )
    elapsed = time.time() - start

    if VERBOSE:
        print(f"\n{LEARNER} completed in {elapsed:.2f}s")
    else:
        print(f"{elapsed:.2f}s")

    # Print summary table
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Field':<20} {'Expected':<25} {'Actual':<25} {'Status'}")
    print("-" * 70)

    all_passed = True

    # Check forecasts shape
    actual_forecasts = result.forecasts.shape
    status = "PASS" if actual_forecasts == expected_forecasts_shape else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"{'forecasts':<20} {str(expected_forecasts_shape):<25} {str(actual_forecasts):<25} {status}")

    # Check forecasts_all shape
    actual_forecasts_all = result.forecasts_all.shape
    status = "PASS" if actual_forecasts_all == expected_forecasts_all_shape else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"{'forecasts_all':<20} {str(expected_forecasts_all_shape):<25} {str(actual_forecasts_all):<25} {status}")

    # Check p_optimal shape
    actual_p_optimal = result.p_optimal.shape
    status = "PASS" if actual_p_optimal == expected_p_optimal_shape else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"{'p_optimal':<20} {str(expected_p_optimal_shape):<25} {str(actual_p_optimal):<25} {status}")

    # Check alpha_optimal shape
    actual_alpha_optimal = result.alpha_optimal.shape
    status = "PASS" if actual_alpha_optimal == expected_alpha_optimal_shape else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"{'alpha_optimal':<20} {str(expected_alpha_optimal_shape):<25} {str(actual_alpha_optimal):<25} {status}")

    # Check method
    expected_method = "ORACLE-VARX"
    actual_method = result.method
    status = "PASS" if actual_method == expected_method else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"{'method':<20} {expected_method:<25} {actual_method:<25} {status}")

    # Check alpha_grid
    expected_alpha_grid = alpha_grid
    actual_alpha_grid = result.alpha_grid
    status = "PASS" if actual_alpha_grid == expected_alpha_grid else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"{'alpha_grid':<20} {str(expected_alpha_grid):<25} {str(actual_alpha_grid):<25} {status}")

    print("-" * 70)
    print(f"{'Total time':<20} {elapsed:.2f}s")
    print()

    if all_passed:
        print("All checks PASSED")
    else:
        print("Some checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
