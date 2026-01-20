"""Simple verification script for OR-VARX model (batched version).

This script verifies that the OR-VARX model works correctly by:
- Using minimal test data (configurable via TOY_EXAMPLE)
- Running with the vectorized fit_orvarx_batched() function
- Timing each learner individually
- Checking that output shapes are correct
"""

import sys
import time
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import fit_orvarx_batched
from src.modules.grid_config import GridConfig

# =============================================================================
# Configuration
# =============================================================================
TOY_EXAMPLE = True  # Set to False for full test (~130k trees)
VERBOSE = True      # Set to True to see detailed progress (fold training)

# CPU cores for parallel training (5 recommended based on benchmarks, -1 for all)
N_CPU_CORES = 5

LEARNERS = ['extra_trees', 'rf', 'lgbm', 'xgboost']

# LEARNERS = ['lgbm']

def main():
    # Set parameters based on TOY_EXAMPLE mode
    # With lookback_orvarx=1018, minimum n_days = 1018 + validation_days + 1 = 1040
    if TOY_EXAMPLE:
        n_days, n_assets, p_max = 1061, 5, 3  # ~4+ years data
        validation_days = 21
        n_confounders = 1
    else:
        n_days, n_assets, p_max = 5000, 10, 10  # ~4.8 years data
        validation_days = 21  # Matches test_size for each fold
        n_confounders = 3

    Y = torch.randn(n_days, n_assets)
    W = torch.randn(n_days, n_confounders)

    config = GridConfig()  # defaults: lookback_orvarx=1018

    print(f"Running OR-VARX verification")
    print(f"  TOY_EXAMPLE: {TOY_EXAMPLE}")
    print(f"  n_days: {n_days}")
    print(f"  n_assets: {n_assets}")
    print(f"  n_confounders: {n_confounders}")
    print(f"  p_max: {p_max}")
    print(f"  lookback: {config.lookback_orvarx}")
    print(f"  validation_days: {validation_days}")
    print(f"  n_cpu_cores: {N_CPU_CORES}")
    print(f"  learners: {LEARNERS}")
    print()

    # Expected output shape (after validation burn-in)
    n_total_test_days = n_days - config.lookback_orvarx
    expected_output_days = n_total_test_days - validation_days
    expected_shape = (n_assets, expected_output_days)

    # Run all learners with timing
    results = {}
    for learner_name in LEARNERS:
        if VERBOSE:
            print(f"\n{'='*60}")
            print(f"Running {learner_name}...")
            print(f"{'='*60}")
        else:
            print(f"Running {learner_name}...", end=" ", flush=True)
        start = time.time()
        result = fit_orvarx_batched(
            Y, W,
            p_max=p_max,
            config=config,
            validation_days=validation_days,
            learner_name=learner_name,
            n_jobs=N_CPU_CORES,
            verbose=VERBOSE,
        )
        elapsed = time.time() - start
        results[learner_name] = {'result': result, 'time': elapsed}
        if VERBOSE:
            print(f"\n{learner_name} completed in {elapsed:.2f}s")
        else:
            print(f"{elapsed:.2f}s")

    # Print summary table
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"{'Learner':<15} {'Time (s)':<12} {'Shape':<20} {'Status'}")
    print("-" * 50)

    all_passed = True
    for learner_name in LEARNERS:
        entry = results[learner_name]
        actual_shape = entry['result'].forecasts.shape
        status = "PASS" if actual_shape == expected_shape else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{learner_name:<15} {entry['time']:<12.2f} {str(actual_shape):<20} {status}")

    print("-" * 50)
    total_time = sum(entry['time'] for entry in results.values())
    print(f"{'Total':<15} {total_time:<12.2f}")
    print()

    if all_passed:
        print("All learners PASSED")
    else:
        print("Some learners FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
