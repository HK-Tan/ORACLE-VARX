#!/usr/bin/env python3
"""Benchmark script for CPU vs GPU tree-based learners.

This script benchmarks tree-based learners training times across:
- Different numbers of features (10, 100)
- Different CPU core counts (1, 5, all)
- CPU vs GPU execution

Learners tested:
- XGBoost: CPU (multi-core) and GPU (CUDA)
- RandomForest: CPU (sklearn) and GPU (cuML)
- ExtraTrees: CPU only (sklearn, no GPU implementation)

The benchmark uses synthetic data to isolate training performance
from data loading overhead.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.modules.factory import get_multi_output_regressor

# =============================================================================
# Benchmark Configuration
# =============================================================================
N_ROWS = 1000
N_OUTPUTS = 10
N_FEATURES_LIST = [10, 100]
N_JOBS_LIST = [1, 5, int(os.cpu_count()/2)-1]  # 1 core, 5 cores, all cores
LEARNERS = ['xgboost', 'lgbm', 'rf', 'extra_trees']  # xgboost & rf have GPU, lgbm & extra_trees CPU-only


def check_gpu_available():
    """Check if GPU is available for training."""
    gpu_available = {'xgboost': False, 'lgbm': False, 'rf': False, 'extra_trees': False}

    # Check XGBoost GPU
    try:
        import xgboost as xgb
        # Try to create a GPU model and fit minimal data
        model = xgb.XGBRegressor(device='cuda', tree_method='hist', n_estimators=1)
        X_test = np.random.randn(10, 2).astype(np.float32)
        y_test = np.random.randn(10).astype(np.float32)
        model.fit(X_test, y_test)
        gpu_available['xgboost'] = True
    except Exception:
        pass

    # Check cuML RandomForest GPU
    try:
        from cuml.ensemble import RandomForestRegressor as cuRF
        model = cuRF(n_estimators=10, n_bins=64)  # n_bins <= n_samples to avoid warning
        X_test = np.random.randn(100, 2).astype(np.float32)
        y_test = np.random.randn(100).astype(np.float32)
        model.fit(X_test, y_test)
        gpu_available['rf'] = True
    except Exception:
        pass

    # LightGBM GPU requires special build, skipping for now
    gpu_available['lgbm'] = False

    # ExtraTrees has no GPU implementation
    gpu_available['extra_trees'] = False

    return gpu_available


def create_gpu_model(learner_name: str, n_jobs: int = 1):
    """Create a GPU-enabled model for the given learner.

    XGBoost supports multi-output natively.
    cuML RandomForest requires MultiOutputRegressor wrapper.

    Args:
        learner_name: Name of the learner
        n_jobs: Number of parallel jobs for MultiOutputRegressor wrapper
                (can help with data transfer/preprocessing)
    """
    if learner_name == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor(device='cuda', tree_method='hist')
    elif learner_name == 'rf':
        from cuml.ensemble import RandomForestRegressor as cuRF
        from sklearn.multioutput import MultiOutputRegressor
        # cuML RF doesn't support multi-output natively, wrap it
        # n_jobs controls parallel fitting of outputs (may help with data transfer)
        # n_bins=256 is default, works fine for N_ROWS >= 256
        return MultiOutputRegressor(cuRF(n_estimators=100, n_bins=256), n_jobs=n_jobs)
    elif learner_name == 'lgbm':
        raise ValueError("LightGBM GPU requires special build")
    elif learner_name == 'extra_trees':
        raise ValueError("ExtraTrees has no GPU implementation")
    else:
        raise ValueError(f"Unknown learner: {learner_name}")


def benchmark_single(model, X: np.ndarray, y: np.ndarray) -> float:
    """Benchmark a single model fit and return elapsed time."""
    start = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - start
    return elapsed


def generate_data(n_rows: int, n_features: int, n_outputs: int):
    """Generate random training data."""
    X = np.random.randn(n_rows, n_features).astype(np.float32)
    y = np.random.randn(n_rows, n_outputs).astype(np.float32)
    return X, y


def run_benchmarks():
    """Run all benchmark configurations and collect results."""
    print("=" * 80)
    print("CPU vs GPU Tree-Based Learners Benchmark")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  n_rows: {N_ROWS}")
    print(f"  n_outputs: {N_OUTPUTS}")
    print(f"  n_features: {N_FEATURES_LIST}")
    print(f"  n_jobs (CPU): {N_JOBS_LIST}")
    print(f"  learners: {LEARNERS}")
    print()

    # Check GPU availability
    print("Checking GPU availability...")
    gpu_available = check_gpu_available()
    for learner, available in gpu_available.items():
        status = "Available" if available else "Not available"
        print(f"  {learner}: {status}")
    print()

    # Collect results
    results = []

    for n_features in N_FEATURES_LIST:
        print(f"Generating data with {n_features} features...")
        X, y = generate_data(N_ROWS, n_features, N_OUTPUTS)

        for learner_name in LEARNERS:
            for n_jobs in N_JOBS_LIST:
                # CPU benchmark
                n_jobs_str = f"all ({n_jobs})" if n_jobs == os.cpu_count() else str(n_jobs)
                print(f"  Benchmarking {learner_name} (CPU, n_jobs={n_jobs_str})...", end=" ", flush=True)

                cpu_model = get_multi_output_regressor(learner_name, n_jobs=n_jobs)
                cpu_time = benchmark_single(cpu_model, X, y)
                print(f"{cpu_time:.3f}s")

                # GPU benchmark (run for each n_jobs to test parallelization benefits)
                gpu_time = None
                if gpu_available.get(learner_name, False):
                    print(f"  Benchmarking {learner_name} (GPU, n_jobs={n_jobs_str})...", end=" ", flush=True)
                    try:
                        gpu_model = create_gpu_model(learner_name, n_jobs=n_jobs)
                        gpu_time = benchmark_single(gpu_model, X, y)
                        print(f"{gpu_time:.3f}s")
                    except Exception as e:
                        print(f"Failed: {e}")
                        gpu_time = None

                results.append({
                    'learner': learner_name,
                    'n_features': n_features,
                    'n_jobs': n_jobs,
                    'cpu_time': cpu_time,
                    'gpu_time': gpu_time,
                })

        print()

    return results, gpu_available


def print_results_table(results: list, gpu_available: dict):
    """Print a formatted results table."""
    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()

    # Print table header
    header = f"{'Learner':<12} | {'n_features':<10} | {'n_jobs':<8} | {'CPU Time':<10} | {'GPU Time':<10} | {'Speedup':<10}"
    print(header)
    print("-" * len(header))

    # Print results
    for r in results:
        learner = r['learner']
        n_features = r['n_features']
        n_jobs = f"all ({r['n_jobs']})" if r['n_jobs'] == os.cpu_count() else str(r['n_jobs'])
        cpu_time = f"{r['cpu_time']:.3f}s"

        # GPU time is now per (learner, n_features, n_jobs)
        gpu_time_val = r['gpu_time']

        if gpu_time_val is not None:
            gpu_time = f"{gpu_time_val:.3f}s"
            speedup = r['cpu_time'] / gpu_time_val
            speedup_str = f"{speedup:.2f}x"
        else:
            if gpu_available.get(learner, False):
                gpu_time = "-"
                speedup_str = "-"
            else:
                gpu_time = "N/A"
                speedup_str = "N/A"

        row = f"{learner:<12} | {n_features:<10} | {n_jobs:<8} | {cpu_time:<10} | {gpu_time:<10} | {speedup_str:<10}"
        print(row)

    print()
    print("Notes:")
    print("  - GPU benchmarks run with varying n_jobs to test parallelization benefits")
    print("  - For cuML RF, n_jobs controls MultiOutputRegressor parallelism")
    print("  - Speedup = CPU Time / GPU Time (higher is better for GPU)")
    print("  - N/A indicates GPU not available for that learner")
    print()


def main():
    """Run benchmarks and print results."""
    results, gpu_available = run_benchmarks()
    print_results_table(results, gpu_available)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
