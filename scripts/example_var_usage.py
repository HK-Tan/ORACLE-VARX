"""Benchmark script for VAR model fitting.

This script measures the performance of the VAR model across different
dataset sizes and verifies that p_optimal is properly day-dependent.
"""

import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
from src.models import fit_var
from src.modules.grid_config import GridConfig


def benchmark_var(
    n_days_list: list,
    n_assets: int = 10,
    config: GridConfig = None,
    validation_days: int = 21,
    p_max: int = 10,
    device: str = "cpu",
):
    """Benchmark VAR fitting across different dataset sizes.

    Args:
        n_days_list: List of dataset sizes to test
        n_assets: Number of assets
        config: GridConfig for lookback settings (default: GridConfig())
        validation_days: Validation window size
        p_max: Maximum lag order
        device: Device to run on ("cpu" or "cuda")

    Returns:
        List of result dictionaries
    """
    if config is None:
        config = GridConfig()
    lookback = config.lookback_var
    print(f"\nBenchmarking VAR model (lookback={lookback}, validation_days={validation_days}, p_max={p_max})")
    print(f"Device: {device}")
    print("-" * 80)
    print("n_distinct_p = number of distinct lag orders selected across days (shows p varies per day)")
    print("-" * 80)

    results = []

    for n_days in n_days_list:
        # Generate random data on CPU (simulating real-world data loading)
        Y_cpu = torch.randn(n_days, n_assets)

        # Time data transfer (if GPU)
        transfer_time = 0.0
        if device == "cuda":
            start = time.perf_counter()
            Y = Y_cpu.to(device)
            torch.cuda.synchronize()
            transfer_time = time.perf_counter() - start
        else:
            Y = Y_cpu

        # Time the fit (excluding data transfer)
        start = time.perf_counter()
        result = fit_var(
            Y,
            p_max=p_max,
            config=config,
            validation_days=validation_days,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Check p_optimal variability
        p_unique = torch.unique(result.p_optimal)
        n_unique = len(p_unique)

        # Expected output days
        expected_output_days = n_days - lookback - validation_days

        results.append({
            "n_days": n_days,
            "n_output_days": result.forecasts.shape[1],
            "expected_output_days": expected_output_days,
            "transfer_seconds": transfer_time,
            "compute_seconds": elapsed,
            "total_seconds": transfer_time + elapsed,
            "p_optimal_min": result.p_optimal.min().item(),
            "p_optimal_max": result.p_optimal.max().item(),
            "p_optimal_unique": n_unique,
        })

        if device == "cuda":
            print(
                f"n_days={n_days:5d} | "
                f"output_days={result.forecasts.shape[1]:4d} | "
                f"transfer={transfer_time:.3f}s | "
                f"compute={elapsed:.3f}s | "
                f"total={transfer_time + elapsed:.3f}s | "
                f"n_distinct_p={n_unique}"
            )
        else:
            print(
                f"n_days={n_days:5d} | "
                f"output_days={result.forecasts.shape[1]:4d} | "
                f"compute={elapsed:.3f}s | "
                f"n_distinct_p={n_unique}"
            )

    return results


def benchmark_cpu_threads(
    n_days: int,
    thread_counts: list,
    n_assets: int = 10,
    config: GridConfig = None,
    validation_days: int = 21,
    p_max: int = 10,
):
    """Benchmark CPU performance across different thread counts.

    Args:
        n_days: Dataset size to test
        thread_counts: List of thread counts to test
        n_assets: Number of assets
        config: GridConfig for lookback settings (default: GridConfig())
        validation_days: Validation window size
        p_max: Maximum lag order

    Returns:
        List of result dictionaries
    """
    if config is None:
        config = GridConfig()
    print(f"\nCPU Thread Scaling (n_days={n_days}, n_assets={n_assets})")
    print("-" * 60)

    # Store original thread count
    original_threads = torch.get_num_threads()

    results = []
    Y = torch.randn(n_days, n_assets)

    for n_threads in thread_counts:
        torch.set_num_threads(n_threads)

        start = time.perf_counter()
        result = fit_var(Y, p_max=p_max, config=config, validation_days=validation_days)
        elapsed = time.perf_counter() - start

        results.append({
            "n_threads": n_threads,
            "time_seconds": elapsed,
        })

        print(f"threads={n_threads:2d} | time={elapsed:.3f}s")

    # Restore original thread count
    torch.set_num_threads(original_threads)

    # Print scaling summary
    if len(results) >= 2:
        single_thread_time = results[0]["time_seconds"]
        print("-" * 60)
        print("Speedup vs single thread:")
        for r in results:
            speedup = single_thread_time / r["time_seconds"]
            print(f"  threads={r['n_threads']:2d} | speedup={speedup:.2f}x")

    return results


def print_summary(results: list):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Verify shapes match
    all_match = all(r["n_output_days"] == r["expected_output_days"] for r in results)
    print(f"Output shape verification: {'PASS' if all_match else 'FAIL'}")

    # Check if p_optimal varies
    any_variation = any(r["p_optimal_unique"] > 1 for r in results)
    print(f"p_optimal day-dependent: {'YES' if any_variation else 'NO (may indicate issue or homogeneous data)'}")


if __name__ == "__main__":
    # Default config uses lookback_var=514 (504 ols_window + 10 p_max_offset)
    config = GridConfig()

    # Test with increasing data sizes
    # Minimum required: lookback_var + validation_days + 1 = 514 + 21 + 1 = 536
    # For meaningful test period: 514 + 100 = 614
    n_days_list = [600, 800, 1000, 2000]

    # Check for CUDA
    has_cuda = torch.cuda.is_available()

    # =========================================================================
    # SYSTEM INFO
    # =========================================================================
    print("=" * 80)
    print("SYSTEM INFO")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads available: {torch.get_num_threads()}")

    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_cores = torch.cuda.get_device_properties(0).multi_processor_count
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem:.1f} GB")
        print(f"GPU SMs (Streaming Multiprocessors): {gpu_cores}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("GPU: None (CUDA not available)")
    print()

    # =========================================================================
    # COLD START OVERHEAD (measured together for fair comparison)
    # =========================================================================
    print("=" * 80)
    print("COLD START OVERHEAD (n_days=900, first run on each device)")
    print("=" * 80)

    # CPU cold start
    Y_cpu_warmup = torch.randn(900, 10, device="cpu")
    start = time.perf_counter()
    _ = fit_var(Y_cpu_warmup, p_max=10, config=config, validation_days=21)
    cpu_coldstart = time.perf_counter() - start
    print(f"CPU cold start:  {cpu_coldstart:.3f}s")

    # CUDA cold start (if available)
    if has_cuda:
        Y_cuda_warmup = torch.randn(900, 10, device="cuda")
        start = time.perf_counter()
        _ = fit_var(Y_cuda_warmup, p_max=10, config=config, validation_days=21)
        torch.cuda.synchronize()
        cuda_coldstart = time.perf_counter() - start
        print(f"CUDA cold start: {cuda_coldstart:.3f}s (includes kernel compilation, memory allocation)")

    print()

    # =========================================================================
    # BENCHMARKS (after warmup)
    # =========================================================================
    if has_cuda:
        cuda_results = benchmark_var(n_days_list, device="cuda")
        print_summary(cuda_results)
        print()

    cpu_results = benchmark_var(n_days_list, device="cpu")
    print_summary(cpu_results)

    # Print speedup comparison if CUDA was available
    if has_cuda:
        print("\n" + "=" * 80)
        print("CUDA vs CPU SPEEDUP (after warmup)")
        print("=" * 80)
        print(f"Note: CPU using {torch.get_num_threads()} threads")
        print("-" * 80)
        for cuda_r, cpu_r in zip(cuda_results, cpu_results):
            compute_speedup = cpu_r["compute_seconds"] / cuda_r["compute_seconds"]
            print(
                f"n_days={cuda_r['n_days']:5d} | "
                f"CPU={cpu_r['compute_seconds']:.3f}s | "
                f"CUDA={cuda_r['compute_seconds']:.3f}s | "
                f"speedup={compute_speedup:.1f}x"
            )

    # =========================================================================
    # CPU THREAD SCALING TEST
    # =========================================================================
    print("\n" + "=" * 80)
    print("CPU THREAD SCALING TEST")
    print("=" * 80)
    print(f"Default thread count: {torch.get_num_threads()}")

    # Test thread scaling for n_days=5000 (where CPU beat GPU)
    thread_counts = [1, 4, 11]
    # Filter to available threads
    max_threads = torch.get_num_threads()
    thread_counts = [t for t in thread_counts if t <= max_threads]
    if max_threads not in thread_counts:
        thread_counts.append(max_threads)
    thread_counts.sort()

    thread_results_5000 = benchmark_cpu_threads(5000, thread_counts)

    # =========================================================================
    # CHUNKED PROCESSING TEST
    # =========================================================================
    if has_cuda:
        print("\n" + "=" * 80)
        print("CHUNKED PROCESSING TEST (n_days=5000 on CUDA)")
        print("=" * 80)
        print("Testing if processing in smaller chunks improves CUDA performance")
        print("-" * 80)

        from src.models.var_pytorch import batch_var_all_days

        n_days_test = 5000
        Y_test = torch.randn(n_days_test, 10, device="cuda")
        lookback_test = config.lookback_var  # 514 by default
        p_max_test = 10

        # Test different chunk sizes (None = no chunking)
        chunk_sizes = [None, 1000, 500, 100]

        for chunk_size in chunk_sizes:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            forecasts_all, coefficients = batch_var_all_days(
                Y_test, p_max_test, lookback_test, chunk_size=chunk_size
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            label = "full batch" if chunk_size is None else f"chunk={chunk_size}"
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(
                f"{label:12s} | time={elapsed:.3f}s | output_shape={forecasts_all.shape} | "
                f"peak_alloc={peak_alloc:.2f}GB"
            )

    # =========================================================================
    # SKLEARN VALIDATION TEST
    # =========================================================================
    print("\n" + "=" * 80)
    print("SKLEARN VALIDATION TEST")
    print("=" * 80)
    print("Comparing PyTorch VAR coefficients against sklearn LinearRegression")
    print("-" * 80)

    from sklearn.linear_model import LinearRegression
    import numpy as np
    from src.models.var_pytorch import batch_var_all_days

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    n_assets_val = 10
    lookback_val = config.lookback_var  # 514 by default
    Y_val = torch.randn(lookback_val + 1, n_assets_val)  # Minimum needed for 1 output day
    p_val = 3  # Test with a specific p

    # Get PyTorch coefficients for first output day
    forecasts_torch, coefs_torch = batch_var_all_days(Y_val, p_max=p_val, lookback=lookback_val)

    # Extract coefficients for day 0, p=p_val
    # Shape: coefs_torch[day, lag_idx, target_asset, source_asset]
    coefs_day0 = coefs_torch[0, :p_val, :, :]  # (p, n_assets, n_assets)

    # Build sklearn regression manually for day 0
    # Training window is Y_val[0:lookback_val]
    window = Y_val[0:lookback_val].numpy()

    # Build X and Y matrices (same logic as build_var_design_batch)
    T = lookback_val - p_val
    X_sklearn = np.ones((T, 1 + n_assets_val * p_val))
    Y_sklearn = window[p_val:, :]  # (T, n_assets)

    for t in range(T):
        for lag in range(1, p_val + 1):
            start_col = 1 + (lag - 1) * n_assets_val
            X_sklearn[t, start_col:start_col + n_assets_val] = window[p_val + t - lag, :]

    # Fit sklearn for each target asset and compare
    max_diff = 0.0
    for target in range(n_assets_val):
        lr = LinearRegression(fit_intercept=False)  # Intercept already in X
        lr.fit(X_sklearn, Y_sklearn[:, target])

        # Extract lag coefficients (skip intercept)
        sklearn_coefs = lr.coef_[1:].reshape(p_val, n_assets_val)  # (p, n_assets)
        torch_coefs = coefs_day0[:, target, :].numpy()  # (p, n_assets)

        diff = np.abs(sklearn_coefs - torch_coefs).max()
        max_diff = max(max_diff, diff)

    print(f"Max coefficient difference: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("PASS: PyTorch VAR matches sklearn LinearRegression")
    else:
        print("FAIL: Coefficients differ significantly")
