#!/usr/bin/env python3
"""Probe VRAM usage to test linearity with batch size.

Tests whether VRAM scales linearly with batch_size for different p values.
If linear, we can use 2-point extrapolation for optimal batch sizing.

Usage:
    export HF_TOKEN=your_token
    python scripts/probe_vram_linearity.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['TABPFN_NO_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

class _FakePosthog:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['posthog'] = _FakePosthog()

import numpy as np
import torch
from typing import List, Tuple


def measure_vram_for_batch(
    tabpfn,
    X_trains: List[np.ndarray],
    Y_trains: List[np.ndarray],
    X_tests: List[np.ndarray],
    batch_size: int,
) -> Tuple[float, float]:
    """Run inference and return (peak_vram_gb, allocated_vram_gb)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run inference
    _ = tabpfn.fit_predict_batch(
        X_trains[:batch_size],
        Y_trains[:batch_size],
        X_tests[:batch_size],
        batch_size=batch_size,
    )

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)

    torch.cuda.empty_cache()

    return peak_gb, allocated_gb


def test_linearity_for_p(p: int, n_folds: int = 5, n_train: int = 504, n_test: int = 21, n_outputs: int = 9):
    """Test VRAM linearity for a given p value."""
    from src.modules.batched_tabpfn import BatchedFoldTabPFN

    n_features = n_outputs * p  # Y-lags only (simplified)

    print(f"\np={p}, features={n_features}: ", end="", flush=True)

    # Create synthetic data
    np.random.seed(42)
    X_trains = [np.random.randn(n_train, n_features).astype(np.float32) for _ in range(n_folds)]
    Y_trains = [np.random.randn(n_train, n_outputs).astype(np.float32) for _ in range(n_folds)]
    X_tests = [np.random.randn(n_test, n_features).astype(np.float32) for _ in range(n_folds)]

    # Initialize TabPFN
    tabpfn = BatchedFoldTabPFN(n_estimators=8, device='cuda', random_state=42)

    # Just 3 batch sizes for quick linear fit
    batch_sizes = [1, 2, 3]

    results = []

    for batch_size in batch_sizes:
        try:
            peak_gb, _ = measure_vram_for_batch(
                tabpfn, X_trains, Y_trains, X_tests, batch_size
            )
            results.append((batch_size, peak_gb))
            print(f"b{batch_size}={peak_gb:.2f}GB ", end="", flush=True)
        except RuntimeError as e:
            print(f"b{batch_size}=OOM ", end="", flush=True)
            break

    # Analyze linearity with 2+ points
    if len(results) >= 2:
        batches = np.array([r[0] for r in results])
        peaks = np.array([r[1] for r in results])

        # Linear regression: peak = a + b * batch
        A = np.vstack([np.ones_like(batches), batches]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, peaks, rcond=None)
        fixed_overhead, per_batch_cost = coeffs

        # R² calculation
        ss_res = np.sum((peaks - (fixed_overhead + per_batch_cost * batches)) ** 2)
        ss_tot = np.sum((peaks - np.mean(peaks)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"→ VRAM={fixed_overhead:.2f}+{per_batch_cost:.3f}×batch, R²={r_squared:.3f}")

        return fixed_overhead, per_batch_cost, r_squared

    print("→ insufficient data")
    return None, None, None


def main():
    print("Checking GPU...")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {total_gb:.1f} GB")

    # Test key p values only (quick test)
    p_values = [1, 3, 5, 7, 10]

    all_results = []

    for p in p_values:
        try:
            fixed, per_batch, r2 = test_linearity_for_p(p, n_folds=20)
            if fixed is not None:
                all_results.append((p, fixed, per_batch, r2))
        except Exception as e:
            print(f"  Error for p={p}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY: Linear if R² > 0.95")
    print("=" * 50)

    all_linear = all(r2 > 0.95 for _, _, _, r2 in all_results) if all_results else False

    if all_linear:
        print("✓ VRAM scales linearly → 2-point extrapolation is valid!")
    else:
        print("✗ Non-linear for some p → need more sophisticated probing")


if __name__ == "__main__":
    main()
