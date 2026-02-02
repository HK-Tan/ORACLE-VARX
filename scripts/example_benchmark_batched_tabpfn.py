#!/usr/bin/env python3
"""Benchmark script for BatchedFoldTabPFN implementation.

Compares sequential vs batched TabPFN inference to verify:
1. Prediction accuracy matches (within tolerance)
2. Speedup is achieved with multi-fold batching

Usage:
    python scripts/example_benchmark_batched_tabpfn.py
"""

import os
import sys
import time
import warnings
from typing import Tuple, List

import numpy as np
import torch

# Suppress warnings during benchmarking
warnings.filterwarnings('ignore')


def benchmark_sequential_tabpfn(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 8,
) -> Tuple[np.ndarray, float]:
    """Benchmark sequential TabPFN (one output at a time).

    Args:
        X_train: Training features, shape (n_train, n_features)
        Y_train: Training targets, shape (n_train, n_outputs)
        X_test: Test features, shape (n_test, n_features)
        n_estimators: Number of ensemble members

    Returns:
        predictions: shape (n_test, n_outputs)
        elapsed_time: seconds
    """
    from tabpfn import TabPFNRegressor

    n_outputs = Y_train.shape[1]
    n_test = X_test.shape[0]
    predictions = np.zeros((n_test, n_outputs), dtype=np.float32)

    start_time = time.time()

    for output_idx in range(n_outputs):
        model = TabPFNRegressor(
            device='cuda',
            n_estimators=n_estimators,
            random_state=42,
        )
        model.fit(X_train, Y_train[:, output_idx])
        predictions[:, output_idx] = model.predict(X_test)

    elapsed = time.time() - start_time
    return predictions, elapsed


def benchmark_batched_fold_tabpfn(
    X_trains: List[np.ndarray],
    Y_trains: List[np.ndarray],
    X_tests: List[np.ndarray],
    n_estimators: int = 8,
    batch_size: int = 50,
) -> Tuple[np.ndarray, float]:
    """Benchmark BatchedFoldTabPFN (true batching using transformer batch dimension).

    Args:
        X_trains: List of training features
        Y_trains: List of training targets
        X_tests: List of test features
        n_estimators: Number of ensemble members
        batch_size: Folds per batch

    Returns:
        predictions: shape (n_folds, n_test, n_outputs)
        elapsed_time: seconds
    """
    from src.modules.batched_tabpfn import BatchedFoldTabPFN

    model = BatchedFoldTabPFN(
        n_estimators=n_estimators,
        device='cuda',
        random_state=42,
    )

    start_time = time.time()
    predictions = model.fit_predict_batch(X_trains, Y_trains, X_tests, batch_size)
    elapsed = time.time() - start_time

    return predictions, elapsed


def check_requirements():
    """Check that GPU and HuggingFace token are available."""
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


def run_quick_benchmark():
    """Run quick benchmark with small data."""
    print("=" * 60)
    print("Quick Benchmark: Batched TabPFN")
    print("=" * 60)

    # Check requirements
    check_requirements()

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print()

    # Generate small test data
    n_train, n_test, n_features, n_outputs = 100, 20, 10, 9
    n_est = 8
    np.random.seed(42)

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    Y_train = np.random.randn(n_train, n_outputs).astype(np.float32)
    X_test = np.random.randn(n_test, n_features).astype(np.float32)

    print(f"Data: {n_train} train, {n_test} test, {n_features} features, {n_outputs} outputs")
    print()

    # Warm up GPU
    print("Warming up GPU...")
    _ = torch.randn(1000, 1000, device='cuda') @ torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()

    # 1. Benchmark sequential (single fold)
    print(f"\n1. Sequential TabPFN (single fold, n_estimators={n_est}):")
    seq_preds, seq_time = benchmark_sequential_tabpfn(X_train, Y_train, X_test, n_estimators=n_est)
    print(f"   Time: {seq_time:.2f}s")
    print(f"   Predictions shape: {seq_preds.shape}")

    # 2. Benchmark BatchedFoldTabPFN (single fold) for comparison
    print(f"\n2. BatchedFoldTabPFN (single fold, n_estimators={n_est}):")
    batched_preds, batched_time = benchmark_batched_fold_tabpfn(
        [X_train], [Y_train], [X_test], n_estimators=n_est
    )
    print(f"   Time: {batched_time:.2f}s")
    print(f"   Predictions shape: {batched_preds.shape}")

    # 3. Compare predictions
    print(f"\n3. Prediction comparison:")
    # batched_preds is (1, n_test, n_outputs), squeeze to compare with seq_preds
    batched_preds_squeezed = batched_preds[0]
    max_diff = np.abs(seq_preds - batched_preds_squeezed).max()
    mean_diff = np.abs(seq_preds - batched_preds_squeezed).mean()
    print(f"   Max absolute difference: {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")

    # Tolerance check - same seed and n_estimators should give similar results
    # but implementation differences mean we allow some tolerance
    if max_diff < 0.5:
        print("   PASS: Predictions match within tolerance")
    else:
        print("   WARNING: Predictions differ significantly")

    # 4. Benchmark BatchedFoldTabPFN (23 folds) for multi-fold speedup
    print(f"\n4. BatchedFoldTabPFN ({23} folds, n_estimators={n_est}):")
    n_folds = 23
    X_trains = [X_train for _ in range(n_folds)]
    Y_trains = [Y_train for _ in range(n_folds)]
    X_tests = [X_test for _ in range(n_folds)]

    fold_preds, fold_time = benchmark_batched_fold_tabpfn(
        X_trains, Y_trains, X_tests, n_estimators=n_est
    )
    print(f"   Time: {fold_time:.2f}s")
    print(f"   Time per fold: {fold_time/n_folds:.2f}s")
    print(f"   Predictions shape: {fold_preds.shape}")

    # Calculate speedup vs sequential
    print(f"\n5. Speedup analysis:")
    seq_fold_time = seq_time * n_folds  # Estimated sequential time for all folds
    print(f"   Sequential (estimated for {n_folds} folds): {seq_fold_time:.2f}s")
    print(f"   BatchedFoldTabPFN ({n_folds} folds):         {fold_time:.2f}s")
    if fold_time > 0:
        print(f"   Speedup: {seq_fold_time/fold_time:.1f}x")

    print("\n" + "=" * 60)
    print("Quick benchmark complete!")
    print("=" * 60)


if __name__ == '__main__':
    run_quick_benchmark()
