#!/usr/bin/env python3
"""Example usage of data loading utilities.

This script demonstrates how to use the data loader functions
for various scenarios in the ORACLE-VARX project.
"""

import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data import (
    load_opcl_data,
    load_confounder,
    prepare_tensors,
    load_test_data,
    ETFS,
)


def example_1_basic_opcl():
    """Example 1: Load basic OPCL returns data."""
    print("=" * 70)
    print("Example 1: Load OPCL Returns for Selected ETFs")
    print("=" * 70)

    # Load returns for 3 ETFs over 200 days
    df = load_opcl_data(tickers=["SPY", "XLF", "XLE"], n_days=200)

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    print()


def example_2_load_confounder():
    """Example 2: Load VIX confounder variable."""
    print("=" * 70)
    print("Example 2: Load VIX Confounder")
    print("=" * 70)

    # Load VIX data
    vix = load_confounder("VIX", n_days=200)

    print(f"Shape: {vix.shape}")
    print(f"Date range: {vix.index[0].date()} to {vix.index[-1].date()}")
    print(f"\nFirst 10 values:\n{vix.head(10)}")
    print(f"\nStatistics:\n{vix.describe()}")
    print()


def example_3_prepare_for_training():
    """Example 3: Prepare data for model training."""
    print("=" * 70)
    print("Example 3: Prepare Tensors for Model Training")
    print("=" * 70)

    # Prepare data for VAR model training
    Y, W, dates, tickers = prepare_tensors(
        tickers=["SPY", "XLF", "XLE", "XLV", "XLK"],
        confounder_names=["VIX"],
        n_days=500,
        device="cpu",
    )

    print(f"Y (returns) shape: {Y.shape}")
    print(f"W (confounders) shape: {W.shape}")
    print(f"Number of dates: {len(dates)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Tickers: {tickers}")
    print(f"\nY statistics:")
    print(f"  Mean: {Y.mean(dim=0)}")
    print(f"  Std: {Y.std(dim=0)}")
    print(f"\nW statistics:")
    print(f"  Mean: {W.mean():.2f}")
    print(f"  Std: {W.std():.2f}")
    print()


def example_4_multiple_confounders():
    """Example 4: Use multiple confounder variables."""
    print("=" * 70)
    print("Example 4: Multiple Confounders (VIX + DFF)")
    print("=" * 70)

    # Load with multiple confounders
    Y, W, dates, tickers = prepare_tensors(
        tickers=["SPY", "XLF"],
        confounder_names=["VIX", "DFF"],
        n_days=300,
    )

    print(f"Y (returns) shape: {Y.shape}")
    print(f"W (confounders) shape: {W.shape}")
    print(f"Confounders: VIX (column 0), DFF (column 1)")
    print(f"\nFirst 5 days of confounders:")
    print(W[:5])
    print()


def example_5_no_confounders():
    """Example 5: Load data without confounders."""
    print("=" * 70)
    print("Example 5: Asset Returns Only (No Confounders)")
    print("=" * 70)

    # Prepare data without confounders (standard VAR model)
    Y, W, dates, tickers = prepare_tensors(
        tickers=["SPY", "XLF", "XLE"],
        confounder_names=None,  # No confounders
        n_days=200,
    )

    print(f"Y (returns) shape: {Y.shape}")
    print(f"W (confounders): {W}")
    print(f"Tickers: {tickers}")
    print(f"\nThis is suitable for standard VAR models without exogenous variables.")
    print()


def example_6_standard_test_dataset():
    """Example 6: Load standard test dataset."""
    print("=" * 70)
    print("Example 6: Standard Test Dataset (9 ETFs + VIX, 624 days)")
    print("=" * 70)

    # Load the standard test configuration
    Y, W, dates, tickers = load_test_data(n_days=624, device="cpu")

    print(f"Y (returns) shape: {Y.shape}")
    print(f"W (VIX) shape: {W.shape}")
    print(f"Number of ETFs: {len(tickers)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"\nETFs included:")
    for i, ticker in enumerate(tickers, 1):
        print(f"  {i:2d}. {ticker}")
    print(f"\nData characteristics:")
    print(f"  Y dtype: {Y.dtype}")
    print(f"  Y device: {Y.device}")
    print(f"  Y value range: [{Y.min():.4f}, {Y.max():.4f}]")
    print(f"  W value range: [{W.min():.2f}, {W.max():.2f}]")
    print()


def example_7_full_dataset():
    """Example 7: Load full available dataset."""
    print("=" * 70)
    print("Example 7: Load Full Available Dataset")
    print("=" * 70)

    # Load all available data (no n_days limit)
    Y, W, dates, tickers = prepare_tensors(
        tickers=ETFS,
        confounder_names=["VIX"],
        n_days=None,  # Load all available data
    )

    print(f"Y (returns) shape: {Y.shape}")
    print(f"W (VIX) shape: {W.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Total trading days: {len(dates)}")
    print(f"Number of years: {(len(dates) / 252):.1f} (approx)")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("ORACLE-VARX Data Loader - Usage Examples")
    print("*" * 70)
    print()

    # Run all examples
    example_1_basic_opcl()
    example_2_load_confounder()
    example_3_prepare_for_training()
    example_4_multiple_confounders()
    example_5_no_confounders()
    example_6_standard_test_dataset()
    example_7_full_dataset()

    print("*" * 70)
    print("All examples completed successfully!")
    print("*" * 70)
    print()


if __name__ == "__main__":
    main()
