"""Data loading and preprocessing utilities for ORACLE-VARX project.

This module provides functions to load financial time series data including:
- Asset returns (OPCL data)
- Confounder variables (VIX, DFF, etc.)
- Tensor preparation for PyTorch models

Main Functions:
    - load_opcl_data: Load asset returns in DataFrame format
    - load_confounder: Load single confounder variable
    - prepare_tensors: Convert to PyTorch tensors for model training
    - load_test_data: Load standard test dataset (9 ETFs + VIX)
    - align_data: Align assets and confounders to common dates

Constants:
    - ETFS: List of 9 ETF tickers for testing
    - CONFOUNDER_FILES: Available confounder variables
    - DATA_DIR: Path to dataset directory
"""

from src.data.constants import (
    CONFOUNDER_FILES,
    DATA_DIR,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_P_MAX,
    DEFAULT_VALIDATION_DAYS,
    ETFS,
    OPCL_FILE,
    PROJECT_ROOT,
    TEST_DAYS,
    VIX_FILE,
)
from src.data.loader import (
    align_data,
    load_confounder,
    load_opcl_data,
    load_test_data,
    prepare_tensors,
)

__all__ = [
    # Loader functions
    "load_opcl_data",
    "load_confounder",
    "align_data",
    "prepare_tensors",
    "load_test_data",
    # Constants
    "ETFS",
    "CONFOUNDER_FILES",
    "DATA_DIR",
    "PROJECT_ROOT",
    "OPCL_FILE",
    "VIX_FILE",
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_VALIDATION_DAYS",
    "DEFAULT_P_MAX",
    "TEST_DAYS",
]
