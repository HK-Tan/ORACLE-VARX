"""Project-wide constants and configuration parameters."""

from pathlib import Path
from typing import Dict

# ============================================================================
# Project Directories
# ============================================================================

PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
"""Project root directory (ORACLE-VARX/)."""

DATA_DIR: Path = PROJECT_ROOT / "dataset"
"""Data directory containing all dataset files."""

# ============================================================================
# Asset Universe
# ============================================================================

ETFS: list[str] = [
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLE",  # Energy
    "XLF",  # Financials
    "XLV",  # Healthcare
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLK",  # Technology
    "XLU",  # Utilities
]
"""ETF tickers for analysis (9 ETFs)."""

ETFS_WITH_VIX: list[str] = ETFS + ["VIX"]
"""ETF tickers plus VIX as a tradeable asset (10 assets)."""

# ============================================================================
# Data Files
# ============================================================================

OPCL_FILE: Path = DATA_DIR / "OPCL_20000103_20201231.csv"
"""Open-Close price data file (2000-01-03 to 2020-12-31)."""

VIX_FILE: Path = DATA_DIR / "VIX_20000103_20201231.csv"
"""VIX volatility index file (2000-01-03 to 2020-12-31)."""

CONFOUNDER_FILES: Dict[str, Path] = {
    "VIX": VIX_FILE,
    "DFF": DATA_DIR / "DFF_20000103_20201231.csv",
    "T5YIE": DATA_DIR / "T5YIE_20030102_20201231.csv",
    "DCOILWTICO": DATA_DIR / "DCOILWTICO_20000103_20201231.csv",
    "USEPUINDXD": DATA_DIR / "USEPUINDXD_20000103_20201231.csv",
}
"""Confounder/macro variable files for potential future use."""

# ============================================================================
# Time Series Parameters
# ============================================================================

DEFAULT_LOOKBACK_DAYS: int = 756
"""Default lookback window in days (approximately 3 years of trading days)."""

DEFAULT_VALIDATION_DAYS: int = 20
"""Default validation period in days (approximately 1 month of trading days)."""

DEFAULT_P_MAX: int = 10
"""Maximum lag order for VAR (Vector Autoregression) model."""

DEFAULT_ALPHA_GRID: list[float] = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
"""Grid of regularization parameters (elastic net alpha values) for tuning."""

# ============================================================================
# Testing & Evaluation Parameters
# ============================================================================

TEST_DAYS: int = 876
"""Initial test period in days for preliminary model evaluation."""

TEST_FORECAST_DAYS: int = TEST_DAYS - DEFAULT_LOOKBACK_DAYS
"""Number of forecast days in test period (876 - 756 = 120 days)."""
