"""Data loading utilities for financial time series data.

This module provides functions to load and prepare asset returns and confounder
variables for time series analysis and VAR modeling.

Data Format Details:
    - OPCL file: Wide format with tickers as rows, dates (X-prefixed) as columns
    - VIX file: Long format with observation_date and VIXCLS columns
    - All other confounders: Same format as VIX (long format)
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch

from src.data.constants import (
    CONFOUNDER_FILES,
    CONFOUNDER_PRESETS,
    DATA_DIR,
    DEFAULT_LOOKBACK_DAYS,
    ETFS,
    ETFS_WITH_VIX,
    OPCL_FILE,
    VIX_FILE,
)


def load_opcl_data(
    tickers: Optional[List[str]] = None,
    n_days: Optional[int] = None,
) -> pd.DataFrame:
    """Load OPCL (Open-Close) returns data.

    The raw file has tickers as rows and dates as columns (wide format).
    This function transposes to return dates as rows, tickers as columns.

    Args:
        tickers: List of tickers to load. If None, uses ETFS (9 ETFs).
        n_days: Number of days to load from start. If None, loads all days.

    Returns:
        DataFrame with shape (n_days, n_tickers), DatetimeIndex, ticker columns.
        Index is sorted chronologically.

    Raises:
        FileNotFoundError: If OPCL_FILE does not exist.
        ValueError: If any requested ticker is not found in the data.

    Example:
        >>> df = load_opcl_data(tickers=["SPY", "XLF"], n_days=100)
        >>> df.shape
        (100, 2)
        >>> df.index  # DatetimeIndex
        >>> df.columns  # ["SPY", "XLF"]
    """
    if tickers is None:
        tickers = ETFS

    if not OPCL_FILE.exists():
        raise FileNotFoundError(f"OPCL data file not found: {OPCL_FILE}")

    # Read CSV with ticker as index
    df = pd.read_csv(OPCL_FILE, index_col="ticker")

    # Verify all requested tickers exist
    missing_tickers = set(tickers) - set(df.index)
    if missing_tickers:
        raise ValueError(
            f"Tickers not found in OPCL data: {sorted(missing_tickers)}\n"
            f"Available tickers: {sorted(df.index.tolist())[:20]}..."
        )

    # Filter to requested tickers
    df = df.loc[tickers]

    # Transpose so dates become rows (index) and tickers become columns
    df = df.T

    # Parse date column names (remove 'X' prefix, convert to datetime)
    # Column names like 'X20000103' -> '2000-01-03'
    df.index = pd.to_datetime(df.index.str[1:], format="%Y%m%d")
    df.index.name = "date"

    # Sort by date (should already be sorted, but ensure it)
    df = df.sort_index()

    # Limit to n_days if specified
    if n_days is not None:
        df = df.iloc[:n_days]

    # Forward-fill missing values (should be rare in OPCL data)
    if df.isna().any().any():
        warnings.warn(
            f"Found {df.isna().sum().sum()} missing values in OPCL data. "
            f"Applying forward fill."
        )
        df = df.ffill()
        # If first rows have NaN (can't forward fill), backfill
        if df.isna().any().any():
            df = df.bfill()

    return df


def load_opcl_with_vix(
    etf_tickers: Optional[List[str]] = None,
    n_days: Optional[int] = None,
) -> pd.DataFrame:
    """Load OPCL returns for ETFs plus VIX converted to log returns.

    VIX is treated as a tradeable asset by converting VIX levels to log returns:
    VIX_return_t = ln(VIX_t / VIX_{t-1})

    Returns are in decimal format (e.g., 0.05 = 5%) to match ETF returns.
    This allows VIX to be included in the VAR model alongside ETF returns.

    Args:
        etf_tickers: List of ETF tickers to load. If None, uses ETFS (9 ETFs).
        n_days: Number of days to load from start. If None, loads all days.

    Returns:
        DataFrame with shape (n_days, n_tickers + 1), DatetimeIndex,
        columns are ETF tickers + "VIX". Index is sorted chronologically.
        The first day is dropped because VIX returns require one prior day.

    Raises:
        FileNotFoundError: If OPCL_FILE or VIX_FILE does not exist.
        ValueError: If any requested ticker is not found in the data.

    Example:
        >>> df = load_opcl_with_vix(etf_tickers=["XLY", "XLF"], n_days=100)
        >>> df.shape
        (99, 3)  # 99 days (first dropped for VIX return), 2 ETFs + VIX
        >>> df.columns.tolist()
        ['XLY', 'XLF', 'VIX']
    """
    import numpy as np

    if etf_tickers is None:
        etf_tickers = ETFS

    # Load ETF returns (already in return format)
    # Load one extra day since we'll drop the first for VIX return calculation
    extra_days = n_days + 1 if n_days is not None else None
    etf_df = load_opcl_data(tickers=etf_tickers, n_days=extra_days)

    # Load VIX levels (raw, not log returns — we convert manually below)
    vix_series = load_confounder("VIX", n_days=None, log_returns=False)

    # Convert VIX levels to log returns: ln(VIX_t / VIX_{t-1})
    # Shift by 1 to get previous day's value
    # Note: Returns are in decimal format to match ETF returns (e.g., 0.05 = 5%)
    vix_returns = np.log(vix_series / vix_series.shift(1))
    vix_returns.name = "VIX"

    # Drop first value (NaN from shift)
    vix_returns = vix_returns.dropna()

    # Filter VIX to match ETF trading dates
    vix_returns = vix_returns[vix_returns.index.isin(etf_df.index)]

    # Reindex to ensure exact alignment with ETF dates
    vix_returns = vix_returns.reindex(etf_df.index)

    # Merge ETF returns and VIX returns
    merged_df = etf_df.copy()
    merged_df["VIX"] = vix_returns

    # Drop rows with NaN (first day has no VIX return)
    merged_df = merged_df.dropna()

    # Limit to n_days if specified (after dropping NaN)
    if n_days is not None:
        merged_df = merged_df.iloc[:n_days]

    return merged_df


def load_opcl_with_confounders(
    confounder_names: List[str],
    etf_tickers: Optional[List[str]] = None,
    n_days: Optional[int] = None,
) -> pd.DataFrame:
    """Load OPCL returns for ETFs plus confounders as endogenous log returns.

    Generalizes load_opcl_with_vix() for any set of confounders. Each confounder
    is converted to log returns via load_confounder(log_returns=True) and included
    as an endogenous variable in the DataFrame.

    Args:
        confounder_names: List of confounder names (keys in CONFOUNDER_FILES).
                         Can also be a preset name from CONFOUNDER_PRESETS
                         (e.g., "vix", "macro5", "all10").
        etf_tickers: List of ETF tickers to load. If None, uses ETFS (9 ETFs).
        n_days: Number of days to load from start. If None, loads all days.

    Returns:
        DataFrame with shape (n_days, n_tickers + n_confounders), DatetimeIndex,
        columns are ETF tickers + confounder names. Index is sorted chronologically.

    Raises:
        FileNotFoundError: If OPCL_FILE or any confounder file does not exist.
        ValueError: If any requested ticker/confounder is not found.
        KeyError: If preset name is not found in CONFOUNDER_PRESETS.

    Example:
        >>> df = load_opcl_with_confounders(["VIX", "DFF"], n_days=100)
        >>> df.columns.tolist()
        ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU', 'VIX', 'DFF']
    """
    from src.data.constants import CONFOUNDER_PRESETS

    # Resolve preset name to list of confounder names
    if isinstance(confounder_names, str):
        if confounder_names in CONFOUNDER_PRESETS:
            confounder_names = CONFOUNDER_PRESETS[confounder_names]
        else:
            raise KeyError(
                f"Unknown confounder preset '{confounder_names}'. "
                f"Available presets: {list(CONFOUNDER_PRESETS.keys())}. "
                f"Or pass a list of confounder names."
            )

    if etf_tickers is None:
        etf_tickers = ETFS

    # Load ETF returns (already in return format)
    # Load extra days since confounders lose one day from log return conversion
    extra_days = n_days + 1 if n_days is not None else None
    etf_df = load_opcl_data(tickers=etf_tickers, n_days=extra_days)

    # Load each confounder as log returns
    confounder_series = []
    for name in confounder_names:
        series = load_confounder(name, n_days=None, log_returns=True)
        confounder_series.append(series)

    # Filter confounders to ETF trading dates and merge
    merged_df = etf_df.copy()
    for series in confounder_series:
        # Filter to ETF trading dates
        filtered = series[series.index.isin(etf_df.index)]
        filtered = filtered.reindex(etf_df.index)
        merged_df[series.name] = filtered

    # Backfill leading NaN confounders with 0.0 (log-return of 0 = "no change").
    # Only fill confounder columns, not ETF columns.
    confounder_cols = [name for name in confounder_names if name in merged_df.columns]
    merged_df[confounder_cols] = merged_df[confounder_cols].fillna(0.0)

    # Drop rows with any NaN
    merged_df = merged_df.dropna()

    # Limit to n_days if specified (after dropping NaN)
    if n_days is not None:
        merged_df = merged_df.iloc[:n_days]

    return merged_df


def load_confounder(
    name: str,
    n_days: Optional[int] = None,
    log_returns: bool = True,
) -> pd.Series:
    """Load a single confounder variable (e.g., VIX).

    Confounder files are in long format with columns:
        - observation_date: date string (YYYY-MM-DD)
        - [variable name]: variable values (may have missing values as empty strings)

    Args:
        name: Confounder name (key in CONFOUNDER_FILES dict).
              Examples: "VIX", "DFF", "T5YIE", "DCOILWTICO", "USEPUINDXD"
        n_days: Number of days to load from start. If None, loads all days.
        log_returns: If True (default), convert levels to log returns via
                     ln(C_t / C_{t-1}) and backfill the first NaN. This ensures
                     stationarity and consistency with ETF return data.

    Returns:
        Series with DatetimeIndex and confounder values.
        Missing values are forward-filled.

    Raises:
        KeyError: If confounder name not found in CONFOUNDER_FILES.
        FileNotFoundError: If confounder file does not exist.

    Example:
        >>> vix = load_confounder("VIX", n_days=100)
        >>> vix.shape
        (100,)
        >>> vix.name
        'VIX'
    """
    if name not in CONFOUNDER_FILES:
        raise KeyError(
            f"Confounder '{name}' not found. "
            f"Available: {list(CONFOUNDER_FILES.keys())}"
        )

    file_path = CONFOUNDER_FILES[name]
    if not file_path.exists():
        raise FileNotFoundError(f"Confounder file not found: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path)

    # Determine variable column name (second column, not observation_date)
    date_col = df.columns[0]
    value_col = df.columns[1]

    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Set date as index
    df = df.set_index(date_col)
    df.index.name = "date"

    # Extract the value column as Series
    series = df[value_col]
    series.name = name

    # Replace empty strings with NaN
    series = series.replace("", pd.NA)

    # Convert to numeric (handles string numbers)
    series = pd.to_numeric(series, errors="coerce")

    # Sort by date
    series = series.sort_index()

    # Limit to n_days if specified
    if n_days is not None:
        series = series.iloc[:n_days]

    # Forward-fill missing values
    if series.isna().any():
        n_missing = series.isna().sum()
        logging.debug(
            f"Found {n_missing} missing values in {name} data. "
            f"Applying forward fill."
        )
        series = series.ffill()
        # If first values have NaN (can't forward fill), backfill
        if series.isna().any():
            series = series.bfill()

    # Convert to log returns if requested
    if log_returns:
        import numpy as np
        series = np.log(series / series.shift(1))
        # Backfill the first NaN from the shift (single value, no signal)
        series = series.bfill()

    return series


def align_data(
    assets: pd.DataFrame,
    confounders: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align assets and confounders to common date index.

    Handles cases where confounders have different date ranges.
    Takes the intersection of dates and drops any remaining missing values.

    Args:
        assets: DataFrame with DatetimeIndex (dates) and ticker columns.
        confounders: DataFrame with DatetimeIndex (dates) and confounder columns.

    Returns:
        Tuple of (aligned_assets, aligned_confounders) with identical date indices.
        Both DataFrames are sorted by date and have no missing values.

    Raises:
        ValueError: If there are no overlapping dates between assets and confounders.

    Example:
        >>> assets = load_opcl_data(n_days=1000)
        >>> vix = load_confounder("VIX", n_days=1000)
        >>> confounders = pd.DataFrame({"VIX": vix})
        >>> assets_aligned, conf_aligned = align_data(assets, confounders)
        >>> assets_aligned.index.equals(conf_aligned.index)
        True
    """
    # Find common dates (intersection)
    common_dates = assets.index.intersection(confounders.index)

    if len(common_dates) == 0:
        raise ValueError(
            "No overlapping dates between assets and confounders.\n"
            f"Assets date range: {assets.index.min()} to {assets.index.max()}\n"
            f"Confounders date range: {confounders.index.min()} to {confounders.index.max()}"
        )

    # Align to common dates
    assets_aligned = assets.loc[common_dates].sort_index()
    confounders_aligned = confounders.loc[common_dates].sort_index()

    # Drop any remaining rows with missing values
    mask = ~(assets_aligned.isna().any(axis=1) | confounders_aligned.isna().any(axis=1))

    n_dropped = len(assets_aligned) - mask.sum()
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} dates due to missing values after alignment."
        )

    assets_aligned = assets_aligned[mask]
    confounders_aligned = confounders_aligned[mask]

    return assets_aligned, confounders_aligned


def prepare_tensors(
    tickers: Optional[List[str]] = None,
    confounder_names: Optional[List[str]] = None,
    n_days: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str], List[str]]:
    """Load data and convert to PyTorch tensors for model training.

    This is the main function for preparing data for VAR model training.
    It handles loading, alignment, and conversion to tensors.

    Args:
        tickers: Asset tickers to load. If None, uses ETFS (9 ETFs).
        confounder_names: Confounders to load (e.g., ["VIX", "DFF"]).
                         If None or empty list, no confounders are loaded.
        n_days: Number of days to load. If None, loads all available days.
               Note: After alignment and missing value removal, actual length may be less.
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        Tuple of:
        - Y: Asset returns tensor, shape (T, n_assets), dtype float32
        - W: Confounder tensor, shape (T, n_confounders), dtype float32, or None if no confounders
        - dates: List of T date strings (YYYY-MM-DD format)
        - tickers: List of n_assets ticker names used

    Raises:
        ValueError: If no data available after loading and alignment.

    Example:
        >>> Y, W, dates, tickers = prepare_tensors(
        ...     tickers=["SPY", "XLF", "XLE"],
        ...     confounder_names=["VIX"],
        ...     n_days=500,
        ...     device="cpu"
        ... )
        >>> Y.shape
        torch.Size([500, 3])  # May be less after alignment
        >>> W.shape
        torch.Size([500, 1])
        >>> len(dates), len(tickers)
        (500, 3)
    """
    if tickers is None:
        tickers = ETFS

    # Load asset returns
    assets = load_opcl_data(tickers=tickers, n_days=n_days)

    # Load confounders if requested
    if confounder_names:
        confounder_series = {}
        for name in confounder_names:
            # Load confounders without n_days limit to ensure full coverage
            # This matches old-code behavior: load all confounders, then filter to trading dates
            confounder_series[name] = load_confounder(name, n_days=None)

        confounders = pd.DataFrame(confounder_series)

        # Filter confounders to match asset dates (trading dates only)
        # This replicates old-code line 288: filtered_confound_df = imputed_confound_df[imputed_confound_df.index.isin(returns_df_cleaned.index)]
        confounders = confounders[confounders.index.isin(assets.index)]

        # Reindex to ensure exact alignment and same order
        confounders = confounders.reindex(assets.index)

        # Backfill leading NaN confounders with 0.0 (log-return of 0 = "no change").
        # Preserves asset data from before late-starting confounders (e.g., GVZCLS 2008).
        confounders = confounders.fillna(0.0)

        # Drop any rows with missing values (should be none after forward fill in load_confounder)
        mask = ~(assets.isna().any(axis=1) | confounders.isna().any(axis=1))
        n_dropped = len(assets) - mask.sum()
        if n_dropped > 0:
            warnings.warn(
                f"Dropped {n_dropped} dates due to missing values after alignment."
            )
            assets = assets[mask]
            confounders = confounders[mask]
    else:
        confounders = None

    # Verify we have data
    if len(assets) == 0:
        raise ValueError("No data available after loading and alignment.")

    # Extract dates as strings
    dates = assets.index.strftime("%Y-%m-%d").tolist()

    # Convert to numpy arrays (float32 for GPU efficiency)
    Y = assets.values.astype("float32")

    if confounders is not None:
        W = confounders.values.astype("float32")
    else:
        W = None

    # Convert to PyTorch tensors
    Y_tensor = torch.from_numpy(Y).to(device)
    W_tensor = torch.from_numpy(W).to(device) if W is not None else None

    return Y_tensor, W_tensor, dates, tickers


def load_test_data(
    n_days: int = 624,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Convenience function to load test dataset (9 ETFs + VIX, 624 days).

    This function loads the standard test configuration used throughout the project:
    - 9 sector ETFs (defined in ETFS constant)
    - VIX as the confounder variable
    - 624 trading days (approximately 2.5 years)

    Args:
        n_days: Number of days to load. Default: 624 (standard test period).
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        Tuple of:
        - Y: Asset returns tensor, shape (T, 9), dtype float32
        - W: VIX tensor, shape (T, 1), dtype float32
        - dates: List of T date strings (YYYY-MM-DD format)
        - tickers: List of 9 ticker names (ETFS constant)

    Example:
        >>> Y, W, dates, tickers = load_test_data()
        >>> Y.shape
        torch.Size([624, 9])  # May be slightly less after alignment
        >>> W.shape
        torch.Size([624, 1])
        >>> tickers
        ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']
    """
    Y, W, dates, tickers = prepare_tensors(
        tickers=ETFS,
        confounder_names=["VIX"],
        n_days=n_days,
        device=device,
    )

    # Ensure W is not None (should always have VIX)
    if W is None:
        raise RuntimeError("Expected VIX data but W is None. Check data files.")

    return Y, W, dates, tickers
