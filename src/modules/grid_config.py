"""Grid configuration for OR-VARX memoization.

The grid structure enables model caching - instead of retraining models
for every day, we pre-train on a fixed grid of 21-day intervals and
reuse models. This significantly reduces computation while maintaining
forecast accuracy.

Grid Layout for VAR:
    |<-- lookback_var -->|
    |   504 + 10 = 514   |
    |  ols_window + p_max_offset  |

Grid Layout for OR-VARX:
    |<--------------- lookback_orvarx ---------------->|
    |<-- tree_train -->|<-- ols_window -->|<-- offset -->|
    |     504 days     |     504 days     |   10 days    |
    |                       1018 days total              |

The lookback includes extra days (p_max_offset) to accommodate VAR lags,
ensuring sufficient history is available for lag computation.
"""

import warnings
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for grid-based model training and caching.

    Attributes:
        ols_window: Rows for final OLS regression (both VAR and OR-VARX), default: 504 ~ 2 years
        tree_train_window: First-stage tree training window (OR-VARX only), default: 504 ~ 2 years
        test_size: Test/forecast window size in days (default: 21 ~ 1 month)
        p_max_offset: Extra days for VAR lag computation (default: 10)

    Properties:
        lookback_var: VAR lookback = ols_window + p_max_offset (514 days)
        lookback_orvarx: OR-VARX lookback = tree_train_window + ols_window + p_max_offset (1018 days)
        train_size: Alias for tree_train_window (backward compatibility)
        lookback: DEPRECATED - use lookback_var or lookback_orvarx

    Example:
        >>> config = GridConfig()
        >>> config.lookback_var
        514
        >>> config.lookback_orvarx
        1018
    """

    # Core window sizes
    ols_window: int = 504          # Rows for final OLS (both methods)
    tree_train_window: int = 504   # First-stage tree training (OR-VARX only)

    # Grid parameters
    test_size: int = 21            # Test window per fold (~1 month)

    # Lag parameters
    p_max_offset: int = 10         # Extra days for VAR lag computation

    # Batch processing parameters
    batch_chunk_size: int = 1000   # Chunk size for batched OLS (GPU memory management)

    @property
    def lookback_var(self) -> int:
        """Lookback for VAR: ols_window + p_max_offset (514 days)."""
        return self.ols_window + self.p_max_offset

    @property
    def lookback_orvarx(self) -> int:
        """Lookback for OR-VARX: tree_train_window + ols_window + p_max_offset (1018 days)."""
        return self.tree_train_window + self.ols_window + self.p_max_offset

    @property
    def train_size(self) -> int:
        """Backward compatibility alias for tree_train_window."""
        return self.tree_train_window

    @property
    def lookback(self) -> int:
        """DEPRECATED: Use lookback_var or lookback_orvarx instead.

        Returns lookback_orvarx for backward compatibility with OR-VARX code.
        """
        warnings.warn(
            "GridConfig.lookback is deprecated. Use lookback_var (for VAR) or "
            "lookback_orvarx (for OR-VARX) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.lookback_orvarx

    # Legacy property for backward compatibility during transition
    @property
    def lookback_base(self) -> int:
        """DEPRECATED: Legacy property, do not use."""
        warnings.warn(
            "GridConfig.lookback_base is deprecated. Use ols_window or tree_train_window instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.tree_train_window + self.ols_window
