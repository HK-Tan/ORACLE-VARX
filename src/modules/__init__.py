"""Consolidated helper modules for ORACLE-VARX.

Contains:
- factory: GPU-accelerated learner factory (needs rework)
- grid_config: Grid configuration for model caching and memoization
- batch_utils: Shared batched operations for VAR and OR-VARX models
"""

from .factory import (
    get_regressor,
    list_available_regressors,
)

from .grid_config import GridConfig

from .batch_utils import batched_ols

__all__ = [
    # factory
    'get_regressor',
    'list_available_regressors',
    # grid_config
    'GridConfig',
    # batch_utils
    'batched_ols',
]
