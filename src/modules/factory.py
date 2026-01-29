"""CPU-only learner factory for DML first-stage estimation.

Supports tree-based learners with native parallelization:
- xgboost: XGBoost with histogram-based tree method
- lgbm: LightGBM
- rf: sklearn RandomForestRegressor
- extra_trees: sklearn ExtraTreesRegressor

All learners use n_jobs for CPU parallelization.
"""
import os
import xgboost as xgb
import lightgbm as lgb
from typing import Any


def _get_physical_cpu_count() -> int:
    """Get the number of physical CPU cores (not logical/hyperthreaded)."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or 1
    except ImportError:
        logical = os.cpu_count() or 2
        return max(1, logical // 2)


def resolve_n_jobs(n_jobs: int) -> int:
    """Resolve n_jobs to actual core count.

    Args:
        n_jobs: Number of jobs. -1 means all cores minus 1 (to avoid LightGBM issues).

    Returns:
        Resolved number of jobs (always positive).
    """
    if n_jobs == -1:
        return max(1, _get_physical_cpu_count() - 1)
    return n_jobs


def get_regressor(name: str = 'xgboost', n_jobs: int = -1, **kwargs) -> Any:
    """CPU-only learners with native parallelization.

    Args:
        name: Regressor name. Options: 'xgboost', 'lgbm', 'rf', 'extra_trees'.
        n_jobs: Number of parallel jobs (-1 for all cores minus 1, to avoid LightGBM issues)
        **kwargs: Additional arguments passed to the regressor

    Returns:
        Sklearn-compatible regressor with fit() and predict() methods

    Raises:
        ValueError: If unknown regressor name
    """
    n_jobs = resolve_n_jobs(n_jobs)

    if name == 'xgboost':
        return xgb.XGBRegressor(
            n_jobs=n_jobs,
            tree_method='hist',
            **kwargs
        )

    elif name == 'lgbm':
        return lgb.LGBMRegressor(
            n_jobs=n_jobs,
            device='cpu',
            verbose=-1,
            force_col_wise=True,
            **kwargs
        )

    elif name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_jobs=n_jobs, max_depth=5, **kwargs)

    elif name == 'extra_trees':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_jobs=n_jobs, max_depth=5, **kwargs)

    else:
        raise ValueError(
            f"Unknown regressor: {name}. "
            f"Available: xgboost, lgbm, rf, extra_trees"
        )


def get_multi_output_regressor(name: str, n_jobs: int = -1, **kwargs) -> Any:
    """Wrap base regressor with MultiOutputRegressor for Y/T prediction.

    Parallelization strategy:
    - Base regressor uses n_jobs for internal tree parallelization
    - MultiOutputRegressor outer loop is sequential (n_jobs=1)
      since base already saturates CPU cores

    Args:
        name: Base regressor name
        n_jobs: Number of parallel jobs (-1 for all cores minus 1)
        **kwargs: Additional arguments passed to the base regressor

    Returns:
        MultiOutputRegressor wrapping the specified base regressor
    """
    from sklearn.multioutput import MultiOutputRegressor
    base = get_regressor(name, n_jobs=n_jobs, **kwargs)
    return MultiOutputRegressor(base, n_jobs=1)


