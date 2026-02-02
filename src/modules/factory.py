"""Learner factory for DML first-stage estimation.

Supports tree-based learners with native parallelization:
- xgboost: XGBoost with histogram-based tree method
- lgbm: LightGBM
- rf: sklearn RandomForestRegressor
- extra_trees: sklearn ExtraTreesRegressor
- tabpfn: TabPFN transformer-based regressor (GPU required)

Tree-based learners use n_jobs for CPU parallelization.
TabPFN requires GPU and a HuggingFace token (HF_TOKEN environment variable).
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
    """Get a regressor for first-stage estimation.

    Args:
        name: Regressor name. Options: 'xgboost', 'lgbm', 'rf', 'extra_trees', 'tabpfn'.
        n_jobs: Number of parallel jobs (-1 for all cores minus 1, to avoid LightGBM issues).
                Ignored for TabPFN which uses GPU.
        **kwargs: Additional arguments passed to the regressor

    Returns:
        Sklearn-compatible regressor with fit() and predict() methods

    Raises:
        ValueError: If unknown regressor name
        RuntimeError: If TabPFN requested but CUDA is not available
        ImportError: If TabPFN not installed
    """
    n_jobs = resolve_n_jobs(n_jobs)

    if name == 'xgboost':
        return xgb.XGBRegressor(
            n_jobs=n_jobs,
            tree_method='hist',
            max_depth=5,
            **kwargs
        )

    elif name == 'lgbm':
        return lgb.LGBMRegressor(
            n_jobs=n_jobs,
            device='cpu',
            verbose=-1,
            force_col_wise=True,
            max_depth=5,
            **kwargs
        )

    elif name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_jobs=n_jobs, max_depth=5, **kwargs)

    elif name == 'extra_trees':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_jobs=n_jobs, max_depth=5, **kwargs)

    elif name == 'tabpfn':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TabPFN requires GPU but CUDA is not available. "
                "Please run on a machine with GPU support."
            )
        # Disable posthog telemetry (~200ms latency per fit() call)
        os.environ['TABPFN_NO_TELEMETRY'] = '1'
        os.environ['DO_NOT_TRACK'] = '1'
        import sys as _sys
        if 'posthog' not in _sys.modules:
            class _FakePosthog:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            _sys.modules['posthog'] = _FakePosthog()
        try:
            from tabpfn import TabPFNRegressor
        except ImportError:
            raise ImportError(
                "TabPFN not installed. Install with: pip install tabpfn\n"
                "Also ensure HF_TOKEN environment variable is set and you have "
                "accepted the TabPFN terms at https://huggingface.co/Prior-Labs/TabPFN"
            )
        # n_estimators=1 reduces CPU preprocessing overhead by ~2.5x
        # (default is 8, but preprocessing is CPU-bound and dominates runtime)
        return TabPFNRegressor(device='cuda', random_state=42, **kwargs)

    else:
        raise ValueError(
            f"Unknown regressor: {name}. "
            f"Available: xgboost, lgbm, rf, extra_trees, tabpfn"
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


