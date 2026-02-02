"""ORACLE-VARX with TabPFN for nuisance estimation (batched GPU approach).

This module implements ORACLE-VARX using TabPFN transformer for first-stage
nuisance function estimation, with batched operations for maximum GPU utilization.

Similar to the VAR-GPU approach in dml_pytorch.py, this groups folds by lag order p
and batches multiple folds together for efficiency.

Key Insight:
    TabPFN's transformer accepts (seq_len, batch_size, features). We leverage this
    to batch multiple folds × outputs together, reducing forward passes from
    ~16,416 (sequential) to ~80 (batched).

Batching Strategy (Level 3):
    For each p value (1 to 10):
        For each estimator (8 times, different preprocessing):
            Collect all folds with this p: ~23 folds
            Batch all folds × 9 outputs in single forward pass
            → Output: (n_folds, n_test, 9) predictions

    Total forward passes: 10 p-values × 8 estimators = 80
    Speedup: ~20-30x compared to sequential approach

Architecture:
    fit_oraclevarx_tabpfn()
        |
        +---> _prepare_fold_data()  # Build lagged features for all folds
        |
        +---> _run_batched_tabpfn() # TabPFN inference, batched by p
        |
        +---> _compute_residuals()  # Compute Y_res, T_res from predictions
        |
        +---> batched_ols()         # Second-stage OLS on residuals
        |
        +---> _apply_significance() # ORACLE significance-based p-selection
"""

import gc
import time
from typing import List, Optional, Tuple, Dict, Any

import torch
import numpy as np


def _clear_gpu_memory():
    """Force GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_vram_usage() -> Tuple[float, float, float]:
    """Get current GPU VRAM usage using nvidia-smi (actual usage).

    Queries actual GPU memory usage as reported by NVIDIA drivers,
    which includes all memory (not just PyTorch tensors).

    Returns:
        (used_gb, total_gb, percent_used)
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0)

    try:
        # Try pynvml first (fastest)
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used / (1024 ** 3)
        total = info.total / (1024 ** 3)
        percent = 100 * used / total
        pynvml.nvmlShutdown()
        return (used, total, percent)
    except Exception:
        pass

    try:
        # Fallback to nvidia-smi subprocess
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            used_mb, total_mb = map(float, result.stdout.strip().split(','))
            used = used_mb / 1024
            total = total_mb / 1024
            percent = 100 * used / total
            return (used, total, percent)
    except Exception:
        pass

    # Final fallback to PyTorch (only tracks PyTorch tensors)
    used = torch.cuda.memory_allocated() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    percent = 100 * used / total
    return (used, total, percent)


def _get_batch_size_for_p(p: int, n_folds: int, verbose: bool = False) -> int:
    """Get batch size for a given lag order p using size-based heuristic.

    Uses formula: batch_size = (4.5 * VRAM_GB) / p^1.5

    This accounts for:
    - Larger p means more features (n_assets * p treatment features)
    - Memory scales super-linearly with feature count due to attention
    - The p^1.5 exponent provides extra margin for larger p values
    - Scales linearly with available VRAM (calibrated: 360 for 80GB)

    Args:
        p: Lag order (1 to p_max)
        n_folds: Total number of folds available (for capping)
        verbose: Print batch size calculation

    Returns:
        Batch size (minimum 1, capped at n_folds).
    """
    # Get available VRAM
    _, total_vram_gb, _ = _get_vram_usage()

    # Scale factor: 48 * (VRAM / 8) = 6 * VRAM
    # Calibrated for 80GB → 480 numerator
    numerator = 6 * total_vram_gb

    batch_size = int(numerator / (p ** 1.5))
    batch_size = min(batch_size, n_folds)
    batch_size = max(1, batch_size)

    if verbose:
        print(f"    p={p}: batch_size = {numerator:.0f}/{p}^1.5 = {int(numerator / (p ** 1.5))} → capped to {batch_size}")

    return batch_size

from src.results import ORACLEVARXResult
from src.modules.grid_config import GridConfig
from src.modules.batch_utils import batched_ols, batched_benjamini_hochberg
from src.modules.batched_tabpfn import BatchedFoldTabPFN
from src.models.var_pytorch import rolling_alpha_selection


def _build_lagged_features(
    Y: np.ndarray,
    W: np.ndarray,
    p: int,
    start_idx: int,
    end_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build lagged features for a contiguous range of rows.

    Given raw Y and W arrays, construct outcome (Y_t), treatment (lagged Y),
    and controls (lagged W) for rows in range [start_idx + p, end_idx).

    Args:
        Y: Full Y array, shape (n_days, n_assets)
        W: Full W array, shape (n_days, n_confounders)
        p: Lag order
        start_idx: Start of range (inclusive, before accounting for lags)
        end_idx: End of range (exclusive)

    Returns:
        outcome: Y values, shape (n_rows, n_assets) where n_rows = end_idx - start_idx - p
        treatment: Lagged Y values, shape (n_rows, n_assets * p)
        controls: Lagged W values, shape (n_rows, n_confounders * p)
    """
    n_assets = Y.shape[1]
    n_confounders = W.shape[1]
    n_rows = end_idx - start_idx - p

    # Outcome: Y[start_idx + p : end_idx]
    outcome = Y[start_idx + p:end_idx]

    # Treatment: lagged Y values
    treatment = np.zeros((n_rows, n_assets * p), dtype=np.float32)
    for lag in range(1, p + 1):
        lag_start = start_idx + p - lag
        lag_end = end_idx - lag
        treatment[:, (lag - 1) * n_assets:lag * n_assets] = Y[lag_start:lag_end]

    # Controls: lagged W values
    controls = np.zeros((n_rows, n_confounders * p), dtype=np.float32)
    for lag in range(1, p + 1):
        lag_start = start_idx + p - lag
        lag_end = end_idx - lag
        controls[:, (lag - 1) * n_confounders:lag * n_confounders] = W[lag_start:lag_end]

    return outcome, treatment, controls


def _build_test_features(
    Y: np.ndarray,
    W: np.ndarray,
    p: int,
    test_start: int,
    test_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build test features allowing lags from BEFORE test_start.

    Unlike _build_lagged_features which requires p "burn-in" rows,
    this function predicts ALL rows in [test_start, test_end) by
    sourcing lags from [test_start - p, test_end - 1].

    Args:
        Y: Full Y array, shape (n_days, n_assets)
        W: Full W array, shape (n_days, n_confounders)
        p: Lag order
        test_start: First row to predict (inclusive)
        test_end: Last row to predict (exclusive)

    Returns:
        outcome: Y values, shape (n_rows, n_assets) where n_rows = test_end - test_start
        treatment: Lagged Y values, shape (n_rows, n_assets * p)
        controls: Lagged W values, shape (n_rows, n_confounders * p)
    """
    n_assets = Y.shape[1]
    n_confounders = W.shape[1]
    n_rows = test_end - test_start

    # Outcome: Y[test_start : test_end]
    outcome = Y[test_start:test_end]

    # Treatment: lagged Y values (can go before test_start)
    treatment = np.zeros((n_rows, n_assets * p), dtype=np.float32)
    for lag in range(1, p + 1):
        lag_start = test_start - lag
        lag_end = test_end - lag
        treatment[:, (lag - 1) * n_assets:lag * n_assets] = Y[lag_start:lag_end]

    # Controls: lagged W values (can go before test_start)
    controls = np.zeros((n_rows, n_confounders * p), dtype=np.float32)
    for lag in range(1, p + 1):
        lag_start = test_start - lag
        lag_end = test_end - lag
        controls[:, (lag - 1) * n_confounders:lag * n_confounders] = W[lag_start:lag_end]

    return outcome, treatment, controls


def _build_forecast_controls(W: np.ndarray, day_idx: int, p: int) -> np.ndarray:
    """Build lagged control features for a single forecast day.

    For forecasting day_idx, we need W values at [day_idx-1, ..., day_idx-p].

    Args:
        W: Full W array, shape (n_days, n_confounders)
        day_idx: The day index to forecast (absolute index)
        p: Lag order

    Returns:
        controls: Lagged W values, shape (n_confounders * p,)
    """
    n_confounders = W.shape[1]
    controls = np.zeros(n_confounders * p, dtype=np.float32)
    for lag in range(1, p + 1):
        controls[(lag - 1) * n_confounders:lag * n_confounders] = W[day_idx - lag]
    return controls


def _compute_fold_boundaries(
    grid_idx: int,
    config: GridConfig,
) -> Tuple[int, int, int, int]:
    """Compute train/test boundaries for a grid fold.

    Args:
        grid_idx: Index of the grid fold (0-indexed)
        config: GridConfig with train_size and test_size

    Returns:
        Tuple of (train_start, train_end, test_start, test_end)
    """
    train_start = grid_idx * config.test_size
    train_end = train_start + config.train_size
    test_start = train_end
    test_end = test_start + config.test_size
    return train_start, train_end, test_start, test_end


def _get_all_required_folds(
    n_days: int,
    p_max: int,
    config: GridConfig,
) -> List[int]:
    """Determine all grid folds needed for the entire test period.

    Args:
        n_days: Total number of days
        p_max: Maximum lag order
        config: GridConfig

    Returns:
        Sorted list of unique grid indices needed
    """
    lookback = config.lookback_orvarx
    n_test_days = n_days - lookback

    if n_test_days < 1:
        return []

    # Find the range of folds we need
    # First test day is at index `lookback`
    # Last test day is at index `n_days - 1`

    # Collect all folds needed across all test days and p values
    all_folds = set()

    for day_rel_idx in range(n_test_days):
        day_idx = lookback + day_rel_idx
        for p in range(1, p_max + 1):
            # Row range in absolute indices
            row_start_abs = day_idx - config.lookback_orvarx + p
            row_end_abs = day_idx - 1

            # Find folds covering this range
            first_grid_idx = max(0, (row_start_abs - config.train_size) // config.test_size)
            last_grid_idx = (row_end_abs - config.train_size) // config.test_size
            last_grid_idx = max(last_grid_idx, first_grid_idx)

            all_folds.update(range(first_grid_idx, last_grid_idx + 1))

    return sorted(all_folds)


def fit_oraclevarx_tabpfn(
    Y: torch.Tensor,
    W: torch.Tensor,
    alpha_grid: Optional[List[float]] = None,
    p_max: int = 10,
    config: Optional[GridConfig] = None,
    validation_days: int = 21,
    asset_names: Optional[List[str]] = None,
    confounder_names: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    n_estimators: int = 8,
    device: str = 'cuda',
    verbose: bool = False,
    target_vram_pct: float = 0.70,
) -> ORACLEVARXResult:
    """Fit ORACLE-VARX using batched TabPFN for first-stage estimation.

    This function uses TabPFN transformer for nuisance function estimation
    instead of tree-based learners. TabPFN provides high-quality predictions
    for small-data regimes typical in financial applications.

    The batched approach groups folds by lag order p and processes them
    together for GPU efficiency.

    Args:
        Y: Asset returns, shape (n_days, n_assets)
        W: Confounders, shape (n_days, n_confounders)
        alpha_grid: Significance levels for ORACLE (default: [0.01, ..., 0.30])
        p_max: Maximum lag order to consider (default: 10)
        config: GridConfig instance (default: None, uses default)
        validation_days: Days for alpha-selection via rolling validation
        asset_names: Names of assets
        confounder_names: Names of confounders
        dates: Date strings for forecast days
        n_estimators: Number of TabPFN ensemble members (default: 8)
        device: Device for TabPFN ('cuda' or 'cpu')
        verbose: Print detailed progress
        target_vram_pct: Target VRAM usage (default 0.70 = 70%). Batch size is
            automatically determined per p using a 2-point VRAM probe.

    Returns:
        ORACLEVARXResult with forecasts and coefficients

    Note:
        TabPFN has limits: max 100 features, max 10,000 training samples.
        For ORACLE-VARX with p_max=10 and 9 assets, we have 90 treatment features
        plus confounders, which fits within TabPFN's limits.
    """
    # Default alpha grid
    if alpha_grid is None:
        alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # Use default config if not provided
    if config is None:
        config = GridConfig()

    lookback = config.lookback_orvarx
    n_days, n_assets = Y.shape
    n_days_w, n_confounders = W.shape
    n_alphas = len(alpha_grid)

    # Validation
    if n_days != n_days_w:
        raise ValueError(f"Y and W must have same number of days: {n_days} vs {n_days_w}")
    if p_max < 1:
        raise ValueError(f"p_max must be >= 1, got {p_max}")
    if lookback < p_max:
        raise ValueError(f"lookback {lookback} must be >= p_max {p_max}")
    if n_days <= lookback:
        raise ValueError(f"Need at least {lookback + 1} days, got {n_days}")
    if validation_days < 1:
        raise ValueError(f"validation_days must be >= 1, got {validation_days}")

    n_total_test_days = n_days - lookback
    n_output_days = n_total_test_days - validation_days

    if n_output_days < 1:
        raise ValueError(
            f"Insufficient data: need > {lookback + validation_days} days, got {n_days}."
        )

    # Set default names
    if asset_names is None:
        asset_names = [f"A{i+1}" for i in range(n_assets)]
    if confounder_names is None:
        confounder_names = [f"W{i+1}" for i in range(n_confounders)]
    if dates is None:
        dates = [f"D{lookback + validation_days + i}" for i in range(n_output_days)]

    dev = Y.device
    dtype = Y.dtype

    print(f"Fitting ORACLE-VARX (TabPFN batched) for {n_total_test_days} total test days")
    print(f"  p_max={p_max}, n_estimators={n_estimators}, target_vram={target_vram_pct*100:.0f}%")
    print(f"  Grid config: train_size={config.train_size}, test_size={config.test_size}")

    # Convert to numpy for TabPFN
    Y_np = Y.cpu().numpy().astype(np.float32)
    W_np = W.cpu().numpy().astype(np.float32)

    # =========================================================================
    # Phase 1: Determine all required folds
    # =========================================================================
    print("  Phase 1: Determining fold structure...")
    t0 = time.time()

    all_folds = _get_all_required_folds(n_days, p_max, config)
    n_folds = len(all_folds)
    print(f"    Total folds needed: {n_folds}")

    # =========================================================================
    # Phase 2: Pre-compute data for all folds
    # =========================================================================
    print("  Phase 2: Building lagged features for all folds...")

    # For each p and each fold, store (X_train, Y_train, X_test, test_row_indices)
    fold_data: Dict[int, List[Dict[str, Any]]] = {p: [] for p in range(1, p_max + 1)}

    for grid_idx in all_folds:
        train_start, train_end, test_start, test_end = _compute_fold_boundaries(grid_idx, config)

        # Clamp to data size
        if train_end > n_days:
            continue
        test_end = min(test_end, n_days)

        for p in range(1, p_max + 1):
            # Build training data
            outcome_train, treatment_train, controls_train = _build_lagged_features(
                Y_np, W_np, p, train_start, train_end
            )

            # Check if we have enough history for lags (need data at test_start - p)
            if test_start - p < 0:
                continue

            # Use new function that allows lags from before test_start
            outcome_test, treatment_test, controls_test = _build_test_features(
                Y_np, W_np, p, test_start, test_end
            )

            # Collect forecast control features for test days in this fold's window
            # For DML forecasting, we need E[Y|W] and E[T|W] for each forecast day
            forecast_controls = []
            forecast_day_indices = []
            for test_day_idx in range(test_start, test_end):
                # Each test day in the fold window may be a forecast day
                # Forecast controls: lagged W values for this day
                fc = _build_forecast_controls(W_np, test_day_idx, p)
                forecast_controls.append(fc)
                forecast_day_indices.append(test_day_idx)

            fold_data[p].append({
                'grid_idx': grid_idx,
                'X_train': controls_train,  # Controls as features
                'Y_train': outcome_train,   # Y as target
                'T_train': treatment_train, # T as additional target
                'X_test': controls_test,
                'Y_test': outcome_test,
                'T_test': treatment_test,
                'test_start': test_start,  # Now the actual test_start, not test_start + p
                'test_end': test_end,
                'forecast_controls': forecast_controls,
                'forecast_day_indices': forecast_day_indices,
            })

    print(f"    Built data for {sum(len(v) for v in fold_data.values())} fold-p combinations")

    # =========================================================================
    # Phase 3: Run TabPFN for all folds, batched by p
    # =========================================================================
    print("  Phase 3: Running batched TabPFN predictions...")

    # Create batched TabPFN predictor
    tabpfn = BatchedFoldTabPFN(
        n_estimators=n_estimators,
        device=device,
        random_state=42,
    )

    # Store residuals indexed by absolute row
    # We'll compute residuals for all rows covered by test windows
    first_residual_row = {}
    R_Y_all = {}  # p -> residuals array
    R_T_all = {}  # p -> residuals array

    # Store forecast predictions (exact E[Y|W] and E[T|W] for each day)
    forecast_Y_preds: Dict[int, Dict[int, np.ndarray]] = {p: {} for p in range(1, p_max + 1)}
    forecast_T_preds: Dict[int, Dict[int, np.ndarray]] = {p: {} for p in range(1, p_max + 1)}

    # Background VRAM monitoring thread - prints every 60 seconds during inference
    import threading
    vram_monitor_stop = threading.Event()
    phase3_start = time.time()

    def _vram_monitor_thread():
        """Background thread that prints VRAM every 60 seconds."""
        interval = 60.0
        while not vram_monitor_stop.wait(timeout=interval):
            used, total, pct = _get_vram_usage()
            elapsed = time.time() - phase3_start
            print(f"      [{elapsed:.0f}s] VRAM: {used:.1f}/{total:.1f} GB ({pct:.0f}%)", flush=True)

    vram_thread = None
    if verbose:
        vram_thread = threading.Thread(target=_vram_monitor_thread, daemon=True)
        vram_thread.start()

    for p in range(1, p_max + 1):
        folds_p = fold_data[p]
        if not folds_p:
            continue

        n_folds_p = len(folds_p)
        n_treatments = n_assets * p

        p_start_time = time.time()

        # Get batch size using size-based heuristic (360/p^1.5)
        effective_batch_size = _get_batch_size_for_p(p, n_folds_p, verbose=verbose)

        # Group folds by test size (batching requires same size)
        folds_by_test_size: Dict[int, List[int]] = {}
        for fold_idx, fold in enumerate(folds_p):
            test_size = fold['X_test'].shape[0]
            if test_size not in folds_by_test_size:
                folds_by_test_size[test_size] = []
            folds_by_test_size[test_size].append(fold_idx)

        # Process each group of folds with the same test size
        Y_preds_all = [None] * n_folds_p
        T_preds_all = [None] * n_folds_p

        for test_size, fold_indices in folds_by_test_size.items():
            X_trains_group = [folds_p[i]['X_train'] for i in fold_indices]
            Y_trains_Y_group = [folds_p[i]['Y_train'] for i in fold_indices]
            Y_trains_T_group = [folds_p[i]['T_train'] for i in fold_indices]
            X_tests_group = [folds_p[i]['X_test'] for i in fold_indices]

            # Run batched TabPFN for this group
            Y_preds_group = tabpfn.fit_predict_batch(X_trains_group, Y_trains_Y_group, X_tests_group, effective_batch_size)

            _clear_gpu_memory()
            T_preds_group = tabpfn.fit_predict_batch(X_trains_group, Y_trains_T_group, X_tests_group, effective_batch_size)
            _clear_gpu_memory()

            # Store results back to original fold indices
            for group_idx, fold_idx in enumerate(fold_indices):
                Y_preds_all[fold_idx] = Y_preds_group[group_idx]
                T_preds_all[fold_idx] = T_preds_group[group_idx]

        # Compute residuals and store by absolute row index
        first_row = min(f['test_start'] for f in folds_p)
        last_row = max(f['test_end'] for f in folds_p)
        n_rows = last_row - first_row

        R_Y = np.zeros((n_rows, n_assets), dtype=np.float32)
        R_T = np.zeros((n_rows, n_treatments), dtype=np.float32)

        for fold_idx, fold in enumerate(folds_p):
            local_start = fold['test_start'] - first_row
            local_end = fold['test_end'] - first_row

            # Residuals = actual - predicted
            R_Y[local_start:local_end] = fold['Y_test'] - Y_preds_all[fold_idx]
            R_T[local_start:local_end] = fold['T_test'] - T_preds_all[fold_idx]

        R_Y_all[p] = R_Y
        R_T_all[p] = R_T
        first_residual_row[p] = first_row

        # Run separate batches for forecast predictions, grouped by forecast size
        folds_by_forecast_size: Dict[int, List[int]] = {}
        for fold_idx, fold in enumerate(folds_p):
            forecast_size = len(fold['forecast_controls'])
            if forecast_size not in folds_by_forecast_size:
                folds_by_forecast_size[forecast_size] = []
            folds_by_forecast_size[forecast_size].append(fold_idx)

        Y_forecast_all = [None] * n_folds_p
        T_forecast_all = [None] * n_folds_p

        for forecast_size, fold_indices in folds_by_forecast_size.items():
            if forecast_size == 0:
                continue

            X_trains_group = [folds_p[i]['X_train'] for i in fold_indices]
            Y_trains_Y_group = [folds_p[i]['Y_train'] for i in fold_indices]
            Y_trains_T_group = [folds_p[i]['T_train'] for i in fold_indices]
            X_forecast_group = [np.array(folds_p[i]['forecast_controls'], dtype=np.float32) for i in fold_indices]

            # Run batched TabPFN for forecast predictions
            Y_forecast_group = tabpfn.fit_predict_batch(X_trains_group, Y_trains_Y_group, X_forecast_group, effective_batch_size)
            _clear_gpu_memory()
            T_forecast_group = tabpfn.fit_predict_batch(X_trains_group, Y_trains_T_group, X_forecast_group, effective_batch_size)
            _clear_gpu_memory()

            for group_idx, fold_idx in enumerate(fold_indices):
                Y_forecast_all[fold_idx] = Y_forecast_group[group_idx]
                T_forecast_all[fold_idx] = T_forecast_group[group_idx]

        # Map forecast predictions to absolute day indices
        for fold_idx, fold in enumerate(folds_p):
            if Y_forecast_all[fold_idx] is None:
                continue
            for local_idx, day_idx in enumerate(fold['forecast_day_indices']):
                forecast_Y_preds[p][day_idx] = Y_forecast_all[fold_idx][local_idx]
                forecast_T_preds[p][day_idx] = T_forecast_all[fold_idx][local_idx]

        if verbose:
            p_elapsed = time.time() - p_start_time
            used, total, pct = _get_vram_usage()
            print(f"    p={p}: completed in {p_elapsed:.1f}s, "
                  f"VRAM: {used:.1f}/{total:.1f} GB ({pct:.0f}%), "
                  f"residuals: {R_Y.shape}, forecast days: {len(forecast_Y_preds[p])}")

    # Clear TabPFN cache to free GPU memory
    tabpfn.clear_cache()

    # Stop VRAM monitor thread
    if vram_thread is not None:
        vram_monitor_stop.set()
        vram_thread.join(timeout=1.0)

    print(f"  Phase 3: Complete ({time.time() - t0:.1f}s)")

    # =========================================================================
    # GPU Memory Cleanup before Phase 4
    # =========================================================================
    # Clear GPU memory after TabPFN to avoid fragmentation issues with MAGMA
    # (MAGMA's batched operations need contiguous memory blocks)
    del tabpfn
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if verbose:
        used, total, _ = _get_vram_usage()
        print(f"  GPU memory cleared: {used:.1f}/{total:.1f} GB")

    # =========================================================================
    # Phase 4: Batched OLS for all days and all lags
    # =========================================================================
    print("  Phase 4: Running batched OLS for coefficient estimation...")

    ols_window = config.ols_window

    # Initialize storage
    forecasts_all = torch.zeros(n_total_test_days, n_assets, p_max, device=dev, dtype=dtype)
    theta_all = torch.zeros(n_total_test_days, p_max, n_assets, n_assets, device=dev, dtype=dtype)
    SE_all = torch.zeros(n_total_test_days, p_max, n_assets, n_assets, device=dev, dtype=dtype)

    for p in range(1, p_max + 1):
        if p not in R_Y_all:
            continue

        R_Y = torch.from_numpy(R_Y_all[p]).to(device=dev, dtype=dtype)
        R_T = torch.from_numpy(R_T_all[p]).to(device=dev, dtype=dtype)
        first_row = first_residual_row[p]
        n_residual_rows = R_Y.shape[0]
        n_treatments = n_assets * p

        if n_residual_rows < ols_window:
            continue

        # Create sliding windows using unfold
        T_windows = R_T.unfold(0, ols_window, 1).transpose(1, 2)
        Y_windows = R_Y.unfold(0, ols_window, 1).transpose(1, 2)
        n_windows = T_windows.shape[0]

        try:
            # Batched OLS: theta = (T'T)^{-1} T'Y
            theta_batch = batched_ols(T_windows, Y_windows, chunk_size=config.batch_chunk_size)

            # Compute standard errors
            TtT = torch.bmm(T_windows.transpose(1, 2), T_windows)
            TtT_inv = torch.linalg.inv(TtT)
            TtT_inv_diag = torch.diagonal(TtT_inv, dim1=-2, dim2=-1)

            Y_pred = torch.bmm(T_windows, theta_batch)
            residuals = Y_windows - Y_pred
            RSS = (residuals ** 2).sum(dim=1)
            df = ols_window - n_treatments
            sigma_sq = RSS / df

            se_batch = torch.sqrt(TtT_inv_diag.unsqueeze(-1) * sigma_sq.unsqueeze(1))

            # Map windows to test days
            offset = first_row + ols_window - lookback

            for i in range(n_windows):
                day_rel_idx = i + offset
                if 0 <= day_rel_idx < n_total_test_days:
                    theta_reshaped = theta_batch[i].view(p, n_assets, n_assets)
                    theta_all[day_rel_idx, :p, :, :] = theta_reshaped

                    se_reshaped = se_batch[i].view(p, n_assets, n_assets)
                    SE_all[day_rel_idx, :p, :, :] = se_reshaped

        except RuntimeError as e:
            if verbose:
                print(f"    p={p}: OLS failed ({e})")
            continue

    # =========================================================================
    # Phase 5: Generate forecasts using exact E[Y|W] and E[T|W]
    # =========================================================================
    print("  Phase 5: Generating forecasts (exact DML)...")

    # DML forecast formula:
    #   forecast = E[Y|W] + (T_actual - E[T|W]) × θ
    #            = E[Y|W] + T_residual × θ
    #
    # Where:
    #   E[Y|W] = Y-model prediction given forecast controls (from Phase 3)
    #   E[T|W] = T-model prediction given forecast controls (from Phase 3)
    #   T_actual = actual treatment values [Y_{t-1}, ..., Y_{t-p}]
    #   T_residual = T_actual - E[T|W]
    #   θ = deconfounded coefficients from second-stage OLS

    for p in range(1, p_max + 1):
        if p not in R_Y_all:
            continue

        n_treatments = n_assets * p

        for day_rel_idx in range(n_total_test_days):
            day_idx = lookback + day_rel_idx

            # Check if we have valid theta for this day
            if theta_all[day_rel_idx, p-1].abs().sum() == 0:
                continue

            # Check if we have forecast predictions for this day
            if day_idx not in forecast_Y_preds[p] or day_idx not in forecast_T_preds[p]:
                continue

            theta = theta_all[day_rel_idx, :p, :, :].reshape(n_treatments, n_assets)

            # Get exact E[Y|W] and E[T|W] from Phase 3
            E_Y_given_W = torch.from_numpy(forecast_Y_preds[p][day_idx]).to(device=dev, dtype=dtype)
            E_T_given_W = torch.from_numpy(forecast_T_preds[p][day_idx]).to(device=dev, dtype=dtype)

            # Build T_actual: actual treatment values [Y_{day-1}, ..., Y_{day-p}]
            indices = list(range(day_idx - 1, day_idx - p - 1, -1))
            T_actual = torch.from_numpy(Y_np[indices, :].reshape(1, n_treatments)).to(device=dev, dtype=dtype)

            # Compute T_residual = T_actual - E[T|W]
            T_residual = T_actual - E_T_given_W.reshape(1, n_treatments)

            # Causal effect: T_residual × θ
            causal_effect = torch.mm(T_residual, theta).squeeze(0)

            # Forecast = E[Y|W] + causal_effect
            forecast = E_Y_given_W + causal_effect
            forecasts_all[day_rel_idx, :, p - 1] = forecast

    # =========================================================================
    # Phase 6: Significance testing and alpha selection
    # =========================================================================
    print("  Phase 6: Running significance tests and alpha selection...")

    # p_alpha_all: (n_total_test_days, n_alphas)
    p_alpha_all = torch.zeros(n_total_test_days, n_alphas, device=dev, dtype=torch.long)

    for alpha_idx, alpha in enumerate(alpha_grid):
        p_selected = torch.ones(n_total_test_days, device=dev, dtype=torch.long)

        for p in range(2, p_max + 1):
            theta_new = theta_all[:, p - 1, :, :]
            SE_new = SE_all[:, p - 1, :, :]

            z_new = torch.abs(theta_new / (SE_new + 1e-10))
            normal_dist = torch.distributions.Normal(0, 1)
            p_vals = 2 * (1 - normal_dist.cdf(z_new))

            p_vals_flat = p_vals.reshape(n_total_test_days, n_assets * n_assets)
            reject = batched_benjamini_hochberg(p_vals_flat, alpha)
            is_significant = reject.any(dim=1)

            still_active = (p_selected == p - 1)
            should_update = is_significant & still_active
            p_selected = torch.where(
                should_update,
                torch.tensor(p, device=dev, dtype=torch.long),
                p_selected
            )

        p_alpha_all[:, alpha_idx] = p_selected

    # Actuals for validation
    actuals = Y[lookback:, :]

    # Rolling alpha selection
    forecasts_all_transposed = forecasts_all.transpose(0, 1)
    alpha_optimal, _alpha_counts, _, p_optimal_all_days = rolling_alpha_selection(
        forecasts_all_batched=forecasts_all_transposed,
        p_alpha_all=p_alpha_all,
        actuals=actuals,
        validation_days=validation_days,
        alpha_grid=alpha_grid,
        verbose=True,
        use_greek_symbol=True,
    )

    # Print p statistics
    p_optimal_np = p_optimal_all_days.cpu().numpy()
    print(f"  Selected p statistics: mean={p_optimal_np.mean():.2f}, "
          f"median={int(np.median(p_optimal_np))}, "
          f"min={p_optimal_np.min()}, max={p_optimal_np.max()}")

    # =========================================================================
    # Phase 7: Extract final forecasts
    # =========================================================================
    print("  Phase 7: Extracting final forecasts...")

    forecasts_sliced = forecasts_all_transposed[:, validation_days:, :]
    p_indices = (p_optimal_all_days - 1).unsqueeze(0).unsqueeze(-1).expand(n_assets, -1, 1)
    forecasts_final = torch.gather(forecasts_sliced, dim=2, index=p_indices).squeeze(-1)

    # Build forecasts_all for result: (n_assets, n_output_days, n_alphas)
    forecasts_all_output = torch.zeros(n_assets, n_output_days, n_alphas, device=dev, dtype=dtype)
    p_alpha_output = p_alpha_all[validation_days:, :]

    for alpha_idx in range(n_alphas):
        p_idx = (p_alpha_output[:, alpha_idx] - 1).unsqueeze(0).unsqueeze(-1).expand(n_assets, -1, 1)
        forecasts_all_output[:, :, alpha_idx] = torch.gather(forecasts_sliced, dim=2, index=p_idx).squeeze(-1)

    # Build coefficients_all: (n_output_days, n_alphas, p_max, n_assets, n_assets)
    coefficients_all = torch.zeros(
        n_output_days, n_alphas, p_max, n_assets, n_assets,
        device=dev, dtype=dtype
    )

    for alpha_idx in range(n_alphas):
        for d in range(n_output_days):
            day_idx_full = validation_days + d
            p_selected = p_alpha_all[day_idx_full, alpha_idx].item()
            coefficients_all[d, alpha_idx, :p_selected, :, :] = theta_all[day_idx_full, :p_selected, :, :]

    p_optimal_all_output = p_alpha_all[validation_days:, :]
    SE_all_output = SE_all

    total_time = time.time() - t0
    print(f"  Complete! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return ORACLEVARXResult(
        forecasts=forecasts_final,
        forecasts_all=forecasts_all_output,
        p_optimal_all=p_optimal_all_output,
        alpha_optimal=alpha_optimal,
        p_optimal=p_optimal_all_days,
        alpha_grid=alpha_grid,
        coefficients_all=coefficients_all,
        asset_names=asset_names,
        confounder_names=confounder_names,
        dates=dates,
        SE_all=SE_all_output,
    )
