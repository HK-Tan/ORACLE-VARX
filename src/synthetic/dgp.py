"""Data-Generating Process for the 3-variable time-varying confounded toy benchmark."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ToyDGPConfig:
    """Configuration for the toy DGP."""

    T: int = 3000
    n_endo: int = 3
    n_confounders: int = 3
    burn_in: int = 50
    seed: int = 42
    noise_scale: float = 0.05

    # Confounder nuisance scaling (λ): g_X, g_Y, g_Z are multiplied by this
    confounder_strength: float = 1.0

    # AR(1) coefficient for confounders
    ar_rho: float = 0.5

    # Fixed lag-1 cross-effects (Z->X, X->Y, Y->Z)
    cross_effect: float = 0.60

    # Time-varying self-effects (initial values)
    a_XX_init: float = 0.30  # X_{t-2} -> X_t
    a_YY_init: float = 0.30  # Y_{t-2} -> Y_t
    a_ZZ_init: float = 0.30  # Z_{t-3} -> Z_t

    # Decay endpoints
    a_XX_decay_end: int = 2000  # a_XX(t) = 0 at t >= 2000
    a_YY_decay_end: int = 2000  # a_YY(t) = 0 at t >= 2000
    a_ZZ_decay_end: int = 1000  # a_ZZ(t) = 0 at t >= 1000

    # Experiment config
    p_max: int = 5
    ols_window: int = 200
    tree_train_window: Optional[int] = None  # defaults to ols_window
    p_max_offset: int = 5
    test_size: int = 20
    validation_days: int = 20

    def __post_init__(self):
        if self.tree_train_window is None:
            self.tree_train_window = self.ols_window

    @property
    def lookback_var(self) -> int:
        return self.ols_window + self.p_max_offset

    @property
    def lookback_orvarx(self) -> int:
        return self.tree_train_window + self.ols_window + self.p_max_offset

    @property
    def endo_names(self) -> List[str]:
        return ["X", "Y", "Z"]

    @property
    def confounder_names(self) -> List[str]:
        return ["W1", "W2", "W3"]

    @property
    def regime_boundaries(self) -> List[Tuple[int, int]]:
        return [(0, 1000), (1000, 2000), (2000, 3000)]


@dataclass
class GroundTruth:
    """Ground truth coefficient matrices for evaluation.

    Attributes:
        A_true: shape (T, p_max_true, n_endo, n_endo)
                A_true[t, k, i, j] = effect of variable j at lag k+1 on variable i at time t
        p_max_true: true maximum lag (3)
        regime_boundaries: list of (start, end) tuples
        config: the ToyDGPConfig used
    """

    A_true: torch.Tensor
    p_max_true: int
    regime_boundaries: List[Tuple[int, int]]
    config: ToyDGPConfig


def _a_XX(t: np.ndarray, config: ToyDGPConfig) -> np.ndarray:
    """Time-varying coefficient X_{t-2} -> X_t."""
    return config.a_XX_init * np.maximum(0.0, 1.0 - t / config.a_XX_decay_end)


def _a_YY(t: np.ndarray, config: ToyDGPConfig) -> np.ndarray:
    """Time-varying coefficient Y_{t-2} -> Y_t."""
    return config.a_YY_init * np.maximum(0.0, 1.0 - t / config.a_YY_decay_end)


def _a_ZZ(t: np.ndarray, config: ToyDGPConfig) -> np.ndarray:
    """Time-varying coefficient Z_{t-3} -> Z_t."""
    return config.a_ZZ_init * np.maximum(0.0, 1.0 - t / config.a_ZZ_decay_end)


def _g_X(W: np.ndarray) -> np.ndarray:
    """Nonlinear nuisance: W_{t-1} -> X_t. Normalized to std ≈ 1."""
    W1, W2, W3 = W[..., 0], W[..., 1], W[..., 2]
    return (np.tanh(2.0 * W1) + W2**2 + np.sin(np.pi * W3)) / 2.2


def _g_Y(W: np.ndarray) -> np.ndarray:
    """Nonlinear nuisance: W_{t-1} -> Y_t. Normalized to std ≈ 1."""
    W1, W2, W3 = W[..., 0], W[..., 1], W[..., 2]
    return (np.maximum(0.0, W1 - 0.5) + np.sin(np.pi * W2) + W3**2) / 2.1


def _g_Z(W: np.ndarray) -> np.ndarray:
    """Nonlinear nuisance: W_{t-1} -> Z_t. Normalized to std ≈ 1."""
    W1, W2, W3 = W[..., 0], W[..., 1], W[..., 2]
    return (W1 * W2 + np.tanh(W3) + np.cos(np.pi * W1)) / 1.7


def generate_toy_data(
    config: ToyDGPConfig = None,
) -> Tuple[np.ndarray, np.ndarray, GroundTruth]:
    """Generate synthetic data from the 3-variable time-varying confounded DGP.

    Args:
        config: DGP configuration. If None, uses defaults.

    Returns:
        Y: endogenous variables, shape (T, 3), columns [X, Y, Z]
        W: confounders, shape (T, 3), columns [W1, W2, W3]
        truth: GroundTruth with A_true tensor and metadata
    """
    if config is None:
        config = ToyDGPConfig()

    rng = np.random.RandomState(config.seed)
    T_total = config.T + config.burn_in
    max_lag = 3  # true max lag

    # --- Generate confounders: independent AR(1) ---
    W_all = np.zeros((T_total, config.n_confounders))
    eta = rng.randn(T_total, config.n_confounders)
    for t in range(1, T_total):
        W_all[t] = config.ar_rho * W_all[t - 1] + eta[t]

    # --- Generate endogenous variables ---
    Y_all = np.zeros((T_total, config.n_endo))
    eps = rng.randn(T_total, config.n_endo)

    for t in range(max_lag, T_total):
        # Post-burn-in time index for decay functions
        t_decay = max(0, t - config.burn_in)

        # Nuisance from W_{t-1}, scaled by ν * λ
        nu_lam = config.noise_scale * config.confounder_strength
        g_x = nu_lam * _g_X(W_all[t - 1])
        g_y = nu_lam * _g_Y(W_all[t - 1])
        g_z = nu_lam * _g_Z(W_all[t - 1])

        # Time-varying coefficients
        axx = config.a_XX_init * max(0.0, 1.0 - t_decay / config.a_XX_decay_end)
        ayy = config.a_YY_init * max(0.0, 1.0 - t_decay / config.a_YY_decay_end)
        azz = config.a_ZZ_init * max(0.0, 1.0 - t_decay / config.a_ZZ_decay_end)

        # X_t = 0.60*Z_{t-1} + a_XX(t)*X_{t-2} + ν*λ*g_X(W_{t-1}) + ν*ε_X
        Y_all[t, 0] = (
            config.cross_effect * Y_all[t - 1, 2]
            + axx * Y_all[t - 2, 0]
            + g_x
            + config.noise_scale * eps[t, 0]
        )

        # Y_t = 0.60*X_{t-1} + a_YY(t)*Y_{t-2} + ν*λ*g_Y(W_{t-1}) + ν*ε_Y
        Y_all[t, 1] = (
            config.cross_effect * Y_all[t - 1, 0]
            + ayy * Y_all[t - 2, 1]
            + g_y
            + config.noise_scale * eps[t, 1]
        )

        # Z_t = 0.60*Y_{t-1} + a_ZZ(t)*Z_{t-3} + ν*λ*g_Z(W_{t-1}) + ν*ε_Z
        Y_all[t, 2] = (
            config.cross_effect * Y_all[t - 1, 1]
            + azz * Y_all[t - 3, 2]
            + g_z
            + config.noise_scale * eps[t, 2]
        )

    # --- Discard burn-in ---
    Y = Y_all[config.burn_in:]
    W = W_all[config.burn_in:]

    # --- Build ground truth A_true ---
    t_arr = np.arange(config.T, dtype=np.float64)
    p_max_true = 3

    A_true = np.zeros((config.T, p_max_true, config.n_endo, config.n_endo))

    # Lag 1 (k=0): fixed cross-effects Z->X, X->Y, Y->Z
    A_true[:, 0, 0, 2] = config.cross_effect  # Z_{t-1} -> X_t
    A_true[:, 0, 1, 0] = config.cross_effect  # X_{t-1} -> Y_t
    A_true[:, 0, 2, 1] = config.cross_effect  # Y_{t-1} -> Z_t

    # Lag 2 (k=1): time-varying self-effects
    A_true[:, 1, 0, 0] = _a_XX(t_arr, config)  # X_{t-2} -> X_t
    A_true[:, 1, 1, 1] = _a_YY(t_arr, config)  # Y_{t-2} -> Y_t

    # Lag 3 (k=2): time-varying self-effect
    A_true[:, 2, 2, 2] = _a_ZZ(t_arr, config)  # Z_{t-3} -> Z_t

    A_true_tensor = torch.from_numpy(A_true.astype(np.float32))

    truth = GroundTruth(
        A_true=A_true_tensor,
        p_max_true=p_max_true,
        regime_boundaries=config.regime_boundaries,
        config=config,
    )

    return Y, W, truth


def get_observed_confounders(
    W_full: np.ndarray, mode: str
) -> Tuple[np.ndarray, List[str]]:
    """Slice confounders by observability mode.

    Args:
        W_full: Full confounder matrix, shape (T, 3)
        mode: One of "all", "partial_2", "partial_1", "none"

    Returns:
        W_obs: Observed confounders (T, n_obs)
        obs_names: Names of observed confounders
    """
    all_names = ["W1", "W2", "W3"]
    if mode == "all":
        return W_full, all_names
    elif mode == "partial_2":
        return W_full[:, :2], all_names[:2]
    elif mode == "partial_1":
        return W_full[:, :1], all_names[:1]
    elif mode == "none":
        return np.empty((W_full.shape[0], 0)), []
    else:
        raise ValueError(f"Unknown observability mode: {mode}")
