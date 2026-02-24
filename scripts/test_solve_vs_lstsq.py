"""
Test A: CPU-only comparison of torch.linalg.solve vs torch.linalg.lstsq

Feeds identical synthetic residuals through both solvers on CPU and traces
divergence at every pipeline stage:
  1. OLS coefficients (theta)
  2. Standard errors (SE)
  3. Z-statistics
  4. BH rejections
  5. P-selection (per alpha)
  6. Rolling alpha selection
  7. Final p_optimal
"""

import argparse
import sys
import os

import torch
from torch.distributions import Normal

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modules.batch_utils import batched_benjamini_hochberg
from src.models.var_pytorch import rolling_alpha_selection, select_optimal_p


def parse_args():
    parser = argparse.ArgumentParser(description="Test A: solve vs lstsq on CPU")
    parser.add_argument("--n-days", type=int, default=2500, help="Total number of days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ols-window", type=int, default=504, help="OLS rolling window size")
    parser.add_argument("--p-max", type=int, default=10, help="Maximum lag order")
    parser.add_argument("--n-assets", type=int, default=9, help="Number of assets")
    parser.add_argument("--validation-days", type=int, default=21, help="Rolling validation window")
    return parser.parse_args()


def generate_synthetic_residuals(n_days, n_assets, p_max, seed):
    """Generate synthetic residuals with mild correlation structure."""
    torch.manual_seed(seed)
    # Cholesky factor for correlation
    raw = torch.randn(n_assets, n_assets)
    L_Y = torch.linalg.cholesky(raw @ raw.T + 0.1 * torch.eye(n_assets))

    # Y residuals: (n_days, n_assets)
    R_Y = (torch.randn(n_days, n_assets) @ L_Y.T) * 0.01  # scale to realistic returns

    # T residuals per lag p: (n_days, n_assets * p_max)
    n_treatments_max = n_assets * p_max
    raw_T = torch.randn(n_treatments_max, n_treatments_max)
    L_T = torch.linalg.cholesky(raw_T @ raw_T.T + 0.1 * torch.eye(n_treatments_max))
    R_T = (torch.randn(n_days, n_treatments_max) @ L_T.T) * 0.01

    return R_Y, R_T


def batched_ols_solve(X_batch, Y_batch):
    """OLS via torch.linalg.solve (normal equations)."""
    XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)
    XtY = torch.bmm(X_batch.transpose(1, 2), Y_batch)
    return torch.linalg.solve(XtX, XtY)


def batched_ols_lstsq(X_batch, Y_batch, rcond=None):
    """OLS via torch.linalg.lstsq (normal equations)."""
    XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)
    XtY = torch.bmm(X_batch.transpose(1, 2), Y_batch)
    return torch.linalg.lstsq(XtX, XtY, rcond=rcond).solution


def compute_se_batched(theta_batch, T_windows, Y_windows, ols_window):
    """
    Compute batched standard errors.

    theta_batch: (batch, n_treatments, n_assets)
    T_windows:   (batch, ols_window, n_treatments)
    Y_windows:   (batch, ols_window, n_assets)
    """
    n_treatments = T_windows.shape[2]

    TtT = torch.bmm(T_windows.transpose(1, 2), T_windows)
    TtT_inv = torch.linalg.inv(TtT)
    TtT_inv_diag = torch.diagonal(TtT_inv, dim1=-2, dim2=-1)  # (batch, n_treatments)

    Y_pred = torch.bmm(T_windows, theta_batch)
    residuals = Y_windows - Y_pred
    RSS = (residuals ** 2).sum(dim=1)  # (batch, n_assets)
    df = ols_window - n_treatments
    sigma_sq = RSS / df  # (batch, n_assets)

    se = torch.sqrt(TtT_inv_diag.unsqueeze(-1) * sigma_sq.unsqueeze(1))
    return se  # (batch, n_treatments, n_assets)


def run_ols_pipeline(R_Y, R_T, n_assets, p_max, ols_window):
    """
    Run OLS for each lag p, returning theta and SE for both solve and lstsq.

    Returns:
        theta_solve_all: dict p -> (n_test_days, n_treatments_p, n_assets)
        theta_lstsq_all: dict p -> same
        se_solve_all:    dict p -> same
        se_lstsq_all:    dict p -> same
        n_test_days: int
    """
    n_days = R_Y.shape[0]
    n_test_days = n_days - ols_window

    theta_solve_all = {}
    theta_lstsq_all = {}
    se_solve_all = {}
    se_lstsq_all = {}

    for p in range(1, p_max + 1):
        n_treatments = n_assets * p
        T_residuals = R_T[:, :n_treatments]  # (n_days, n_treatments)

        # Build sliding windows
        T_windows = torch.stack(
            [T_residuals[d : d + ols_window] for d in range(n_test_days)], dim=0
        )  # (n_test_days, ols_window, n_treatments)
        Y_windows = torch.stack(
            [R_Y[d : d + ols_window] for d in range(n_test_days)], dim=0
        )  # (n_test_days, ols_window, n_assets)

        # Solve variant
        theta_solve = batched_ols_solve(T_windows, Y_windows)
        se_solve = compute_se_batched(theta_solve, T_windows, Y_windows, ols_window)

        # Lstsq variant
        theta_lstsq = batched_ols_lstsq(T_windows, Y_windows)
        se_lstsq = compute_se_batched(theta_lstsq, T_windows, Y_windows, ols_window)

        theta_solve_all[p] = theta_solve
        theta_lstsq_all[p] = theta_lstsq
        se_solve_all[p] = se_solve
        se_lstsq_all[p] = se_lstsq

    return theta_solve_all, theta_lstsq_all, se_solve_all, se_lstsq_all, n_test_days


def report_divergence(name, a, b, prefix="  "):
    """Print max|delta|, mean|delta|, max_rel for two tensors."""
    delta = (a - b).abs()
    denom = a.abs().clamp(min=1e-20)
    max_abs = delta.max().item()
    mean_abs = delta.mean().item()
    max_rel = (delta / denom).max().item() * 100
    print(f"{prefix}max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  max_rel={max_rel:.4f}%")


def run_significance_pipeline(theta_all, se_all, n_assets, p_max, n_test_days, alpha_grid):
    """
    Run Phase 6 significance pipeline.

    Returns:
        p_selected: (n_test_days, n_alphas) — selected lag per (day, alpha)
        z_all: dict p -> (n_test_days, n_treatments, n_assets) z-statistics
        reject_all: dict (p, alpha) -> (n_test_days,) boolean any-rejection
    """
    normal = Normal(0, 1)
    n_alphas = len(alpha_grid)

    z_all = {}
    reject_all = {}

    # p_selected: (n_test_days, n_alphas) initialized to 1
    p_selected = torch.ones(n_test_days, n_alphas, dtype=torch.long)

    for p in range(2, p_max + 1):
        # Extract lag-p coefficients: last n_assets rows
        theta_p = theta_all[p][:, (p - 1) * n_assets : p * n_assets, :]  # (n_test, n_assets, n_assets)
        se_p = se_all[p][:, (p - 1) * n_assets : p * n_assets, :]

        z = (theta_p / (se_p + 1e-10)).abs()
        z_all[p] = z

        # p-values: two-tailed
        p_vals = 2 * (1 - normal.cdf(z))
        p_vals_flat = p_vals.reshape(n_test_days, n_assets * n_assets)

        for ai, alpha in enumerate(alpha_grid):
            reject = batched_benjamini_hochberg(p_vals_flat, alpha)
            is_sig = reject.any(dim=1)  # (n_test_days,)
            reject_all[(p, alpha)] = is_sig

            still_active = (p_selected[:, ai] == p - 1)
            should_update = is_sig & still_active
            p_selected[:, ai] = torch.where(should_update, torch.tensor(p), p_selected[:, ai])

    return p_selected, z_all, reject_all


def generate_synthetic_forecasts(n_test_days, n_assets, p_max, seed):
    """Generate synthetic forecasts for each lag p."""
    torch.manual_seed(seed + 1000)
    # forecasts_all: (n_test_days, n_assets, p_max)
    forecasts_all = torch.randn(n_test_days, n_assets, p_max) * 0.005
    # actuals: (n_test_days, n_assets)
    actuals = torch.randn(n_test_days, n_assets) * 0.01
    return forecasts_all, actuals


def main():
    args = parse_args()
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float64)

    alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("=" * 60)
    print("Test A: torch.linalg.solve vs torch.linalg.lstsq (CPU only)")
    print("=" * 60)
    print(f"Config: n_assets={args.n_assets}, p_max={args.p_max}, "
          f"ols_window={args.ols_window}, n_days={args.n_days}, seed={args.seed}")
    print()

    # --- Generate data ---
    print("Generating synthetic residuals...")
    R_Y, R_T = generate_synthetic_residuals(args.n_days, args.n_assets, args.p_max, args.seed)
    n_test_days = args.n_days - args.ols_window

    print(f"n_test_days = {n_test_days}")
    print()

    # --- Stage 1: OLS Coefficients ---
    print("Running OLS pipelines (this may take a moment)...")
    theta_solve, theta_lstsq, se_solve, se_lstsq, _ = run_ols_pipeline(
        R_Y, R_T, args.n_assets, args.p_max, args.ols_window
    )

    print("\n--- Stage 1: OLS Coefficients (theta) ---")
    for p in range(1, args.p_max + 1):
        print(f"  p={p}:", end="")
        report_divergence("theta", theta_solve[p], theta_lstsq[p], prefix="  ")

    # --- Stage 2: Standard Errors ---
    print("\n--- Stage 2: Standard Errors (SE) ---")
    for p in range(1, args.p_max + 1):
        print(f"  p={p}:", end="")
        report_divergence("SE", se_solve[p], se_lstsq[p], prefix="  ")

    # --- Stage 3: Z-statistics ---
    print("\n--- Stage 3: Z-statistics ---")
    normal = Normal(0, 1)
    z_critical_values = [1.645, 1.96, 2.326, 2.576]  # 90%, 95%, 98%, 99%

    for p in range(2, args.p_max + 1):
        # Compute z for both
        theta_s_p = theta_solve[p][:, (p-1)*args.n_assets:p*args.n_assets, :]
        se_s_p = se_solve[p][:, (p-1)*args.n_assets:p*args.n_assets, :]
        z_solve = (theta_s_p / (se_s_p + 1e-10)).abs()

        theta_l_p = theta_lstsq[p][:, (p-1)*args.n_assets:p*args.n_assets, :]
        se_l_p = se_lstsq[p][:, (p-1)*args.n_assets:p*args.n_assets, :]
        z_lstsq = (theta_l_p / (se_l_p + 1e-10)).abs()

        delta_z = (z_solve - z_lstsq).abs()
        n_total = z_solve.numel()

        # Count boundary crossings: z crosses a critical value differently
        boundary_crossings = 0
        for zc in z_critical_values:
            above_solve = z_solve > zc
            above_lstsq = z_lstsq > zc
            boundary_crossings += (above_solve != above_lstsq).sum().item()

        print(f"  p={p}:  max|Δ|={delta_z.max().item():.3e}  "
              f"mean|Δ|={delta_z.mean().item():.3e}  "
              f"boundary_crossings={boundary_crossings}/{n_total} "
              f"({100*boundary_crossings/n_total:.2f}%)")

    # --- Stage 4: BH Rejections ---
    print("\n--- Stage 4: BH Rejections ---")
    p_sel_solve, z_solve_all, rej_solve = run_significance_pipeline(
        theta_solve, se_solve, args.n_assets, args.p_max, n_test_days, alpha_grid
    )
    p_sel_lstsq, z_lstsq_all, rej_lstsq = run_significance_pipeline(
        theta_lstsq, se_lstsq, args.n_assets, args.p_max, n_test_days, alpha_grid
    )

    for alpha in alpha_grid:
        flipped = 0
        total = 0
        for p in range(2, args.p_max + 1):
            rej_s = rej_solve[(p, alpha)]
            rej_l = rej_lstsq[(p, alpha)]
            flipped += (rej_s != rej_l).sum().item()
            total += rej_s.numel()
        print(f"  alpha={alpha:.2f}:  flipped_decisions={flipped}/{total} "
              f"({100*flipped/total:.2f}%)")

    # --- Stage 5: P-selection per alpha ---
    print("\n--- Stage 5: P-selection (per alpha) ---")
    for ai, alpha in enumerate(alpha_grid):
        p_s = p_sel_solve[:, ai]
        p_l = p_sel_lstsq[:, ai]
        differ = (p_s != p_l).sum().item()
        print(f"  alpha={alpha:.2f}:  days_differ={differ}/{n_test_days}  "
              f"mean_p_solve={p_s.float().mean().item():.2f}  "
              f"mean_p_lstsq={p_l.float().mean().item():.2f}")

    # --- Stage 6: Rolling Alpha Selection ---
    print("\n--- Stage 6: Rolling Alpha Selection ---")
    forecasts_all, actuals = generate_synthetic_forecasts(
        n_test_days, args.n_assets, args.p_max, args.seed
    )

    # rolling_alpha_selection expects:
    #   forecasts_all_batched: (n_assets, n_test_days, p_max)
    #   p_alpha_all: (n_test_days, n_alphas)
    #   actuals: (n_test_days, n_assets)
    forecasts_batched = forecasts_all.permute(1, 0, 2)  # (n_assets, n_test_days, p_max)

    alpha_opt_solve, _, _, p_opt_solve = rolling_alpha_selection(
        forecasts_batched, p_sel_solve, actuals,
        args.validation_days, alpha_grid, verbose=False
    )
    alpha_opt_lstsq, _, _, p_opt_lstsq = rolling_alpha_selection(
        forecasts_batched, p_sel_lstsq, actuals,
        args.validation_days, alpha_grid, verbose=False
    )

    n_output = alpha_opt_solve.shape[0]
    alpha_differ = (alpha_opt_solve != alpha_opt_lstsq).sum().item()
    print(f"  days_differ={alpha_differ}/{n_output} ({100*alpha_differ/n_output:.2f}%)")

    # --- Stage 7: Final p_optimal ---
    print("\n--- Stage 7: Final p_optimal ---")
    p_differ = (p_opt_solve != p_opt_lstsq).sum().item()
    print(f"  days_differ={p_differ}/{n_output} ({100*p_differ/n_output:.2f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    max_theta_delta = max(
        (theta_solve[p] - theta_lstsq[p]).abs().max().item()
        for p in range(1, args.p_max + 1)
    )
    max_se_delta = max(
        (se_solve[p] - se_lstsq[p]).abs().max().item()
        for p in range(1, args.p_max + 1)
    )
    print(f"  Max theta divergence:    {max_theta_delta:.3e}")
    print(f"  Max SE divergence:       {max_se_delta:.3e}")
    print(f"  Alpha selection differ:  {alpha_differ}/{n_output} ({100*alpha_differ/n_output:.2f}%)")
    print(f"  Final p_optimal differ:  {p_differ}/{n_output} ({100*p_differ/n_output:.2f}%)")


if __name__ == "__main__":
    main()
