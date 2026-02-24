"""
Test B: CPU BLAS vs GPU BLAS rounding differences

Compares three variants using the SAME solver (torch.linalg.solve):
  Variant A: theta on CPU + SE on CPU  (all MKL/OpenBLAS)
  Variant B: theta on GPU + SE on GPU  (all cuBLAS/MAGMA)
  Variant C: theta on GPU + SE on CPU  (mixed — the current code path)

Traces divergence at every pipeline stage (same format as Test A).
Exits gracefully if no GPU is available.
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
    parser = argparse.ArgumentParser(description="Test B: CPU vs GPU BLAS rounding")
    parser.add_argument("--n-days", type=int, default=2500, help="Total number of days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ols-window", type=int, default=504, help="OLS rolling window size")
    parser.add_argument("--p-max", type=int, default=10, help="Maximum lag order")
    parser.add_argument("--n-assets", type=int, default=9, help="Number of assets")
    parser.add_argument("--validation-days", type=int, default=21, help="Rolling validation window")
    return parser.parse_args()


def generate_synthetic_residuals(n_days, n_assets, p_max, seed):
    """Generate synthetic residuals with mild correlation structure (CPU)."""
    torch.manual_seed(seed)
    raw = torch.randn(n_assets, n_assets, dtype=torch.float64)
    L_Y = torch.linalg.cholesky(raw @ raw.T + 0.1 * torch.eye(n_assets, dtype=torch.float64))
    R_Y = (torch.randn(n_days, n_assets, dtype=torch.float64) @ L_Y.T) * 0.01

    n_treatments_max = n_assets * p_max
    raw_T = torch.randn(n_treatments_max, n_treatments_max, dtype=torch.float64)
    L_T = torch.linalg.cholesky(raw_T @ raw_T.T + 0.1 * torch.eye(n_treatments_max, dtype=torch.float64))
    R_T = (torch.randn(n_days, n_treatments_max, dtype=torch.float64) @ L_T.T) * 0.01

    return R_Y, R_T


def batched_ols_solve(X_batch, Y_batch):
    """OLS via torch.linalg.solve (normal equations)."""
    XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)
    XtY = torch.bmm(X_batch.transpose(1, 2), Y_batch)
    return torch.linalg.solve(XtX, XtY)


def compute_se_batched(theta_batch, T_windows, Y_windows, ols_window):
    """Compute batched standard errors."""
    n_treatments = T_windows.shape[2]
    TtT = torch.bmm(T_windows.transpose(1, 2), T_windows)
    TtT_inv = torch.linalg.inv(TtT)
    TtT_inv_diag = torch.diagonal(TtT_inv, dim1=-2, dim2=-1)

    Y_pred = torch.bmm(T_windows, theta_batch)
    residuals = Y_windows - Y_pred
    RSS = (residuals ** 2).sum(dim=1)
    df = ols_window - n_treatments
    sigma_sq = RSS / df

    se = torch.sqrt(TtT_inv_diag.unsqueeze(-1) * sigma_sq.unsqueeze(1))
    return se


def build_windows(R_Y, R_T, n_assets, p, ols_window, n_test_days):
    """Build sliding OLS windows for a given lag p."""
    n_treatments = n_assets * p
    T_residuals = R_T[:, :n_treatments]
    T_windows = torch.stack(
        [T_residuals[d : d + ols_window] for d in range(n_test_days)], dim=0
    )
    Y_windows = torch.stack(
        [R_Y[d : d + ols_window] for d in range(n_test_days)], dim=0
    )
    return T_windows, Y_windows


def run_variant(label, R_Y, R_T, n_assets, p_max, ols_window, n_test_days,
                theta_device, se_device):
    """
    Run OLS pipeline on specified devices.
    theta_device: device for computing theta
    se_device: device for computing SE (uses theta moved to se_device)
    """
    theta_all = {}
    se_all = {}

    for p in range(1, p_max + 1):
        T_windows_cpu, Y_windows_cpu = build_windows(
            R_Y, R_T, n_assets, p, ols_window, n_test_days
        )

        # Compute theta on theta_device
        T_w = T_windows_cpu.to(theta_device)
        Y_w = Y_windows_cpu.to(theta_device)
        theta = batched_ols_solve(T_w, Y_w)

        # Compute SE on se_device
        if se_device != theta_device:
            theta_se = theta.to(se_device)
            T_w_se = T_windows_cpu.to(se_device)
            Y_w_se = Y_windows_cpu.to(se_device)
        else:
            theta_se = theta
            T_w_se = T_w
            Y_w_se = Y_w

        se = compute_se_batched(theta_se, T_w_se, Y_w_se, ols_window)

        # Move back to CPU for downstream comparison
        theta_all[p] = theta.cpu()
        se_all[p] = se.cpu()

    return theta_all, se_all


def run_significance_pipeline(theta_all, se_all, n_assets, p_max, n_test_days, alpha_grid):
    """Run Phase 6 significance pipeline (CPU)."""
    normal = Normal(0, 1)
    n_alphas = len(alpha_grid)

    z_all = {}
    reject_all = {}
    p_selected = torch.ones(n_test_days, n_alphas, dtype=torch.long)

    for p in range(2, p_max + 1):
        theta_p = theta_all[p][:, (p-1)*n_assets:p*n_assets, :]
        se_p = se_all[p][:, (p-1)*n_assets:p*n_assets, :]

        z = (theta_p / (se_p + 1e-10)).abs()
        z_all[p] = z

        p_vals = 2 * (1 - normal.cdf(z))
        p_vals_flat = p_vals.reshape(n_test_days, n_assets * n_assets)

        for ai, alpha in enumerate(alpha_grid):
            reject = batched_benjamini_hochberg(p_vals_flat, alpha)
            is_sig = reject.any(dim=1)
            reject_all[(p, alpha)] = is_sig

            still_active = (p_selected[:, ai] == p - 1)
            should_update = is_sig & still_active
            p_selected[:, ai] = torch.where(should_update, torch.tensor(p), p_selected[:, ai])

    return p_selected, z_all, reject_all


def generate_synthetic_forecasts(n_test_days, n_assets, p_max, seed):
    """Generate synthetic forecasts for each lag p."""
    torch.manual_seed(seed + 1000)
    forecasts_all = torch.randn(n_test_days, n_assets, p_max, dtype=torch.float64) * 0.005
    actuals = torch.randn(n_test_days, n_assets, dtype=torch.float64) * 0.01
    return forecasts_all, actuals


def report_divergence(a, b, prefix="  "):
    """Print max|delta|, mean|delta|, max_rel for two tensors."""
    delta = (a - b).abs()
    denom = a.abs().clamp(min=1e-20)
    max_abs = delta.max().item()
    mean_abs = delta.mean().item()
    max_rel = (delta / denom).max().item() * 100
    print(f"{prefix}max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  max_rel={max_rel:.4f}%")


def print_comparison(label_a, label_b, theta_a, theta_b, se_a, se_b,
                     p_sel_a, p_sel_b, args, n_test_days, alpha_grid):
    """Print full stage-by-stage comparison between two variants."""
    print(f"\n{'='*60}")
    print(f"Comparing: {label_a} vs {label_b}")
    print(f"{'='*60}")

    # Stage 1: theta
    print("\n--- Stage 1: OLS Coefficients (theta) ---")
    for p in range(1, args.p_max + 1):
        print(f"  p={p}:", end="")
        report_divergence(theta_a[p], theta_b[p], prefix="  ")

    # Stage 2: SE
    print("\n--- Stage 2: Standard Errors (SE) ---")
    for p in range(1, args.p_max + 1):
        print(f"  p={p}:", end="")
        report_divergence(se_a[p], se_b[p], prefix="  ")

    # Stage 3: Z-stats
    print("\n--- Stage 3: Z-statistics ---")
    z_critical_values = [1.645, 1.96, 2.326, 2.576]
    for p in range(2, args.p_max + 1):
        theta_a_p = theta_a[p][:, (p-1)*args.n_assets:p*args.n_assets, :]
        se_a_p = se_a[p][:, (p-1)*args.n_assets:p*args.n_assets, :]
        z_a = (theta_a_p / (se_a_p + 1e-10)).abs()

        theta_b_p = theta_b[p][:, (p-1)*args.n_assets:p*args.n_assets, :]
        se_b_p = se_b[p][:, (p-1)*args.n_assets:p*args.n_assets, :]
        z_b = (theta_b_p / (se_b_p + 1e-10)).abs()

        delta_z = (z_a - z_b).abs()
        n_total = z_a.numel()

        boundary_crossings = 0
        for zc in z_critical_values:
            above_a = z_a > zc
            above_b = z_b > zc
            boundary_crossings += (above_a != above_b).sum().item()

        print(f"  p={p}:  max|Δ|={delta_z.max().item():.3e}  "
              f"mean|Δ|={delta_z.mean().item():.3e}  "
              f"boundary_crossings={boundary_crossings}/{n_total} "
              f"({100*boundary_crossings/n_total:.2f}%)")

    # Stage 4: BH rejections
    print("\n--- Stage 4: BH Rejections ---")
    _, _, rej_a = run_significance_pipeline(
        theta_a, se_a, args.n_assets, args.p_max, n_test_days, alpha_grid
    )
    _, _, rej_b = run_significance_pipeline(
        theta_b, se_b, args.n_assets, args.p_max, n_test_days, alpha_grid
    )

    for alpha in alpha_grid:
        flipped = 0
        total = 0
        for p in range(2, args.p_max + 1):
            r_a = rej_a[(p, alpha)]
            r_b = rej_b[(p, alpha)]
            flipped += (r_a != r_b).sum().item()
            total += r_a.numel()
        print(f"  alpha={alpha:.2f}:  flipped_decisions={flipped}/{total} "
              f"({100*flipped/total:.2f}%)")

    # Stage 5: P-selection per alpha
    print("\n--- Stage 5: P-selection (per alpha) ---")
    for ai, alpha in enumerate(alpha_grid):
        p_a = p_sel_a[:, ai]
        p_b = p_sel_b[:, ai]
        differ = (p_a != p_b).sum().item()
        print(f"  alpha={alpha:.2f}:  days_differ={differ}/{n_test_days}  "
              f"mean_p_A={p_a.float().mean().item():.2f}  "
              f"mean_p_B={p_b.float().mean().item():.2f}")

    # Stage 6 & 7: Rolling alpha + final p
    print("\n--- Stage 6: Rolling Alpha Selection ---")
    forecasts_all, actuals = generate_synthetic_forecasts(
        n_test_days, args.n_assets, args.p_max, args.seed
    )
    forecasts_batched = forecasts_all.permute(1, 0, 2)

    alpha_opt_a, _, _, p_opt_a = rolling_alpha_selection(
        forecasts_batched, p_sel_a, actuals,
        args.validation_days, alpha_grid, verbose=False
    )
    alpha_opt_b, _, _, p_opt_b = rolling_alpha_selection(
        forecasts_batched, p_sel_b, actuals,
        args.validation_days, alpha_grid, verbose=False
    )

    n_output = alpha_opt_a.shape[0]
    alpha_differ = (alpha_opt_a != alpha_opt_b).sum().item()
    print(f"  days_differ={alpha_differ}/{n_output} ({100*alpha_differ/n_output:.2f}%)")

    print("\n--- Stage 7: Final p_optimal ---")
    p_differ = (p_opt_a != p_opt_b).sum().item()
    print(f"  days_differ={p_differ}/{n_output} ({100*p_differ/n_output:.2f}%)")

    return alpha_differ, p_differ, n_output


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("=" * 60)
        print("Test B: CPU vs GPU BLAS rounding")
        print("=" * 60)
        print("\nNo GPU available. This test requires CUDA.")
        print("Please run on a GPU-enabled machine.")
        sys.exit(0)

    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    torch.set_default_dtype(torch.float64)

    alpha_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("=" * 60)
    print("Test B: CPU vs GPU BLAS rounding (same solver: solve)")
    print("=" * 60)
    print(f"Config: n_assets={args.n_assets}, p_max={args.p_max}, "
          f"ols_window={args.ols_window}, n_days={args.n_days}, seed={args.seed}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Generate data on CPU
    print("Generating synthetic residuals...")
    R_Y, R_T = generate_synthetic_residuals(args.n_days, args.n_assets, args.p_max, args.seed)
    n_test_days = args.n_days - args.ols_window
    print(f"n_test_days = {n_test_days}")

    # Run three variants
    print("\nRunning Variant A: theta=CPU, SE=CPU (all MKL/OpenBLAS)...")
    theta_A, se_A = run_variant("A", R_Y, R_T, args.n_assets, args.p_max,
                                args.ols_window, n_test_days, cpu, cpu)

    print("Running Variant B: theta=GPU, SE=GPU (all cuBLAS/MAGMA)...")
    theta_B, se_B = run_variant("B", R_Y, R_T, args.n_assets, args.p_max,
                                args.ols_window, n_test_days, gpu, gpu)

    print("Running Variant C: theta=GPU, SE=CPU (mixed — current code path)...")
    theta_C, se_C = run_variant("C", R_Y, R_T, args.n_assets, args.p_max,
                                args.ols_window, n_test_days, gpu, cpu)

    # Run significance pipeline for each
    p_sel_A, _, _ = run_significance_pipeline(theta_A, se_A, args.n_assets,
                                              args.p_max, n_test_days, alpha_grid)
    p_sel_B, _, _ = run_significance_pipeline(theta_B, se_B, args.n_assets,
                                              args.p_max, n_test_days, alpha_grid)
    p_sel_C, _, _ = run_significance_pipeline(theta_C, se_C, args.n_assets,
                                              args.p_max, n_test_days, alpha_grid)

    # Print pairwise comparisons
    results = {}

    ad, pd_, no = print_comparison(
        "A (CPU/CPU)", "B (GPU/GPU)",
        theta_A, theta_B, se_A, se_B, p_sel_A, p_sel_B,
        args, n_test_days, alpha_grid
    )
    results["A_vs_B"] = (ad, pd_, no)

    ad, pd_, no = print_comparison(
        "A (CPU/CPU)", "C (GPU/CPU mixed)",
        theta_A, theta_C, se_A, se_C, p_sel_A, p_sel_C,
        args, n_test_days, alpha_grid
    )
    results["A_vs_C"] = (ad, pd_, no)

    ad, pd_, no = print_comparison(
        "B (GPU/GPU)", "C (GPU/CPU mixed)",
        theta_B, theta_C, se_B, se_C, p_sel_B, p_sel_C,
        args, n_test_days, alpha_grid
    )
    results["B_vs_C"] = (ad, pd_, no)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, (ad, pd_, no) in results.items():
        print(f"  {name}: alpha_differ={ad}/{no} ({100*ad/no:.2f}%), "
              f"p_differ={pd_}/{no} ({100*pd_/no:.2f}%)")

    print("\nInterpretation:")
    a_vs_b = results["A_vs_B"]
    a_vs_c = results["A_vs_C"]
    b_vs_c = results["B_vs_C"]

    if a_vs_b[1] > b_vs_c[1]:
        print("  -> CPU vs GPU BLAS rounding dominates over mixing backends")
    elif b_vs_c[1] > a_vs_b[1]:
        print("  -> Backend mixing (GPU theta + CPU SE) dominates over BLAS rounding")
    else:
        print("  -> Both effects are comparable")


if __name__ == "__main__":
    main()
