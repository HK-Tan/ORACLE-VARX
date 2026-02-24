#!/usr/bin/env python3
"""Generate and save synthetic toy benchmark data.

Produces Y.csv, W.csv, A_true.pt, and dgp_config.json in dataset/toy/.
Run once; experiment scripts load from CSV.

Usage:
    python scripts/generate_toy_data.py [--seed 42] [--noise-scale 1.0]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic.dgp import ToyDGPConfig, generate_toy_data


def main():
    parser = argparse.ArgumentParser(description="Generate toy benchmark data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-scale", type=float, default=0.05,
                        help="Innovation noise scaling ν (default: 0.05)")
    parser.add_argument("--confounder-strength", type=float, default=1.0,
                        help="Confounder nuisance scaling λ (default: 1.0)")
    parser.add_argument("--T", type=int, default=3000)
    args = parser.parse_args()

    config = ToyDGPConfig(
        seed=args.seed, noise_scale=args.noise_scale,
        confounder_strength=args.confounder_strength, T=args.T,
    )

    print(f"Generating toy data: T={config.T}, seed={config.seed}, noise_scale={config.noise_scale}")
    Y, W, truth = generate_toy_data(config)

    # Save to dataset/toy/
    out_dir = Path("dataset/toy")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Y.csv
    Y_df = pd.DataFrame(Y, columns=config.endo_names)
    Y_df.to_csv(out_dir / "Y.csv", index=False)
    print(f"  Y.csv: shape {Y_df.shape}")

    # W.csv
    W_df = pd.DataFrame(W, columns=config.confounder_names)
    W_df.to_csv(out_dir / "W.csv", index=False)
    print(f"  W.csv: shape {W_df.shape}")

    # A_true.pt
    torch.save(truth.A_true, out_dir / "A_true.pt")
    print(f"  A_true.pt: shape {tuple(truth.A_true.shape)}")

    # dgp_config.json
    config_dict = {
        "T": config.T,
        "n_endo": config.n_endo,
        "n_confounders": config.n_confounders,
        "burn_in": config.burn_in,
        "seed": config.seed,
        "noise_scale": config.noise_scale,
        "confounder_strength": config.confounder_strength,
        "ar_rho": config.ar_rho,
        "cross_effect": config.cross_effect,
        "a_XX_init": config.a_XX_init,
        "a_YY_init": config.a_YY_init,
        "a_ZZ_init": config.a_ZZ_init,
        "a_XX_decay_end": config.a_XX_decay_end,
        "a_YY_decay_end": config.a_YY_decay_end,
        "a_ZZ_decay_end": config.a_ZZ_decay_end,
        "p_max": config.p_max,
        "ols_window": config.ols_window,
        "tree_train_window": config.tree_train_window,
        "p_max_offset": config.p_max_offset,
        "test_size": config.test_size,
        "validation_days": config.validation_days,
        "endo_names": config.endo_names,
        "confounder_names": config.confounder_names,
    }
    with open(out_dir / "dgp_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  dgp_config.json saved")

    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"Y mean:  {Y.mean(axis=0)}")
    print(f"Y std:   {Y.std(axis=0)}")
    print(f"W mean:  {W.mean(axis=0)}")
    print(f"W std:   {W.std(axis=0)}")

    # Check A_true regimes
    A = truth.A_true.numpy()
    for start, end in truth.regime_boundaries:
        mid = (start + end) // 2
        nonzero = np.count_nonzero(np.abs(A[mid]) > 1e-8)
        print(f"Regime [{start}, {end}): {nonzero} non-zero entries at t={mid}")

    print(f"\nFiles saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
