"""Add stage (iii) BH edge metrics (FDR, power, F1) to saved toy results.

run_toy_benchmark.py computes these metrics when it fits a model. This script
computes the same metrics from saved result.pt files, so runs made before the
metric existed do not need refitting.

Each method pair shares one p_max fit, which gives the coefficients and SEs:
    VAR / ACLE-VAR, VARX / ACLE-VARX:   coefficients of VAR(X), SE_all of ACLE-VAR(X)
    OR-VARX / ORACLE-VARX (any learner): coefficients of OR-VARX, SE_all of ORACLE-VARX
Each method uses its own p_optimal. The metrics are merged into each run's
metrics.json and into metrics_summary.csv.

Usage (from code/):
    python scripts/eval_bh_edges.py
    python scripts/eval_bh_edges.py --all-lags   # diagnostic, writes nothing

--all-lags runs BH over all p_max lags on every day instead of lags 1..p_hat (the
range PCMCI tests) and only prints the metrics; the saved metrics are left unchanged.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_toy_benchmark import (  # noqa: E402
    OBS_LEVELS, LEARNERS, RESULTS_DIR, load_toy_data,
    compute_bh_edge_metrics, append_metrics_summary,
)


def method_pairs():
    """(coef/base method, SE/ACLE method, obs level) for every toy result."""
    pairs = [("VAR", "ACLE-VAR", "none")]
    pairs += [("VARX", "ACLE-VARX", obs) for obs in OBS_LEVELS]
    for learner in LEARNERS:
        pairs += [(f"OR-VARX_{learner}", f"ORACLE-VARX_{learner}", obs) for obs in OBS_LEVELS]
    pairs += [("OR-VARX-TabPFN", "ORACLE-VARX-TabPFN", obs) for obs in OBS_LEVELS]
    return pairs


def load_result(name: str):
    path = RESULTS_DIR / name / "result.pt"
    return torch.load(path, weights_only=False, map_location="cpu") if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-lags", action="store_true",
                    help="BH over all p_max lags; print only, do not write metrics")
    all_lags = ap.parse_args().all_lags
    _, _, A_true, dgp_config = load_toy_data()
    window = dgp_config["ols_window"]

    for base, acle, obs in method_pairs():
        res_base, res_acle = load_result(f"{base}_{obs}"), load_result(f"{acle}_{obs}")
        if res_base is None or res_acle is None:
            print(f"  skip {base}_{obs} / {acle}_{obs}: result.pt missing")
            continue
        if res_acle.get("SE_all") is None:
            print(f"  skip {base}_{obs} / {acle}_{obs}: no SE_all (rerun the benchmark phase)")
            continue
        if res_base["dates"] != res_acle["dates"]:
            raise ValueError(f"{base}_{obs} and {acle}_{obs} have different output days")

        coefs = res_base["coefficients"]
        se = res_acle["SE_all"][-coefs.shape[0]:]  # SE_all also covers the validation days

        for name, res in [(base, res_base), (acle, res_acle)]:
            p_hat = torch.as_tensor(res["p_optimal"])
            if all_lags:
                p_hat = torch.full_like(p_hat, coefs.shape[1])
            bh = compute_bh_edge_metrics(coefs, se, p_hat, res["dates"], A_true, window)
            if all_lags:
                print(f"  {name}_{obs:10s} all lags: FDR={bh['edge_fdr']:.3f}  "
                      f"power={bh['edge_power']:.3f}  F1={bh['edge_f1']:.3f}")
                continue
            metrics_path = RESULTS_DIR / f"{name}_{obs}" / "metrics.json"
            with open(metrics_path) as f:
                metrics = json.load(f)
            metrics.update(bh)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            append_metrics_summary(metrics, RESULTS_DIR)
            print(f"  {name}_{obs:10s} FDR={bh['edge_fdr']:.3f}  power={bh['edge_power']:.3f}  "
                  f"F1={bh['edge_f1']:.3f}")


if __name__ == "__main__":
    main()
