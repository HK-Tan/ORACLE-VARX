#!/usr/bin/env python3
"""Causal-discovery baselines on the toy benchmark: rolling-window PCMCI and VAR-LiNGAM.

Each baseline is refit from scratch on every output day and scored with the same metric
functions as the toy benchmark (scripts/run_toy_benchmark.py), so its numbers sit next to
VAR / VARX / ORACLE-VARX in results-toy/metrics_summary.csv.

Setup, per output day t (T225..T2999, the days the OLS baselines are scored on):
  targets = rows t-200 .. t-1, the ols_window = 200 rows VAR/VARX regress on; each method
            gets exactly the extra past rows it needs for lags up to p_max = 5 (VAR-LiNGAM:
            t-205 .. t-1, like VAR; PCMCI: t-210 .. t-1, because tigramite drops the first
            2 * tau_max rows of a window);
  nodes   = the 3 endogenous series + the confounders observed at this --obs level;
  lags    = 1..p_max, no same-day (lag-0) links scored.

PCMCI (tigramite, ParCorr test, pc_alpha = 0.2, the tigramite default):
  edges    = the MCI p-values of the 3 x 3 x 5 endogenous-to-endogenous links, through the
             same per-day Benjamini-Hochberg step at q = 0.05 as ours (bh_edge_discovery,
             fed z = Phi^{-1}(1 - p/2) with SE = 1, over all 5 lags); an untestable link
             (NaN p-value) counts as p = 1;
  lag p_hat = the largest lag with a discovered endogenous edge (1 if none);
  coefficients and forecast: PCMCI returns a graph, not a model, so each endogenous series
             is refit by OLS (with an intercept) on the target rows, using its discovered
             endogenous parents plus the observed-confounder links PCMCI keeps at its own
             alpha_level = 0.05 (tigramite's default graph); every other coefficient is 0.

VAR-LiNGAM (lingam defaults: lag chosen by BIC within 1..5, DirectLiNGAM on the VAR
residuals, adaptive-lasso pruning). lingam's VAR has no intercept, so each window is centred
by its own column means first (ours fits an intercept):
  edges     = the nonzero pruned lagged coefficients B_k in the endogenous block (fed to the
              same metric as z = 1e6 for nonzero, 0 for zero, so BH keeps exactly those);
  lag p_hat = the BIC lag;
  coefficients = the endogenous block of B_k;
  forecast of Y_t = endogenous rows of mean + (I - B_0)^{-1} sum_k B_k (x_{t-k} - mean).

Metrics are reported on T225..T2999 (the OLS baselines' days) and T425..T2999 (the DML
methods' days). Results go to results-toy/cd-baselines/.

Usage:
    python scripts/run_cd_baselines.py                          # both methods, all 4 levels
    python scripts/run_cd_baselines.py --method pcmci --obs all --n-proc 8
Needs requirements-cd.txt (tigramite, lingam).
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_toy_benchmark as rtb
from src.synthetic.dgp import get_observed_confounders

METHODS = ["pcmci", "varlingam"]
OBS_LEVELS = ["none", "all", "partial_2", "partial_1"]
OUT_DIR = rtb.RESULTS_DIR / "cd-baselines"
SCORE_FROM = {"T225": 225, "T425": 425}  # first scored day: OLS baselines, DML methods

# Filled in each worker by _init (fork or spawn)
_X = None
_CFG = None


def _init(X, cfg):
    global _X, _CFG
    _X, _CFG = X, cfg


def fit_pcmci(t):
    """PCMCI on day t: endogenous MCI p-values (p_max, n, n), OLS-refit coefficients (p_max, n, n)
    and the one-step forecast of Y_t (n,). [k, i, j] = variable j at lag k+1 -> variable i."""
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    n, p_max, window = _CFG["n"], _CFG["p_max"], _CFG["window"]
    win = _X[t - window - 2 * p_max:t]  # tigramite's targets are then rows t-window .. t-1
    res = PCMCI(dataframe=pp.DataFrame(win), cond_ind_test=ParCorr(), verbosity=0).run_pcmci(
        tau_min=1, tau_max=p_max, pc_alpha=0.2)
    pm = np.nan_to_num(res["p_matrix"], nan=1.0)  # [source, target, tau]: X^source_{t-tau} -> X^target_t
    pvals = np.stack([pm[:n, :n, k + 1].T for k in range(p_max)])

    # Endogenous parents from our BH step; confounder parents from tigramite's default graph
    z = torch.from_numpy(norm.isf(np.clip(pvals, 0, 1) / 2))[None]
    edges = rtb.bh_edge_discovery(z, torch.ones_like(z), torch.tensor([p_max]), rtb.BH_EDGE_Q)[0].numpy()
    rows = np.arange(t - window, t)
    coef, fcst = np.zeros((p_max, n, n)), np.zeros(n)
    for i in range(n):
        parents = [(k, j) for k in range(p_max) for j in range(n) if edges[k, i, j]]
        parents += [(k, c) for k in range(p_max) for c in range(n, _X.shape[1])
                    if pm[c, i, k + 1] <= _CFG["alpha_level"]]
        design = np.column_stack([np.ones(window)] + [_X[rows - (k + 1), j] for k, j in parents])
        beta = np.linalg.lstsq(design, _X[rows, i], rcond=None)[0]
        fcst[i] = beta[0] + sum(b * _X[t - (k + 1), j] for b, (k, j) in zip(beta[1:], parents))
        for b, (k, j) in zip(beta[1:], parents):
            if j < n:
                coef[k, i, j] = b
    return pvals, coef, fcst


def fit_varlingam(t):
    """Lagged coefficients (p_max, n, n), BIC lag, and the one-step forecast of Y_t on day t."""
    import lingam

    n, p_max, lookback = _CFG["n"], _CFG["p_max"], _CFG["window"] + _CFG["p_max_offset"]
    win = _X[t - lookback:t]  # rows t-205 .. t-1, the lookback_var rows VAR/VARX use
    mu = win.mean(axis=0)
    model = lingam.VARLiNGAM(lags=p_max, random_state=0).fit(win - mu)
    B = model.adjacency_matrices_  # [0] = B_0, [k] = B_k; B[i, j] = effect of j on i
    lags = B.shape[0] - 1
    coef = np.zeros((p_max, n, n))
    coef[:lags] = B[1:, :n, :n]
    d = win.shape[1]
    x_hat = mu + np.linalg.solve(np.eye(d) - B[0], sum(B[k] @ (_X[t - k] - mu) for k in range(1, lags + 1)))
    return coef, lags, x_hat[:n]


def run(method, obs, n_proc):
    Y_t, W_full, A_true, dgp = rtb.load_toy_data()
    p_max, window = dgp["p_max"], dgp["ols_window"]
    lookback = window + dgp["p_max_offset"]
    first_day = lookback + dgp["validation_days"]
    Y = Y_t.numpy().astype(np.float64)
    W_obs, _ = get_observed_confounders(W_full, obs)
    X = np.hstack([Y, np.asarray(W_obs, dtype=np.float64)])
    n = Y.shape[1]
    days = list(range(first_day, Y.shape[0]))
    cfg = dict(n=n, p_max=p_max, window=window, p_max_offset=dgp["p_max_offset"], alpha_level=0.05)

    print(f"[{method}, obs={obs}] {len(days)} days, {X.shape[1]} nodes, {n_proc} processes")
    t0 = time.perf_counter()
    fit = fit_pcmci if method == "pcmci" else fit_varlingam
    with Pool(n_proc, initializer=_init, initargs=(X, cfg)) as pool:
        out = pool.map(fit, days, chunksize=20)
    wall = time.perf_counter() - t0

    dates = [f"T{t}" for t in days]
    D = len(days)
    coef = torch.from_numpy(np.stack([o[1 if method == "pcmci" else 0] for o in out]))  # (D, p_max, n, n)
    fcst = torch.from_numpy(np.stack([o[2] for o in out]).T)  # (n, D)
    if method == "pcmci":
        pvals = np.stack([o[0] for o in out])  # (D, p_max, n, n)
        z = torch.from_numpy(norm.isf(np.clip(pvals, 0, 1) / 2))
        p_bh = torch.full((D,), p_max)  # BH over all lags, like our p_max fit
        edges = rtb.bh_edge_discovery(z, torch.ones_like(z), p_bh, rtb.BH_EDGE_Q).numpy()
        found = edges.any(axis=(2, 3))  # (D, p_max)
        p_hat = np.where(found.any(axis=1), p_max - np.argmax(found[:, ::-1], axis=1), 1)
    else:
        p_hat = np.array([o[1] for o in out])
        z = torch.where(coef.abs() > 0, torch.full_like(coef, 1e6), torch.zeros_like(coef))
        p_bh = torch.from_numpy(p_hat)

    results = dict(method=method, obs_level=obs, n_days=D, wall_seconds=wall, n_proc=n_proc,
                   mean_p_hat=float(np.mean(p_hat)), scored={})
    for label, lo in SCORE_FROM.items():
        keep = np.array([t >= lo for t in days])
        kept_dates = [d for d, k in zip(dates, keep) if k]
        m = {}
        m.update(rtb.compute_edge_metrics(coef[keep].float(), kept_dates, A_true))
        m.update(rtb.compute_forecast_metrics(fcst[:, keep], Y_t, kept_dates))
        m.update(rtb.compute_lag_metrics(p_hat[keep], kept_dates, A_true))
        m.update(rtb.compute_bh_edge_metrics(z[keep], torch.ones_like(z[keep]), p_bh[keep],
                                             kept_dates, A_true, window))
        results["scored"][label] = {k: v for k, v in m.items() if not k.startswith("regime")}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{method}_{obs}.json").write_text(json.dumps(results, indent=1))
    per_day = dict(days=np.array(days), p_hat=p_hat, z=z.numpy(), coef=coef.numpy(), forecast=fcst.numpy())
    np.savez_compressed(OUT_DIR / f"{method}_{obs}_perday.npz", **per_day)
    print(f"  done in {wall:.1f}s: {json.dumps(results['scored']['T225'])}")
    return results


def write_summary():
    """One row per (method, obs level, scored days): metrics_summary.csv's columns, plus the
    scored days and the mean number of discovered edges per day."""
    cols = ["method", "obs_level", "days", "overall_nonzero_mae", "overall_nonzero_mse", "overall_forecast_mae",
            "overall_forecast_mse", "overall_lag_rmse", "spearman_rho",
            "edge_fdr", "edge_power", "edge_f1", "edge_n_discovered"]
    rows = []
    for f in sorted(OUT_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        for label, m in r["scored"].items():
            rows.append(dict(method={"pcmci": "PCMCI+OLS", "varlingam": "VAR-LiNGAM"}[r["method"]],
                             obs_level=r["obs_level"], days=f"{label}..T2999", **m))
    pd.DataFrame(rows, columns=cols).to_csv(OUT_DIR / "metrics_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS + ["all"], default="all")
    ap.add_argument("--obs", choices=OBS_LEVELS + ["all_levels"], default="all_levels")
    ap.add_argument("--n-proc", type=int, default=rtb.get_physical_cpu_count())
    args = ap.parse_args()
    for method in METHODS if args.method == "all" else [args.method]:
        for obs in OBS_LEVELS if args.obs == "all_levels" else [args.obs]:
            run(method, obs, args.n_proc)
    write_summary()


if __name__ == "__main__":
    main()
