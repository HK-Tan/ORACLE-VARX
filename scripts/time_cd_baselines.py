#!/usr/bin/env python3
"""Per-window wall-clock of causal-discovery baselines on our rolling windows (timing only).

A rolling-window causal-discovery method refits every window from scratch, so its cost is
(seconds per window) x (number of windows). This script times a few evenly spaced windows.

Datasets:
  toy  6 nodes (3 endogenous + 3 confounders), window 200 rows, p_max = 5
  etf  19 nodes (9 sector ETFs + the 10 all10 confounders), window 504 rows, p_max = 10
Methods:
  parcorr    PCMCI with the linear partial-correlation test (analytic p-values)
  cmiknn     PCMCI with the nonparametric kNN conditional-mutual-information test
             (tigramite defaults: shuffle test, 500 permutations)
  varlingam  VAR-LiNGAM (lingam package) with the lag fixed at p_max (no BIC search)
All variables are nodes; tau_min = 1, tau_max = p_max, tigramite defaults otherwise
(pc_alpha = 0.2, alpha_level = 0.05). Each timed window is the rows before an evenly spaced
end day, T_w raw rows long (tigramite drops the first 2 * p_max rows as lag padding); imports
(and numba compilation for CMIknn) are warmed up outside the timer. This times the discovery
step itself; run_cd_baselines.py adds a BIC lag search (VAR-LiNGAM) or a BH step and OLS refit
(PCMCI) on top.

Usage:
    python scripts/time_cd_baselines.py --dataset toy --method parcorr --n-windows 3
    python scripts/time_cd_baselines.py --dataset etf --method cmiknn --n-windows 1 --max-seconds 3600
Appends one JSON line per window to results/cd-timing/timing.jsonl. With --max-seconds, the
windows run in a child process that is stopped at the cap, and a line
{"timeout_s": ..., "windows_done": ...} records that the cap was hit.
Needs requirements-cd.txt (tigramite, lingam).
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_FILE = Path("results/cd-timing/timing.jsonl")


def load(dataset):
    if dataset == "toy":
        X = np.hstack([pd.read_csv("dataset/toy/Y.csv").values, pd.read_csv("dataset/toy/W.csv").values])
        return X.astype(np.float64), 200, 5
    from src.data.loader import prepare_tensors
    from src.data.constants import ETFS, CONFOUNDER_PRESETS
    Y, W, _, tickers = prepare_tensors(tickers=ETFS + ["SPY"], confounder_names=CONFOUNDER_PRESETS["all10"],
                                       device="cpu")
    keep = [i for i, t in enumerate(tickers) if t != "SPY"]
    return np.hstack([Y.numpy()[:, keep], W.numpy()]).astype(np.float64), 504, 10


def fit_one(method, win, p_max):
    if method == "varlingam":
        import lingam
        lingam.VARLiNGAM(lags=p_max, criterion=None, random_state=0).fit(win)
        return {}
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    if method == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr
        test = ParCorr()
    else:
        from tigramite.independence_tests.cmiknn import CMIknn
        test = CMIknn()
    res = PCMCI(dataframe=pp.DataFrame(win), cond_ind_test=test, verbosity=0).run_pcmci(tau_min=1, tau_max=p_max)
    return {"n_links": int((res["p_matrix"][:, :, 1:] < 0.05).sum())}


def time_windows(dataset, method, n_windows, queue):
    try:
        _time_windows(dataset, method, n_windows, queue)
    except Exception as e:  # report the crash instead of leaving the parent waiting
        queue.put(dict(dataset=dataset, method=method, error=repr(e)))
    queue.put(None)


def _time_windows(dataset, method, n_windows, queue):
    X, T_w, p_max = load(dataset)
    T, N = X.shape
    ends = np.linspace(T_w + 50, T, n_windows).astype(int)  # window = rows [end - T_w, end)
    if method == "cmiknn":  # numba compilation, on a tiny 3-node problem
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.cmiknn import CMIknn
        PCMCI(dataframe=pp.DataFrame(X[:100, :3]), cond_ind_test=CMIknn(sig_samples=20)).run_pcmci(tau_min=1, tau_max=1)
    else:
        fit_one(method, X[:T_w], p_max)
    for end in ends:
        t0 = time.perf_counter()
        extra = fit_one(method, X[end - T_w:end], p_max)
        seconds = time.perf_counter() - t0
        queue.put(dict(dataset=dataset, method=method, N=N, T_w=T_w, p_max=p_max, end=int(end),
                       seconds=seconds, **extra))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["toy", "etf"], required=True)
    ap.add_argument("--method", choices=["parcorr", "cmiknn", "varlingam"], required=True)
    ap.add_argument("--n-windows", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=None, help="stop after this many seconds in total")
    args = ap.parse_args()

    queue = mp.Queue()
    proc = mp.Process(target=time_windows, args=(args.dataset, args.method, args.n_windows, queue))
    proc.start()
    deadline = None if args.max_seconds is None else time.monotonic() + args.max_seconds
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    while True:
        wait = None if deadline is None else max(deadline - time.monotonic(), 0)
        try:
            row = queue.get(timeout=wait)
        except Exception:  # queue.Empty: the cap was hit
            proc.terminate()
            row = dict(dataset=args.dataset, method=args.method, timeout_s=args.max_seconds, windows_done=done)
            print(json.dumps(row), flush=True)
            with OUT_FILE.open("a") as f:
                f.write(json.dumps(row) + "\n")
            break
        if row is None:
            break
        done += 1
        print(json.dumps(row), flush=True)
        with OUT_FILE.open("a") as f:
            f.write(json.dumps(row) + "\n")
    proc.join()


if __name__ == "__main__":
    main()
