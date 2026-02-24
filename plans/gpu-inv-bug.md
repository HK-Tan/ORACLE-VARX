# GPU `torch.linalg.inv` Bug — Phase 4 OLS

## Summary

`torch.linalg.inv` crashes with `cudaErrorIllegalAddress` when run on GPU for
batch sizes typical in Phase 4 (e.g., 4272 × 18×18 matrices at p=2). This is a
CUDA kernel bug in MAGMA's batched inverse, not a context contamination issue.

## What crashes

- `torch.linalg.inv(TtT)` on GPU, where `TtT` has shape `(~4272, 18, 18)` (p=2)
- Confirmed with `CUDA_LAUNCH_BLOCKING=1` — the crash is deterministic

## What works fine on GPU

- `batched_ols()` (uses `torch.linalg.lstsq`, not `inv`)
- `torch.bmm(T_windows.T, T_windows)` — the XtX computation itself
- Smaller batch sizes or matrix sizes may work, but the threshold is unpredictable

## Root cause

MAGMA's batched `getrf`/`getri` kernels (used by `torch.linalg.inv`) have known
issues with certain batch × matrix-size combinations on some GPU architectures.
This is a CUDA/MAGMA kernel bug, not a PyTorch or user code issue.

## Fix

Move `torch.linalg.inv` to CPU. The batched OLS (`lstsq`) remains on GPU for
speed. Only the SE computation (which needs `inv`) is done on CPU:

1. `batched_ols(..., return_XtX=True)` returns XtX from GPU without recomputing
2. Move XtX to CPU: `TtT_cpu = TtT.cpu()`
3. `torch.linalg.inv(TtT_cpu)` — works reliably on CPU
4. Rest of SE computation (diagonal extraction, sigma_sq, sqrt) on CPU

Performance impact: negligible — `inv` on 4272 × 18×18 CPU matrices takes <1s,
vs. the 60-90s spent in `batched_ols` on GPU.

## Affected files

- `src/modules/batch_utils.py` — added `return_XtX` parameter to `batched_ols()`
- `src/models/oracle_var_tabpfn.py` — `_run_ols_phase4()` does `inv` on CPU
