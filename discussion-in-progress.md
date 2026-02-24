# Phase 4 GPU OLS Crash — Discussion in Progress

## The Problem

Phase 4 batched OLS crashes with `cudaErrorIllegalAddress` at p=2, even in a
spawned subprocess (`mp.get_context('spawn')`) with a fresh CUDA context.

```
p=2: OLS failed (CUDA error: an illegal memory access was encountered)
p=3: OLS failed (CUDA error: an illegal memory access was encountered)
...
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

## Root Cause Analysis

CUDA's own error message says:
> "CUDA kernel errors might be asynchronously reported at some other API call,
> so the stacktrace below might be incorrect."

This means the error likely originates in **p=1's computation** but is only
reported when p=2 runs. A CUDA kernel (probably `torch.linalg.inv` or
`torch.linalg.lstsq`) writes out of bounds during p=1. The corrupted memory
goes unnoticed until p=2 reuses the same GPU memory region (via PyTorch's
caching allocator), at which point CUDA detects the illegal access.

**Key evidence:**
- Fresh CUDA context (spawn subprocess) — so this is NOT Phase 3 contamination
- p=1 completes "successfully" but likely corrupts memory silently
- p=2 crashes because it touches the corrupted region
- The crash is in the CUDA kernel itself, not a Python-level issue

## What the Code Does (per p)

**File:** `src/models/oracle_var_tabpfn.py`, `_run_ols_phase4()` (lines 213-262)

```python
for p in range(1, p_max + 1):
    # 1. Build sliding windows on GPU
    T_windows = R_T.unfold(0, ols_window, 1).transpose(1, 2).contiguous()
    Y_windows = R_Y.unfold(0, ols_window, 1).transpose(1, 2).contiguous()
    # T_windows: (4271, 504, n_assets*p), Y_windows: (4271, 504, n_assets)

    # 2. Batched OLS on GPU — calls batched_ols() from batch_utils.py
    theta_batch = batched_ols(T_windows, Y_windows, chunk_size=batch_chunk_size)
    #   internally:  XtX = bmm(X.T, X)    ← ~35 GFLOPS at p=10
    #                XtY = bmm(X.T, Y)    ← ~3.5 GFLOPS
    #                lstsq(XtX, XtY)      ← ~1.7 GFLOPS

    # 3. SE computation on GPU — CRASHES here
    TtT = torch.bmm(T_windows.transpose(1, 2), T_windows)  # redundant! same as XtX above
    TtT_inv = torch.linalg.inv(TtT)                        # batched inv of 4271 matrices
    # ... rest of SE formula ...
```

## Why VAR Works but Phase 4 Doesn't

`var_pytorch.py` (`fit_var_batched`) uses the **same `batched_ols()`** on GPU
without issues — because it **never computes SEs**. No `torch.linalg.inv`.

## FLOP Comparison: OLS vs SE (at p=10)

Both are dominated by the same `bmm(T.T, T)` operation:

| Operation                  | FLOPs     | Notes                            |
|----------------------------|-----------|----------------------------------|
| **OLS: bmm(X.T, X)**      | ~35 GFLOPS | Inside batched_ols               |
| **OLS: bmm(X.T, Y)**      | ~3.5 GFLOPS|                                  |
| **OLS: lstsq(XtX, XtY)**  | ~1.7 GFLOPS|                                  |
| **OLS total**              | **~40 GFLOPS** |                             |
| **SE: bmm(T.T, T)**       | ~35 GFLOPS | REDUNDANT — same as XtX above!   |
| **SE: inv(TtT)**          | ~3.1 GFLOPS| The crash point                  |
| **SE: bmm(T, theta)**     | ~3.5 GFLOPS|                                  |
| **SE total**               | **~42 GFLOPS** |                             |

SE is roughly the **same cost** as OLS, not cheaper. Moving SE to CPU would
approximately double the total compute time per p value. The redundant
`bmm(T.T, T)` in the SE section recomputes what `batched_ols()` already computed
internally but doesn't return.

## SE Is Required

SE is used in Phase 6 (lines 1017-1040) for significance testing:
```python
z_new = torch.abs(theta_new / (SE_new + 1e-10))
p_vals = 2 * (1 - normal_dist.cdf(z_new))
reject = batched_benjamini_hochberg(p_vals_flat, alpha)
```
This drives the alpha/p selection. Cannot be dropped.

## Options Under Discussion

### Option A: Return XtX from batched_ols, compute SE inv on CPU

Modify `batched_ols()` to optionally return the already-computed XtX.
Then SE only needs `inv(XtX)` on CPU (~3 GFLOPS) — eliminates the redundant
35 GFLOPS bmm. Makes SE ~10x cheaper than OLS.

```python
# In batch_utils.py:
def batched_ols(X_batch, Y_batch, ..., return_XtX=False):
    XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)
    XtY = torch.bmm(X_batch.transpose(1, 2), Y_batch)
    beta = torch.linalg.lstsq(XtX, XtY, rcond=rcond).solution
    if return_XtX:
        return beta, XtX
    return beta

# In oracle_var_tabpfn.py:
theta_batch, TtT = batched_ols(T_windows, Y_windows, ..., return_XtX=True)
TtT_inv = torch.linalg.inv(TtT.cpu())  # only ~3 GFLOPS on CPU
# ... rest of SE on CPU (cheap without the redundant bmm)
```

### Option B: torch.linalg.solve on GPU instead of inv

Replace `inv(TtT)` with `solve(TtT, I)` — uses a different CUDA kernel path
that might not have the same bug. Everything stays on GPU.

```python
I = torch.eye(TtT.shape[-1], device=TtT.device, dtype=TtT.dtype)
I = I.expand_as(TtT)
TtT_inv = torch.linalg.solve(TtT, I)
```

### Option C: Chunk the GPU inv into smaller batches

Instead of `inv()` on all 4271 matrices at once, chunk into batches of e.g. 500.
The kernel bug may only trigger at large batch sizes.

```python
chunk = 500
TtT_inv_chunks = []
for i in range(0, TtT.shape[0], chunk):
    TtT_inv_chunks.append(torch.linalg.inv(TtT[i:i+chunk]))
TtT_inv = torch.cat(TtT_inv_chunks, dim=0)
```

## Open Questions

1. Would the crash reproduce on a cold GPU (no prior TabPFN)? A standalone test
   would confirm whether it's a PyTorch/CUDA kernel bug vs. some spawn isolation issue.
2. Which option (A/B/C) balances safety, performance, and code simplicity best?
3. Should `dml_pytorch.py` (lines 753-754) get the same fix? It has the identical
   batched `inv()` pattern but runs with XGBoost (no TabPFN VRAM thrashing), so
   it hasn't crashed yet.
