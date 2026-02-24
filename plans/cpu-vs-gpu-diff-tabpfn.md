# CPU vs GPU BLAS Rounding & Solver Differences — TabPFN Impact Analysis

## Context

ORACLE-VARX-TabPFN performance degraded between commit `9ee2cc0` and HEAD.
Two changes affect the OLS -> BH significance testing -> p-selection pipeline:

1. **OLS solver change**: `torch.linalg.solve` -> `torch.linalg.lstsq` (mild implicit regularization via rcond)
2. **BLAS backend mixing**: Old code computed theta AND SE on GPU (same cuBLAS rounding). New code computes SE on CPU (`torch.linalg.inv` on CPU) while theta comes from GPU — mixing two BLAS backends in the z-stat ratio `z = theta / SE`.

Since BH is a ranking-based procedure, even ~1e-7 differences in z-stats near significance boundaries can flip rejection decisions, cascading through p-selection -> alpha-selection -> forecast selection.

---

## Test A: `torch.linalg.solve` vs `torch.linalg.lstsq` (CPU only)

**Purpose:** Isolate the effect of switching solvers, with identical BLAS backend.

**Configuration:**
- n_assets=9, p_max=10, ols_window=504, seed=42
- All computation on CPU (MKL/OpenBLAS)

### Results (seed=42)

```
<paste output of: python scripts/test_solve_vs_lstsq.py>
```

### Results (seed=123)

```
<paste output of: python scripts/test_solve_vs_lstsq.py --seed 123>
```

---

## Test B: CPU vs GPU BLAS Rounding (same solver)

**Purpose:** Isolate the effect of BLAS backend rounding, using the same solver (`solve`).

**Variants:**
- A: theta=CPU + SE=CPU (all MKL/OpenBLAS)
- B: theta=GPU + SE=GPU (all cuBLAS/MAGMA)
- C: theta=GPU + SE=CPU (mixed — current code path)

**Configuration:**
- n_assets=9, p_max=10, ols_window=504, seed=42
- GPU: <device name>

### Results

```
<paste output of: python scripts/test_cpu_vs_gpu_blas.py>
```

---

## Interpretation

### Which effect dominates?

| Effect | Alpha selection diff | p_optimal diff |
|--------|---------------------|----------------|
| Solver (solve vs lstsq) | __%  | __% |
| BLAS rounding (CPU vs GPU) | __% | __% |
| Backend mixing (GPU+CPU) | __% | __% |

### Analysis

<to be filled after running tests>

---

## Recommendation

<to be filled after running tests>

Potential fixes to evaluate:
1. **Revert solver**: Switch back to `torch.linalg.solve` in `batched_ols`
2. **Unify BLAS backend**: Compute SE on same device as theta
3. **Both**: Revert solver AND unify BLAS backend
4. **Add regularization**: Small diagonal perturbation to XtX before inversion
