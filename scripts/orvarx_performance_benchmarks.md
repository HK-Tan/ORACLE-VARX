# OR-VARX Performance Benchmarks

This document consolidates performance benchmarks for both VAR and OR-VARX models, comparing hardware configurations and ML learner choices.

## System Configuration

```
PyTorch version: 2.9.1+cu128
CPU threads available: 11
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
GPU Memory: 8.0 GB
CUDA version: 12.8
```

---

## VAR Model Performance (GPU-Accelerated)

Summary from detailed analysis in `cpu_vs_gpu_analysis_varx.md`.

### CPU vs CUDA with Chunking

| n_days | Output Days | CPU (11 threads) | CUDA (chunk=500) | Speedup |
|--------|-------------|------------------|------------------|---------|
| 900    | 124         | 0.78s            | 0.04s            | ~20x    |
| 2000   | 1224        | 1.65s            | 0.37s            | ~4.5x   |
| 5000   | 4224        | 8.13s            | 0.88s            | **9x**  |

**Key Finding:** CUDA with `chunk_size=500` provides 9x+ speedup over multi-threaded CPU for large datasets. Always use chunked processing for n_days > 2000.

---

## OR-VARX Learner Comparison

Benchmark configuration:
- **Dataset:** n_days=1061, n_assets=5
- **Model:** p_max=3, 27 cross-validation folds
- **Task:** Training nuisance functions (propensity scores, outcome models)

### Training Time by Learner

| Learner     | Training Time | Total Time | Relative Speed |
|-------------|---------------|------------|----------------|
| **lgbm**    | 25.94s        | 30.22s     | 1.0x (fastest) |
| xgboost     | 109.74s       | 111.38s    | 3.7x slower    |
| extra_trees | 86.21s        | 155.10s    | 5.1x slower    |
| rf          | 179.38s       | 249.80s    | 8.3x slower    |

### Analysis

- **LightGBM** is the clear winner for iteration speed, finishing in ~30 seconds
- **XGBoost** provides a middle ground with GPU acceleration available
- **Tree-based methods** (ExtraTrees, RandomForest) are significantly slower due to:
  - No native GPU support
  - Full tree construction overhead
  - Higher memory bandwidth requirements

---

## Recommendations

### For Quick Iteration / Development
Use `lgbm` (LightGBM):
- 5-8x faster than tree-based methods
- Good default hyperparameters
- Fast cross-validation

### For Production / Final Models
Use `lgbm` or `xgboost`:
- Both support GPU acceleration for larger datasets
- Well-tuned implementations
- Extensive hyperparameter options

### When to Use Tree Methods
Consider `extra_trees` or `rf` when:
- Interpretability is critical
- Dataset is small enough that training time is acceptable
- Ensembling with gradient boosting for diversity

---

## Related Documentation

- **Algorithm details:** See `explain_var_and_orvarx.md` for detailed walkthrough of VAR and OR-VARX implementations
- **GPU optimization:** See `cpu_vs_gpu_analysis_varx.md` for detailed CPU vs GPU benchmarks and chunking strategies
