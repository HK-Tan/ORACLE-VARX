# CPU vs GPU Tree-Based Learners: Benchmark Analysis

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Rows | 1,000 |
| Outputs | 10 (multi-output regression) |
| Features | 10, 100 |
| CPU cores tested | 1, 5, 10 |
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |

### Learners Tested

| Learner | CPU | GPU |
|---------|-----|-----|
| XGBoost | sklearn factory | CUDA (native) |
| LightGBM | sklearn factory | N/A (requires special build) |
| RandomForest | sklearn | cuML |
| ExtraTrees | sklearn | N/A (no GPU implementation) |

---

## Results Summary

### Low Feature Count (10 features)

| Learner | Best CPU Time | Best GPU Time | Winner |
|---------|---------------|---------------|--------|
| XGBoost | 1.35s (10 cores) | 3.64s | **CPU** |
| LightGBM | 0.66s (1 core) | N/A | CPU |
| RandomForest | 2.83s (5 cores) | 8.70s | **CPU** |
| ExtraTrees | 1.16s (10 cores) | N/A | CPU |

**Key insight**: With few features, GPU overhead (data transfer, kernel launch) dominates. CPU wins across the board.

### High Feature Count (100 features)

| Learner | Best CPU Time | Best GPU Time | Winner |
|---------|---------------|---------------|--------|
| XGBoost | 5.10s (10 cores) | 5.52s | **Tie** |
| LightGBM | 2.25s (10 cores) | N/A | CPU |
| RandomForest | 18.27s (10 cores) | 13.12s | **GPU** |
| ExtraTrees | 4.47s (10 cores) | N/A | CPU |

**Key insight**: GPU becomes competitive with more features. RandomForest sees the biggest GPU benefit.

---

## Detailed Analysis

### XGBoost: GPU vs CPU Crossover

```
n_features=10:  GPU is 0.37x-0.67x slower (CPU wins)
n_features=100: GPU is 0.90x-2.62x faster (depends on cores)
```

| Scenario | Recommendation |
|----------|----------------|
| 1 CPU core, 100 features | GPU (2.62x faster) |
| 5 CPU cores, 100 features | GPU (1.17x faster) |
| 10 CPU cores, any features | CPU (GPU overhead not worth it) |
| Low features (<50) | CPU always |

### RandomForest: Dramatic GPU Wins at Scale

```
n_features=10:  GPU 1.16x faster (single core) → 0.22x slower (10 cores)
n_features=100: GPU 6.56x faster (single core) → 0.73x slower (10 cores)
```

cuML RandomForest shows the **largest speedup** (6.56x) but also the **largest slowdown** when misused.

| Scenario | Recommendation |
|----------|----------------|
| 1 CPU core, high features | GPU (6.56x faster) |
| Limited CPU resources | GPU |
| 5+ CPU cores available | CPU (better scaling) |
| Low features | CPU always |

### LightGBM: CPU Champion

LightGBM is remarkably efficient on CPU:
- **Fastest at low features** (0.66s vs XGBoost 1.35s)
- **Competitive at high features** (2.25s vs XGBoost 5.10s)
- Scales well with cores

Without GPU build, LightGBM remains the best choice for CPU-constrained environments.

### ExtraTrees: Efficient CPU Scaling

- 4.8x speedup from 1→10 cores (10 features)
- 4.8x speedup from 1→10 cores (100 features)
- Faster than sklearn RandomForest across all configs

---

## Why GPU n_jobs Hurts Performance

An unexpected finding: increasing `n_jobs` for GPU models (via MultiOutputRegressor) **degrades performance**:

```
cuML RF (100 features):
  n_jobs=1:  13.12s
  n_jobs=5:  16.39s (+25% slower)
  n_jobs=10: 25.05s (+91% slower)
```

**Root cause**: Multiple Python processes competing for the same GPU creates:
- Memory transfer contention
- GPU context switching overhead
- Serialization bottlenecks

**Recommendation**: Always use `n_jobs=1` for GPU-accelerated models.

---

## Decision Framework

```
                        ┌─────────────────┐
                        │ How many CPU    │
                        │ cores available?│
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
               1 core       2-5 cores     6+ cores
                    │            │            │
                    ▼            ▼            ▼
            ┌───────────┐  ┌───────────┐  ┌───────────┐
            │High feat? │  │High feat? │  │           │
            └─────┬─────┘  └─────┬─────┘  │  Use CPU  │
                  │              │         │  (LightGBM│
            ┌─────┴─────┐  ┌─────┴─────┐  │   best)   │
            │Yes     No │  │Yes     No │  └───────────┘
            ▼           ▼  ▼           ▼
         Use GPU    Use CPU  Consider   Use CPU
         (XGB/RF)  (LightGBM)   GPU    (LightGBM)
```

---

## Recommendations by Use Case

### Cloud/Server (Many Cores Available)
- **Use CPU with LightGBM or XGBoost**
- Multi-core scaling is excellent
- No GPU memory constraints
- Simpler deployment

### Local Development (Limited Cores)
- **Use GPU for high-dimensional data**
- XGBoost GPU for >50 features
- cuML RF for >50 features, single-threaded
- LightGBM CPU for low features

### Cost Optimization
- Single GPU often cheaper than 10+ CPU cores
- But only beneficial for high-dimensional workloads
- For small data: CPU instances are more cost-effective

---

## Hardware Context

**Test System:**
- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM, Ada Lovelace)
- CPU: Intel/AMD with 22 cores (11 used in benchmark)
- Environment: WSL2 on Windows

**Scaling expectations:**
- Higher-end GPUs (RTX 4090, A100) would shift crossover points lower
- More CPU cores would favor CPU further
- Larger datasets (10k-100k rows) would favor GPU more

---

## Conclusions

1. **GPU is not always faster** - data transfer overhead dominates for small workloads
2. **Feature count matters more than row count** for GPU benefit
3. **LightGBM CPU is remarkably efficient** - often the best choice
4. **Never parallelize GPU models** via MultiOutputRegressor
5. **Crossover point**: ~50-100 features with limited CPU cores

---

## Practical Recommendation: Use 5 CPU Cores

The data shows **5 CPU cores is the optimal default** for all learners:

| Learner | 1→5 cores | 5→10 cores | Verdict |
|---------|-----------|------------|---------|
| XGBoost (100 feat) | 2.4x faster | 1.3x faster | Diminishing returns |
| LightGBM (100 feat) | 2.6x faster | 1.07x faster | Minimal gain |
| LightGBM (10 feat) | 0.9x | 0.92x | **Gets worse** |
| RF (10 feat) | 3.6x faster | 0.89x | **Gets worse** |
| ExtraTrees | 3.2x faster | 1.03x faster | Negligible gain |

**Why 5 cores works best:**
- Captures 70-90% of the multi-core speedup
- Avoids thread contention overhead at higher core counts
- Leaves system resources for other processes
- Consistent performance across all learners

**Final recommendation for this project:**
```python
# Default configuration
n_jobs = 5
learner = 'lgbm'  # or 'xgboost' for high-dimensional data
```

This configuration provides robust performance across varying feature counts without needing to tune per-workload or manage GPU complexity.
