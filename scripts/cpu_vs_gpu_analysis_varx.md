# CPU vs GPU Performance Analysis for VAR Model

## System Configuration

```
PyTorch version: 2.9.1+cu128
CPU threads available: 11
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
GPU Memory: 8.0 GB
GPU SMs (Streaming Multiprocessors): 24
CUDA version: 12.8
```

## Key Findings

### 1. Cold Start Overhead

| Device | Cold Start Time | Cause |
|--------|-----------------|-------|
| CPU | ~0.45s | PyTorch lazy init, BLAS/LAPACK library loading |
| CUDA | ~0.39s | CUDA context creation, kernel compilation |

**Takeaway:** Always do a warmup run before benchmarking to exclude cold start overhead.

---

### 2. Data Transfer Overhead

For typical financial data sizes (5000 days × 10 assets = 200KB), CPU→GPU transfer is **negligible** (~0ms). PCIe bandwidth (~10+ GB/s) handles small tensors instantly.

**Takeaway:** Data transfer is not a bottleneck for this workload.

---

### 3. CUDA vs CPU Performance (Full Batch)

| n_days | Output Days | CPU (11 threads) | CUDA | Speedup |
|--------|-------------|------------------|------|---------|
| 900 | 124 | 0.78s | 0.04s | **18.6x** |
| 1000 | 224 | 0.35s | 0.08s | **4.3x** |
| 2000 | 1224 | 1.65s | 0.37s | **4.5x** |
| 5000 | 4224 | 8.37s | 21.37s | **0.4x** |

**Observation:** CUDA wins for small-medium workloads but loses significantly at n_days=5000.

---

### 4. CPU Thread Scaling (n_days=5000)

| Threads | Time | Speedup |
|---------|------|---------|
| 1 | 21.12s | 1.0x |
| 4 | 7.66s | 2.76x |
| 11 | 8.13s | 2.60x |

**Observation:** Multi-threading provides ~2.6-2.8x speedup. Note that 11 threads is slightly slower than 4 threads, suggesting diminishing returns from hyperthreading.

---

### 5. Chunked Processing on CUDA (Game Changer!)

| Chunk Size | Time | Peak Alloc | vs Full Batch |
|------------|------|------------|---------------|
| Full batch | 12.74s | 4.81 GB | 1.0x |
| 1000 | 0.93s | 1.17 GB | **14x faster** |
| **500** | **0.88s** | 0.61 GB | **14x faster** |
| 100 | 1.28s | 0.16 GB | 10x faster |

**Critical Finding:** Processing in chunks of 500-1000 makes CUDA **14x faster** than full batch!

### With Chunking: Final Comparison

| Device | Time (n_days=5000) |
|--------|-------------------|
| CPU (11 threads) | 8.13s |
| CUDA (full batch) | 12.74s |
| **CUDA (chunk=500)** | **0.88s** |

**CUDA with chunking is 9x faster than multi-threaded CPU!**

---

## Why Does Chunking Help?

The slowdown for large batch sizes comes from **tensor scaling with lag order p** and **cuSOLVER kernel behavior**, not L2 cache effects.

### 1. Tensor and Compute Scaling with p

The VAR(p) model has `n_features = 1 + n_assets × p`. For 10 assets and p=10, that's 101 features. As p increases:

| Component | Scaling | Size at p=10 (batch=4496) |
|-----------|---------|---------------------------|
| `X_batch` | O(p) | ~0.89 GB |
| `lagged_all` | O(p) | ~0.83 GB |
| `XtX`, `XtY` | O(p²) | ~0.17 GB |
| `torch.linalg.solve` | O(p³) compute | - |

This means the p=10 iteration moves **~2GB through VRAM** and does **~10x more compute** than p=1.

### 2. cuSOLVER Kernel Selection Threshold

Profiling shows a sharp slowdown at p≥8 (where `n_features = 81`). This coincides with known kernel-selection thresholds in cuSOLVER's batched LU/solve routines, which often switch strategies around matrix sizes like 64/96/128. When the solver flips from a true batched kernel to a slower fallback path, performance drops dramatically.

### 3. Memory Bandwidth Saturation

Even though total GPU memory isn't full, the **working set** at large batch sizes is orders of magnitude larger than any cache level. The GPU becomes memory-bound, streaming gigabytes of data per iteration. Chunking reduces the working set per call, improving effective memory bandwidth utilization.

### Summary

Chunking helps because:
- Each chunk keeps tensor sizes (and memory traffic) bounded
- Smaller batch sizes may stay below cuSOLVER's kernel-switch thresholds
- Better memory access patterns when working set is smaller

---

## GPU-Dependent Optimal Chunk Size

| GPU Tier | Example GPUs | Suggested Chunk Size |
|----------|--------------|---------------------|
| High-end | RTX 4090, A100, H100 | 2000-4000 |
| Mid-range | RTX 4060, RTX 3070 | 500-1000 |
| Entry-level | GTX 1650, RTX 3050 | 100-250 |

**Factors affecting optimal chunk size:**
- GPU memory bandwidth
- Total VRAM (affects how large tensors can get before pressure)
- cuSOLVER version and kernel thresholds
- `n_assets` and `p_max` (both affect tensor sizes)

**Recommendation:** Auto-tune by testing a few chunk sizes on your hardware. The exact optimum doesn't matter much—anything in the right ballpark (e.g., 500-1000 for mid-range GPUs) gives similar performance.

---

## Recommendations

### For Small Workloads (n_days < 2000)
- Use CUDA with full batch processing
- Expect 4-19x speedup over multi-threaded CPU

### For Large Workloads (n_days > 2000)
- **Always use chunked processing on CUDA**
- Optimal chunk size: 500-1000 for mid-range GPUs
- With chunking, CUDA remains 9x+ faster than CPU

### Production Code Pattern

```python
def fit_var_chunked(Y, p_max, lookback, chunk_size=500):
    """VAR fitting with chunked processing for large datasets."""
    n_days = Y.shape[0]
    n_test_days = n_days - lookback

    if n_test_days <= chunk_size:
        # Small enough for full batch
        return batch_var_all_days(Y, p_max, lookback)

    # Process in chunks
    forecasts_chunks = []
    coef_chunks = []

    for chunk_start in range(0, n_test_days, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_test_days)
        data_start = chunk_start
        data_end = chunk_end + lookback
        Y_chunk = Y[data_start:data_end]

        fc, coef = batch_var_all_days(Y_chunk, p_max, lookback)
        forecasts_chunks.append(fc)
        coef_chunks.append(coef)

    return torch.cat(forecasts_chunks, dim=0), torch.cat(coef_chunks, dim=0)
```

---

## Summary Table

| Scenario | Best Approach | Expected Time (n=5000) |
|----------|---------------|------------------------|
| No GPU | CPU (all threads) | ~8s |
| GPU available | CUDA + chunking | **~0.9s** |
| GPU, naive approach | CUDA full batch | ~13s (avoid!) |

---

## Benchmark Script

Run the full benchmark with:
```bash
python scripts/example_var_usage.py
```

This will output:
1. System info (GPU, CPU threads)
2. Cold start overhead
3. CPU vs CUDA comparison
4. CPU thread scaling
5. Chunked processing test
6. sklearn validation test (verifies PyTorch VAR matches sklearn LinearRegression)
