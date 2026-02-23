# TabPFN Computational Complexity & VRAM Scaling

Date: 2026-02-22

## TabPFN Architecture Overview

TabPFN is an in-context learning transformer. It takes training + test data as a single
sequence and produces predictions in one forward pass — no gradient updates at inference.

Key dimensions in our ORACLE-VARX context:

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| n | seq_len = n_train + n_test | 525 (504 + 21) |
| B | effective batch = n_folds_batch × n_outputs | varies (e.g., 228 × 9 = 2052) |
| f | n_features = n_confounders × p | 1–100 |
| d | d_model (hidden dim) | architecture-dependent |
| h | n_heads | architecture-dependent |
| d_h | d / h (per-head dim) | architecture-dependent |
| d_ff | FFN hidden dim (typically 4d) | architecture-dependent |
| L | n_layers | architecture-dependent |

Note: `f = n_confounders × p` is the feature count fed to TabPFN. The 9 assets
contribute to `n_outputs` (batch dimension), not to `f`. If the asset count ever
changes, `n_outputs` changes but `f` stays the same.

## Forward Pass — Step by Step

### Step 1: Input Embedding

Project raw features to hidden dimension:

```
(n, B, f) × (f, d) → (n, B, d)
```

- FLOPs: O(B · n · f · d)
- VRAM: O(B · n · f) for input buffer + O(B · n · d) for output
- Input buffer is proportional to f (linear)

### Step 2: Row Attention — across samples (per layer)

Each sample attends to all other samples. This is standard transformer self-attention
applied along the sequence (sample) dimension:

```
Q, K, V projections:  (n, B, d) × (d, d_h)  → (n, B, d_h)    [×3, per head]
Attention scores:      (B, h, n, d_h) × (B, h, d_h, n) → (B, h, n, n)
Attention output:      (B, h, n, n) × (B, h, n, d_h) → (B, h, n, d_h)
```

- FLOPs: O(B · h · n² · d_h) = O(B · n² · d)
- **VRAM for attention matrix: B × h × n² × 4 bytes**
- Since n = 525 is constant, this is a **fixed cost per fold**

### Step 3: Column Attention — across features (per layer)

TabPFN v2 uses a PerFeatureTransformer with dual attention. Each feature attends
to all other features:

```
Q, K, V projections:  (B, n, f) × (f, d_h)  → (B, n, d_h)    [×3, per head]
Attention scores:      (B·n, h, f, d_h) × (B·n, h, d_h, f) → (B·n, h, f, f)
Attention output:      (B·n, h, f, f) × (B·n, h, f, d_h) → (B·n, h, f, d_h)
```

- FLOPs: O(B · n · h · f² · d_h) = O(B · n · f² · d)
- **VRAM for feature attention matrix: B × n × h × f² × 4 bytes**
- This scales **quadratically** in f

### Step 4: FFN (per layer)

```
(B, n, d) × (d, d_ff) → (B, n, d_ff) → ReLU → (B, n, d_ff) × (d_ff, d) → (B, n, d)
```

- FLOPs: O(B · n · d · d_ff)
- VRAM: O(B · n · d_ff) — constant per fold (n, d, d_ff all fixed)

### Step 5: Output Head

Project to prediction buckets for test samples only:

```
(n_test, B, d) × (d, n_buckets) → (n_test, B, n_buckets)
```

- FLOPs: O(B · n_test · d · n_buckets)
- VRAM: O(B · n_test · n_buckets) — small, constant per fold

## Per-Fold VRAM Model (Theoretical)

Summing all terms that depend on B (since per_fold = total / B):

```
per_fold_gb = C₁ + C₂ · f + C₃ · f²
```

Where:
- **C₁** (constant term): row-attention O(h · n²) + FFN activations O(n · d_ff) + output head
  - Dominated by attention matrix: h × n² = h × 525² ≈ h × 275,625
  - This is the base cost even with 0 features (pure sample-level processing)

- **C₂ · f** (linear term): input/output buffers O(n · f) + embedding matmul intermediates
  - Proportional to moving f-dimensional vectors through n=525 samples

- **C₃ · f²** (quadratic term): column-attention O(n · f²)
  - Feature-to-feature attention matrices
  - Small coefficient because f ≤ 100 (TabPFN limit)

## Empirical Fit

Fitted from 37 probe data points (A6000/A100, n_confounders ∈ {1, 5, 10}):

```
per_fold_gb = 0.0872 + 0.01637 · f + 0.00001635 · f²
```

| Term | At f=5 | At f=10 | At f=50 | At f=70 | At f=100 |
|------|--------|---------|---------|---------|----------|
| C₁ (row-attn, constant) | 0.087 (51%) | 0.087 (35%) | 0.087 (9%) | 0.087 (7%) | 0.087 (5%) |
| C₂·f (input buffers, linear) | 0.082 (48%) | 0.164 (65%) | 0.818 (86%) | 1.146 (87%) | 1.637 (87%) |
| C₃·f² (col-attn, quadratic) | 0.000 (0%) | 0.002 (1%) | 0.041 (4%) | 0.080 (6%) | 0.163 (9%) |
| **Total** | **0.169** | **0.252** | **0.946** | **1.313** | **1.887** |

Goodness of fit:
- R² (linear, 2 params): 0.981
- R² (quadratic, 3 params): 0.996

### Key Takeaway

Over the operational range f ∈ [1, 100], the **linear term dominates** (65–87% of cost).
The quadratic column-attention term is measurable but small (1–9%). A linear model is
a reasonable approximation; a quadratic model is more precise.

### Staircase at Low f

For f = 1, 2, 3 the actual per-fold cost is ~0.093 GB, lower than any smooth model
predicts (~0.10–0.15). This is likely due to CUDA kernel quantization: the smallest
allocation granularity creates a floor. Both the linear and quadratic models overestimate
at low f, which is conservative and safe.

## Cost Breakdown by Operation

For a single forward pass with B effective items:

| Operation | VRAM Scaling | Per-Fold Contribution |
|-----------|-------------|----------------------|
| Row attention matrix | O(B · h · n²) | **C₁** — constant, large |
| QKV projections (row) | O(B · n · d) | C₁ — constant |
| Column attention matrix | O(B · n · h · f²) | **C₃ · f²** — quadratic, small |
| QKV projections (col) | O(B · n · f · d_h) | C₂ · f — linear |
| Input embedding | O(B · n · f · d) | C₂ · f — linear |
| FFN activations | O(B · n · d_ff) | C₁ — constant |
| Output head | O(B · n_test · n_buckets) | C₁ — constant, negligible |

## What This Model Does NOT Capture

### CUDA Memory Fragmentation (solved with `expandable_segments`)

When processing successive p values (p=1, 2, ..., 10), each p creates tensors of
different sizes. PyTorch's default block allocator carves CUDA memory into fixed-size
blocks. After `torch.cuda.empty_cache()`, PyTorch calls `cudaFree()` — but the CUDA
driver retains the freed blocks in fragmented pools. These fragmented pools cannot
serve large contiguous allocations needed by the next p value, so driver-level VRAM
stays high even though torch reports near-zero usage.

This was observed on A100 (80 GB) with macro5 probe **before the fix**:

| After p | Driver VRAM | Torch Reserved | Gap (fragmented pools) |
|---------|-------------|---------------|------------------------|
| p=1 | 36.5 GB | 0.1 GB | **36.4 GB** |
| p=2 | 74.4 GB | 0.1 GB | **74.3 GB** |
| p=3 | 74.4 GB | 0.1 GB | **74.3 GB** |
| p=4 | OOM | — | **73.7 GB** (5.6 GB free) |

The 74 GB "non-torch" memory was **not** cuBLAS workspace — it was fragmented CUDA
allocator pools held by the driver after `cudaFree()`. The key evidence: setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` drops driver VRAM to ~1.4 GB
between p values, and all p=1..10 complete without OOM.

### The Fix: `expandable_segments:True`

The `expandable_segments` allocator (PyTorch 2.1+) grows memory segments in-place
instead of allocating many fixed-size blocks. This prevents fragmentation because:

1. Segments expand contiguously rather than creating new disjoint blocks
2. When freed, the contiguous memory can be fully reclaimed by the driver
3. Different tensor shapes across p values reuse the same expandable segments

The experiment script sets this automatically:
```python
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
```

**After the fix** (A100 80 GB, macro5 probe):

| After p | Driver VRAM | Torch Reserved | Gap |
|---------|-------------|---------------|-----|
| p=1 | 1.4 GB | 0.1 GB | 1.3 GB |
| p=2 | 1.4 GB | 0.1 GB | 1.3 GB |
| ... | 1.4 GB | 0.1 GB | 1.3 GB |
| p=10 | 1.4 GB | 0.1 GB | 1.3 GB |

All p values pass. Driver VRAM drops cleanly between p values.

### Implications

With `expandable_segments` enabled, the batch size formula based on torch utilization
is correct — there is no hidden memory consumer competing for VRAM. The per-fold model
and 65% target work as designed.
