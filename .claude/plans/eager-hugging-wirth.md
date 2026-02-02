# Plan: Adaptive Batch Size with VRAM Monitoring

## Status: ✅ Complete

| Component | Status |
|-----------|--------|
| `_get_vram_usage()` helper | ✅ Complete |
| `_compute_adaptive_batch_size()` function | ✅ Complete |
| Phase 3 adaptive batch integration | ✅ Complete |
| Verbose VRAM logging | ✅ Complete |

## Implementation Summary

Added to `src/models/oracle_var_tabpfn.py`:

1. **VRAM Monitoring**: `_get_vram_usage()` returns (used_gb, total_gb, percent_used)

2. **Adaptive Batch Size**: `_compute_adaptive_batch_size(base_batch_size, p)` scales batch size inversely with p to maintain ~6GB VRAM:
   - p=1 → batch_size=50
   - p=2 → batch_size=25
   - p=10 → batch_size=5

3. **Enhanced Verbose Output**: Shows VRAM and timing per p value:
   ```
   p=1: 23 folds, batch_size=50, VRAM: 2.1/8.0 GB (26%)
   p=1: completed in 4.2s, VRAM: 5.8/8.0 GB (73%), ...
   ```

4. **Total Time**: Final output shows time in seconds and minutes

## Verification

```bash
python scripts/run_oraclevarx_tabpfn_experiment.py --n-days 1500 --verbose
```
