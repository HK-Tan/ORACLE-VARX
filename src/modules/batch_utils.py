"""Shared batch utilities for VAR and OR-VARX models.

This module provides common batched operations used by both plain VAR
and orthogonalized OR-VARX models.

Key functions:
- batched_ols: Batched ordinary least squares using torch.linalg.solve
- batched_benjamini_hochberg: Vectorized Benjamini-Hochberg FDR correction
"""

import torch
from typing import Optional


def batched_ols(
    X_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    rcond: Optional[float] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Batched OLS regression: beta = (X'X)^{-1} X'Y

    Solves the normal equations for multiple regression problems in parallel
    using batched matrix operations. This is the core computation shared by
    both VAR and OR-VARX models.

    For each batch element i:
        beta[i] = argmin_b ||Y[i] - X[i] @ b||^2
                = (X[i]'X[i])^{-1} X[i]'Y[i]

    Args:
        X_batch: Design matrices, shape (batch, n_samples, n_features)
        Y_batch: Response matrices, shape (batch, n_samples, n_targets)
        rcond: Reciprocal condition number cutoff for regularization.
               If None, uses torch.linalg.solve (faster, no regularization).
               If set, uses torch.linalg.lstsq (slower, more robust).
        chunk_size: If set and batch > chunk_size, process in chunks to manage
                   GPU memory. Default None means no chunking.

    Returns:
        beta: Coefficient matrices, shape (batch, n_features, n_targets)
              beta[i, j, k] is the coefficient of feature j for target k in batch i

    Raises:
        RuntimeError: If OLS solve fails (singular matrix and rcond=None)
        ValueError: If input shapes are incompatible

    Example:
        >>> # Solve 100 regression problems in parallel
        >>> X = torch.randn(100, 50, 10)  # 100 problems, 50 samples, 10 features
        >>> Y = torch.randn(100, 50, 5)   # 100 problems, 50 samples, 5 targets
        >>> beta = batched_ols(X, Y)
        >>> print(beta.shape)  # (100, 10, 5)

    Notes:
        - Uses torch.bmm for batched matrix multiplication (GPU-accelerated)
        - Uses torch.linalg.solve for numerical stability (Cholesky-based)
        - For ill-conditioned problems, set rcond to a small positive value
        - Use chunk_size for large batches (>1000) to avoid GPU OOM
    """
    batch_size = X_batch.shape[0]

    # Chunking logic for large batches (GPU memory management)
    if chunk_size is not None and batch_size > chunk_size:
        results = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk_result = batched_ols(
                X_batch[start:end],
                Y_batch[start:end],
                rcond=rcond,
                chunk_size=None,  # Don't recurse further
            )
            results.append(chunk_result)
        return torch.cat(results, dim=0)

    # Validate input shapes
    if X_batch.dim() != 3:
        raise ValueError(f"X_batch must be 3D (batch, samples, features), got shape {X_batch.shape}")
    if Y_batch.dim() != 3:
        raise ValueError(f"Y_batch must be 3D (batch, samples, targets), got shape {Y_batch.shape}")
    if X_batch.shape[0] != Y_batch.shape[0]:
        raise ValueError(
            f"Batch sizes must match: X_batch has {X_batch.shape[0]}, Y_batch has {Y_batch.shape[0]}"
        )
    if X_batch.shape[1] != Y_batch.shape[1]:
        raise ValueError(
            f"Sample counts must match: X_batch has {X_batch.shape[1]}, Y_batch has {Y_batch.shape[1]}"
        )

    # Compute X'X and X'Y using batched matrix multiplication
    # X_batch: (batch, n_samples, n_features)
    # X_batch.transpose(1, 2): (batch, n_features, n_samples)
    XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)  # (batch, n_features, n_features)
    XtY = torch.bmm(X_batch.transpose(1, 2), Y_batch)  # (batch, n_features, n_targets)

    # Use lstsq: batched, handles singular matrices gracefully.
    # Default rcond=None lets PyTorch use eps * max(m, n), which adapts
    # to dtype and matrix size — safer than a hardcoded threshold.
    return torch.linalg.lstsq(XtX, XtY, rcond=rcond).solution


def batched_benjamini_hochberg(p_values: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Vectorized Benjamini-Hochberg FDR correction.

    Args:
        p_values: shape (batch_size, n_tests) or (n_tests,)
        alpha: FDR level

    Returns:
        reject: boolean tensor, same shape as p_values
    """
    if p_values.dim() == 1:
        p_values = p_values.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    batch_size, n_tests = p_values.shape
    device = p_values.device

    # Sort p-values
    sorted_pvals, sorted_idx = torch.sort(p_values, dim=1)

    # ECDF factor: [1/m, 2/m, ..., 1]
    ecdf = torch.arange(1, n_tests + 1, device=device, dtype=p_values.dtype) / n_tests

    # Critical values: (i/m) * alpha
    thresholds = ecdf * alpha

    # Find largest i where p_(i) <= threshold_i
    reject_sorted = sorted_pvals <= thresholds

    # Expand rejection: if any position i is rejected, all positions <= i are rejected
    # Use cummax to propagate rejection forward, then find max rejection index
    reject_cummax = reject_sorted.cummax(dim=1).values

    # But we need the LARGEST i where p_(i) <= threshold_i, then reject all <= i
    # Find the rightmost True in reject_sorted
    has_any_rejection = reject_sorted.any(dim=1, keepdim=True)

    # Create mask where positions <= max_reject_idx are True
    max_reject_idx = torch.where(
        reject_sorted,
        torch.arange(n_tests, device=device).unsqueeze(0).expand(batch_size, -1),
        torch.zeros(batch_size, n_tests, device=device, dtype=torch.long)
    ).max(dim=1, keepdim=True).values

    positions = torch.arange(n_tests, device=device).unsqueeze(0)
    reject_expanded = (positions <= max_reject_idx) & has_any_rejection

    # Restore original order using scatter
    reject = torch.zeros_like(p_values, dtype=torch.bool)
    reject.scatter_(1, sorted_idx, reject_expanded)

    if squeeze:
        reject = reject.squeeze(0)

    return reject
