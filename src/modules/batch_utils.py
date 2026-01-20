"""Shared batch utilities for VAR and OR-VARX models.

This module provides common batched operations used by both plain VAR
and orthogonalized OR-VARX models.

Key functions:
- batched_ols: Batched ordinary least squares using torch.linalg.solve
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

    if rcond is not None:
        # Use lstsq for regularization (slower but more robust)
        # lstsq returns (solution, residuals, rank, singular_values)
        # We need to iterate over batch since lstsq doesn't support batched input well
        batch_size = X_batch.shape[0]
        n_features = X_batch.shape[2]
        n_targets = Y_batch.shape[2]
        device = X_batch.device
        dtype = X_batch.dtype

        beta = torch.zeros(batch_size, n_features, n_targets, device=device, dtype=dtype)
        for i in range(batch_size):
            solution = torch.linalg.lstsq(XtX[i], XtY[i], rcond=rcond)
            beta[i] = solution.solution
        return beta

    # Use solve for speed (assumes non-singular X'X)
    try:
        beta = torch.linalg.solve(XtX, XtY)  # (batch, n_features, n_targets)
    except torch.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Singular matrix encountered in batched OLS. "
            f"This may indicate multicollinearity or insufficient data. "
            f"Consider using rcond parameter for regularization or checking input data."
        ) from e

    return beta
