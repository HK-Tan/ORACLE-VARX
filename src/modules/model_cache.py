"""
Model Cache Module for OR-VARX Grid-Based Memoization.

This module provides caching infrastructure for pre-trained models across
different grid points (folds) and lag orders. It enables efficient model
reuse during the rolling window estimation process.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FoldModels:
    """
    Container for trained models and fold boundary information.

    Stores the trained MultiOutputRegressor models for a specific fold
    and lag order, along with the indices defining the train/test split.

    Attributes:
        model_y: Trained MultiOutputRegressor for Y ~ W (outcome model).
        model_t: Trained MultiOutputRegressor for T ~ W (treatment model).
        train_start: Start index of the training window (inclusive).
        train_end: End index of the training window (exclusive).
        test_start: Start index of the test window (inclusive).
        test_end: End index of the test window (exclusive).
        p: Lag order this model was trained for.
    """

    model_y: Any  # MultiOutputRegressor for Y ~ W
    model_t: Any  # MultiOutputRegressor for T ~ W
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    p: int


class ModelCache:
    """
    Cache for storing and retrieving pre-trained fold models.

    The ModelCache stores trained models keyed by grid_idx (which fold)
    and p (lag order). This enables looking up pre-trained models instead
    of retraining, significantly improving performance during grid search
    and rolling window estimation.

    Attributes:
        n_assets: Number of assets in the model.
        n_confounders: Number of confounder variables.
        p_max: Maximum lag order.
        folds: Nested dictionary mapping grid_idx -> p -> FoldModels.
    """

    def __init__(self, n_assets: int, n_confounders: int, p_max: int):
        """
        Initialize the ModelCache.

        Args:
            n_assets: Number of assets in the model.
            n_confounders: Number of confounder variables.
            p_max: Maximum lag order to consider.
        """
        self.n_assets = n_assets
        self.n_confounders = n_confounders
        self.p_max = p_max
        self.folds: Dict[int, Dict[int, FoldModels]] = {}

    def get_fold(self, grid_idx: int, p: int) -> Optional[FoldModels]:
        """
        Retrieve a cached fold model.

        Args:
            grid_idx: The grid/fold index to look up.
            p: The lag order to look up.

        Returns:
            The cached FoldModels if found, None otherwise.
        """
        if grid_idx not in self.folds:
            return None
        return self.folds[grid_idx].get(p)

    def add_fold(self, grid_idx: int, p: int, fold: FoldModels) -> None:
        """
        Add a fold model to the cache.

        Args:
            grid_idx: The grid/fold index for this model.
            p: The lag order for this model.
            fold: The FoldModels instance to cache.
        """
        if grid_idx not in self.folds:
            self.folds[grid_idx] = {}
        self.folds[grid_idx][p] = fold

    def get_all_trained_folds(self, p: int) -> list:
        """
        Get all grid indices that have a trained model for a given lag order.

        Args:
            p: The lag order to look up.

        Returns:
            Sorted list of grid indices that have models trained for this p.
        """
        result = []
        for grid_idx, p_dict in self.folds.items():
            if p in p_dict:
                result.append(grid_idx)
        return sorted(result)
