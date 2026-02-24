"""Synthetic data generation for toy benchmarks."""

from src.synthetic.dgp import (
    ToyDGPConfig,
    GroundTruth,
    generate_toy_data,
    get_observed_confounders,
)

__all__ = [
    "ToyDGPConfig",
    "GroundTruth",
    "generate_toy_data",
    "get_observed_confounders",
]
