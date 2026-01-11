# ORACLE-VARX

A machine learning framework for financial time series forecasting using Vector Autoregression (VAR) combined with Double Machine Learning (DML) for causal inference.

> **Note:** This repository contains the original implementation using scikit-learn and pandas. A refactored version using cuML and PyTorch is currently in development.

## Overview

This project implements three VAR-based analysis methods:
- **Plain VAR** - Traditional Vector Autoregression
- **OR-VAR** - Orthogonal Regression with Double Machine Learning
- **ORACLE-VAR** - Significance-level-based lag selection

The framework supports configurable ML models (Extra Trees, Random Forest, Lasso, OLS) for outcome and treatment modeling, with parallelized execution for computational efficiency.

## Repository Structure

### `dataset/`
Financial time series data (2000-2020) including VIX, oil prices, interest rates, and other market indicators.

### `old-code/`
Original implementation:

- `oracle_var_experiment.py` - Main experiment script with configurable analysis parameters
- `old-modules/`
  - `VAR_parallelized.py` - Parallelized VAR/VARX estimation
  - `DML_parallelized.py` - Double Machine Learning implementation using EconML
  - `DML_tools.py` - DML utility functions and regressor factory
  - `core_utils.py` - Core utilities for lag creation and data alignment
  - `pnl_calculator.py` - PnL calculation with rolling beta

## Tech Stack

### Current Implementation
- Python
- scikit-learn
- pandas / numpy
- EconML
- statsmodels

### Planned Refactor (Work in Progress)
- **cuML** - GPU-accelerated ML algorithms for faster model training
- **PyTorch** - Deep learning framework for flexible model architectures
- **RAPIDS** - GPU DataFrame operations to replace pandas bottlenecks

The refactor aims to leverage GPU acceleration for significant performance improvements on large-scale financial datasets.
