# ORACLE-VARX

A machine learning framework for financial time series forecasting using Vector Autoregression (VAR) combined with Double Machine Learning (DML) for causal inference.

> **Note:** Refactoring complete. The codebase now features PyTorch batched OLS, grid-based memoization (250x speedup), and CPU-native parallelization for tree-based methods.

## Methods

This project implements four VAR-based analysis methods:

- **Plain VAR** - Traditional Vector Autoregression with PyTorch batched OLS
- **OR-VARX** - Orthogonal Regression with Double Machine Learning + grid-based memoization
- **ORACLE-VARX** - Significance-based lag selection with rolling α-selection
- **ACLE-VARX** - VAR with significance-based selection (no DML)

The framework supports configurable ML models (Extra Trees, Random Forest, Lasso, OLS) for outcome and treatment modeling, with parallelized execution for computational efficiency.

## Architecture Highlights

- **CPU for tree-based learners** - Native parallelization via `n_jobs=-1` (scikit-learn)
- **PyTorch batched OLS** - Efficient matrix operations using `torch.bmm` and `torch.linalg.solve`
- **Grid-based memoization** - Reduces model trainings from ~33M to ~131K (250x speedup)
- **Vectorized significance testing** - Benjamini-Hochberg FDR correction for multiple comparisons

## Repository Structure

### `src/models/`
Main model implementations:
- `var_pytorch.py` - Batched VAR estimation
- `dml_pytorch.py` - OR-VARX with memoization
- `oracle_var.py` - ORACLE-VARX with rolling α-selection
- `acle_var.py` - ACLE-VARX (significance-based, no DML)

### `src/modules/`
Shared utilities:
- `grid_config.py` - Lookback configuration
- `model_cache.py` - Fold-level memoization
- `batch_utils.py` - Batched OLS and BH correction
- `factory.py` - Model factory for tree/linear regressors

### `scripts/`
Usage examples demonstrating the methods.

### `dataset/`
Financial time series data (2000-2020) including VIX, oil prices, interest rates, and other market indicators.

### `old-code/`
Original implementation (preserved for reference):
- `oracle_var_experiment.py` - Original experiment script
- `old-modules/` - Original parallelized VAR/DML modules

## Tech Stack

- **PyTorch** - Batched OLS operations
- **scikit-learn** - Tree-based methods (Extra Trees, Random Forest) with CPU parallelization
- **pandas / numpy** - Data manipulation
- **statsmodels** - Statistical tests and diagnostics
