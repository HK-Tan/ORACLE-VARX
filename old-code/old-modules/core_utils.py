# import warnings
# warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
# Only import estimators that natively support multi-output regression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, Lars, LassoLars
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor


def make_lags(df, p, include_original=False):
    """
    Create lagged copies of a DataFrame with enhanced flexibility.

    This unified function consolidates the make_lags functionality from VAR_tools.py,
    DML_tools.py, and DML_parallelized.py with consistent behavior and additional options.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to create lags from.
    p : int
        The number of lags to create (must be a positive integer).
    include_original : bool, default=False
        If True, includes the original columns (lag 0) in the output.
        If False, only includes lagged columns starting from lag 1.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with lagged columns. Column names include suffix '_lag{k}'
        where k is the lag number.

    Raises:
    -------
    ValueError
        If p is not a positive integer.

    Examples:
    ---------
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    >>> make_lags(df, 2)  # Returns lags 1 and 2
    >>> make_lags(df, 2, include_original=True)  # Returns original, lag 1, and lag 2
    """
    if not isinstance(p, int) or p <= 0:
        raise ValueError(f"Value of p for computing lags must be a positive integer, actual value is p={p}")

    lagged_dfs = []

    # Include original if requested (equivalent to make_lags_with_original from DML_tools)
    if include_original:
        lagged_dfs.append(df)

    # Create lagged versions starting from lag 1
    for k in range(1, p + 1):
        lagged_dfs.append(df.shift(k).add_suffix(f'_lag{k}'))

    return pd.concat(lagged_dfs, axis=1)


def realign(*dfs):
    """
    Realign multiple DataFrames by dropping rows with NaN values.

    This unified function consolidates the realign functionality from VAR_tools.py,
    DML_tools.py, and DML_parallelized.py with enhanced flexibility to handle
    variable number of DataFrames.

    Parameters:
    -----------
    *dfs : variable number of pd.DataFrame
        Variable number of DataFrames to realign. Handles the common cases:
        - 2 DataFrames: (Y, X) for VAR methods
        - 3 DataFrames: (Y, T, W) for DML methods
        - Any number of DataFrames for general use

    Returns:
    --------
    tuple
        Tuple of realigned DataFrames in the same order as input, with all
        NaN rows removed consistently across all DataFrames.

    Examples:
    ---------
    >>> Y, X = realign(Y_df, X_df)  # VAR case
    >>> Y, T, W = realign(Y_df, T_df, W_df)  # DML case
    >>> df1, df2, df3, df4 = realign(df1, df2, df3, df4)  # General case
    """
    if len(dfs) == 0:
        raise ValueError("At least one DataFrame must be provided")

    # Concatenate all DataFrames and drop NaN rows
    full = pd.concat(dfs, axis=1).dropna()

    # Split back into original DataFrames based on column counts
    result = []
    col_start = 0

    for df in dfs:
        col_end = col_start + len(df.columns)
        result.append(full.iloc[:, col_start:col_end])
        col_start = col_end

    return tuple(result)


def get_regressor(regressor_name, **kwargs):
    """
    Factory function to create ML regressors that NATIVELY support multi-output regression.

    This function only includes estimators verified to handle multiple outputs without
    requiring MultiOutputRegressor wrapper, based on scikit-learn documentation.

    Parameters:
    -----------
    regressor_name : str
        Name of the regressor to create. Supported options (all native multi-output):

        **Tree-based methods:**
        - 'decision_tree': DecisionTreeRegressor
        - 'extra_trees': ExtraTreesRegressor
        - 'random_forest': RandomForestRegressor

        **Linear methods:**
        - 'ols': Ordinary Least Squares (LinearRegression)
        - 'lasso': Lasso regression
        - 'ridge': Ridge regression
        - 'elastic_net': ElasticNet regression
        - 'lars': Least Angle Regression
        - 'lasso_lars': LASSO with LARS algorithm

        **Nearest neighbors:**
        - 'knn': K-Nearest Neighbors regressor
        - 'radius_neighbors': Radius-based neighbors regressor

        **Other methods:**
        - 'gaussian_process': Gaussian Process regressor
        - 'pls': Partial Least Squares regression
        - 'dummy': Dummy regressor for baselines

    **kwargs : dict
        Additional parameters to pass to the regressor constructor.

        **Common parameters for tree-based methods:**
        - n_estimators: Number of trees (200 for forests, optimized for performance)
        - max_depth: Maximum tree depth (10, prevents overfitting and theoretically motivated)
        - min_samples_split: Minimum samples required to split (20, better generalization)
        - min_samples_leaf: Minimum samples at leaf (10, prevents overfitting)
        - max_features: Fraction of features to consider (0.8, adds randomness)
        - random_state: Random seed (0, for reproducibility)
        - ccp_alpha: Cost complexity pruning (0.01, prevents overfitting)

        **Common parameters for linear methods:**
        - alpha: Regularization parameter for penalized methods (default: 1.0)
        - max_iter: Maximum iterations (default: 1000)
        - fit_intercept: Whether to fit intercept (default: True)

    Returns:
    --------
    sklearn estimator
        Configured regressor instance that natively handles multiple outputs.

    Raises:
    -------
    KeyError
        If regressor_name is not supported.

    Notes:
    ------
    All included regressors are verified from scikit-learn documentation to support
    native multi-output regression without requiring MultiOutputRegressor wrapper.

    Examples:
    ---------
    >>> # Tree-based methods with improved defaults
    >>> regressor = get_regressor('extra_trees')  # Uses n_estimators=200, max_depth=15
    >>> regressor = get_regressor('random_forest', max_depth=20)  # Override specific params
    >>> regressor = get_regressor('decision_tree', ccp_alpha=0.05)  # More aggressive pruning
    >>>
    >>> # Linear methods
    >>> regressor = get_regressor('lasso', alpha=0.1, max_iter=2000)
    >>> regressor = get_regressor('ridge', alpha=10.0)
    >>> regressor = get_regressor('elastic_net', alpha=0.5, l1_ratio=0.7)
    """

    # Only regressors verified to natively support multi-output from scikit-learn docs
    regressor_configs = {
        # Tree-based methods (verified native multi-output support)
        'decision_tree': {
            'class': DecisionTreeRegressor,
            'default_params': {
                'max_depth': 10,              # Limit depth to prevent overfitting
                'min_samples_split': 20,       # Higher value for smoother model
                'min_samples_leaf': 10,        # Prevent low-variance leaf nodes
                'min_impurity_decrease': 0.01, # Only split if improvement > threshold
                'random_state': 0,
                'ccp_alpha': 0.01             # Minimal cost complexity pruning
            }
        },
        'extra_trees': {
            'class': ExtraTreesRegressor,
            'default_params': {
                'n_estimators': 200,           # More trees for stability
                'max_depth': 5,               # Limit depth to prevent overfitting
                'min_samples_split': 20,       # Better generalization
                'min_samples_leaf': 10,        # Prevent overfitting
                'max_features': 0.8,           # Consider subset of features
                'bootstrap': False,            # Default for Extra Trees
                'random_state': 0,
                'n_jobs': -1
            }
        },
        'random_forest': {
            'class': RandomForestRegressor,
            'default_params': {
                'n_estimators': 200,           # More trees often improve performance
                'max_depth': 5,               # Limit depth to prevent overfitting
                'min_samples_split': 20,       # Higher for better generalization
                'min_samples_leaf': 10,        # Prevent overfitting
                'max_features': 0.8,           # Consider subset of features
                'bootstrap': True,             # Default for Random Forest
                'oob_score': False,             # Get out-of-bag score for validation
                'random_state': 0,
                'n_jobs': -1
            }
        },

        # Linear methods (verified native multi-output support)
        'ols': {
            'class': LinearRegression,
            'default_params': {
                'fit_intercept': True,
                'n_jobs': -1
            }
        },
        'lasso': {
            'class': Lasso,
            'default_params': {
                'alpha': 0.001,
                'max_iter': 1000,
                'fit_intercept': True,
                'random_state': 0
            }
        },
        'ridge': {
            'class': Ridge,
            'default_params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'random_state': 0
            }
        },
        'elastic_net': {
            'class': ElasticNet,
            'default_params': {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'max_iter': 1000,
                'fit_intercept': True,
                'random_state': 0
            }
        },
        'lars': {
            'class': Lars,
            'default_params': {
                'fit_intercept': True,
                'normalize': False
            }
        },
        'lasso_lars': {
            'class': LassoLars,
            'default_params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'normalize': False
            }
        },

        # Nearest neighbors (verified native multi-output support)
        'knn': {
            'class': KNeighborsRegressor,
            'default_params': {
                'n_neighbors': 5,
                'weights': 'uniform',
                'n_jobs': -1
            }
        },
        'radius_neighbors': {
            'class': RadiusNeighborsRegressor,
            'default_params': {
                'radius': 1.0,
                'weights': 'uniform',
                'n_jobs': -1
            }
        },

        # Other verified native multi-output methods
        'gaussian_process': {
            'class': GaussianProcessRegressor,
            'default_params': {
                'random_state': 0,
                'normalize_y': True
            }
        },
        'pls': {
            'class': PLSRegression,
            'default_params': {
                'n_components': 2,
                'scale': True
            }
        },
        'dummy': {
            'class': DummyRegressor,
            'default_params': {
                'strategy': 'mean'
            }
        }
    }

    if regressor_name not in regressor_configs:
        available = list(regressor_configs.keys())
        raise KeyError(f"Unsupported regressor '{regressor_name}'. Available native multi-output options: {available}")

    config = regressor_configs[regressor_name]

    # Merge default parameters with user-provided kwargs
    final_params = config['default_params'].copy()
    final_params.update(kwargs)

    # Create and return the regressor instance
    return config['class'](**final_params)


def validate_data_compatibility(asset_df, confound_df=None, lookback_days=None,
                               days_valid=None, min_required_days=None):
    """
    Validate data compatibility for VAR/DML analysis with comprehensive checks.

    This function performs common validation checks used across VAR and DML methods
    to ensure data meets minimum requirements for analysis.

    Parameters:
    -----------
    asset_df : pd.DataFrame
        Primary time series data (asset returns).
    confound_df : pd.DataFrame, optional
        Confounding/exogenous variables. If provided, must have compatible index.
    lookback_days : int, optional
        Lookback window size. If provided, validates sufficient data length.
    days_valid : int, optional
        Validation period size. If provided with lookback_days, validates relationship.
    min_required_days : int, optional
        Minimum required days. Overrides automatic calculation from other parameters.

    Returns:
    --------
    dict
        Validation results containing:
        - 'valid': bool, True if all checks pass
        - 'errors': list of error messages if any checks fail
        - 'warnings': list of warning messages for potential issues
        - 'data_info': dict with data statistics

    Raises:
    -------
    ValueError
        If critical validation errors are found that would prevent analysis.

    Examples:
    ---------
    >>> result = validate_data_compatibility(asset_df, confound_df, lookback_days=1008)
    >>> if not result['valid']:
    ...     print("Validation errors:", result['errors'])
    """
    errors = []
    warnings = []

    # Basic data checks
    if asset_df.empty:
        errors.append("Asset DataFrame is empty")

    if confound_df is not None:
        if confound_df.empty:
            errors.append("Confound DataFrame is empty")
        elif len(asset_df.index.intersection(confound_df.index)) == 0:
            errors.append("Asset and confound DataFrames have no overlapping dates")
        elif len(asset_df) != len(confound_df):
            warnings.append(f"Asset DataFrame has {len(asset_df)} rows, "
                          f"confound DataFrame has {len(confound_df)} rows")

    # Length validation
    if min_required_days is None and lookback_days is not None:
        min_required_days = lookback_days + 1
        if days_valid is not None:
            min_required_days = max(min_required_days, lookback_days + days_valid + 1)

    if min_required_days is not None:
        if len(asset_df) < min_required_days:
            errors.append(f"Dataset too small: {len(asset_df)} days available, "
                         f"{min_required_days} days required")

    # Parameter relationship checks
    if lookback_days is not None and days_valid is not None:
        if lookback_days <= days_valid:
            errors.append(f"Lookback days ({lookback_days}) must be greater than "
                         f"validation days ({days_valid})")

    # Data quality checks
    asset_na_count = asset_df.isna().sum().sum()
    if asset_na_count > 0:
        warnings.append(f"Asset DataFrame contains {asset_na_count} NaN values")

    if confound_df is not None:
        confound_na_count = confound_df.isna().sum().sum()
        if confound_na_count > 0:
            warnings.append(f"Confound DataFrame contains {confound_na_count} NaN values")

    # Collect data statistics
    data_info = {
        'asset_rows': len(asset_df),
        'asset_cols': len(asset_df.columns),
        'asset_date_range': (asset_df.index.min(), asset_df.index.max()) if hasattr(asset_df.index, 'min') else None,
        'confound_info': None
    }

    if confound_df is not None:
        data_info['confound_info'] = {
            'rows': len(confound_df),
            'cols': len(confound_df.columns),
            'date_range': (confound_df.index.min(), confound_df.index.max()) if hasattr(confound_df.index, 'min') else None
        }

    # Determine if validation passed
    is_valid = len(errors) == 0

    # Raise exception for critical errors if requested
    if not is_valid:
        error_msg = "Data validation failed:\n" + "\n".join(errors)
        if warnings:
            error_msg += "\nWarnings:\n" + "\n".join(warnings)
        raise ValueError(error_msg)

    return {
        'valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'data_info': data_info
    }


def create_index_mapping(T_names_prev, T_names_post, d_y):
    """
    Create index mapping for coefficient comparison between model iterations.

    This function supports the ORACLE VAR method by mapping coefficient indices
    between consecutive model iterations, enabling tracking of coefficient changes
    and significance testing.

    Parameters:
    -----------
    T_names_prev : list of str
        Treatment variable names from the previous model iteration.
    T_names_post : list of str
        Treatment variable names from the current model iteration.
    d_y : int
        Number of outcome variables (assets) in the model.

    Returns:
    --------
    tuple of (list, list)
        - idx_pairs: List of (pre_idx, post_idx) tuples for existing coefficients
        - idx_new: List of post_idx for new coefficients introduced in current iteration

    Notes:
    ------
    This function assumes that coefficients are arranged in blocks by outcome variable,
    i.e., for d_y outcomes and d_T treatment variables, coefficient layout is:
    [Y1_T1, Y1_T2, ..., Y1_Td_T, Y2_T1, Y2_T2, ..., Y2_Td_T, ..., Yd_y_Td_T]

    Examples:
    ---------
    >>> T_prev = ['asset_lag1']
    >>> T_post = ['asset_lag1', 'asset_lag2']
    >>> d_y = 2
    >>> pairs, new = create_index_mapping(T_prev, T_post, d_y)
    >>> # pairs = [(0, 0), (1, 2)], new = [1, 3]
    """
    d_T_prev = len(T_names_prev)
    d_T_post = len(T_names_post)

    idx_pairs = []
    idx_new = []

    # Create mapping for all coefficients organized by outcome variable
    for y in range(d_y):
        for j_post, name_post in enumerate(T_names_post):
            post_idx = y * d_T_post + j_post

            if name_post in T_names_prev:
                # This coefficient existed in the previous model
                j_prev = T_names_prev.index(name_post)
                pre_idx = y * d_T_prev + j_prev
                idx_pairs.append((pre_idx, post_idx))
            else:
                # This is a new coefficient in the current model
                idx_new.append(post_idx)

    return idx_pairs, idx_new


def get_available_regressors():
    """
    Get list of all available native multi-output regressor names and their categories.

    Returns:
    --------
    dict
        Dictionary with categories as keys and lists of regressor names as values.

    Examples:
    ---------
    >>> regressors = get_available_regressors()
    >>> print("Tree-based:", regressors['tree_based'])
    >>> print("Linear:", regressors['linear'])
    >>> print("All verified native multi-output:", regressors['all'])
    """
    return {
        'tree_based': ['decision_tree', 'extra_trees', 'random_forest'],
        'linear': ['ols', 'lasso', 'ridge', 'elastic_net', 'lars', 'lasso_lars'],
        'neighbors': ['knn', 'radius_neighbors'],
        'other': ['gaussian_process', 'pls', 'dummy'],
        'all': ['decision_tree', 'extra_trees', 'random_forest',
                'ols', 'lasso', 'ridge', 'elastic_net', 'lars', 'lasso_lars',
                'knn', 'radius_neighbors', 'gaussian_process', 'pls', 'dummy']
    }


# =============================================================================
# Plotting and Analysis Utilities
# =============================================================================

def rolling_beta(portfolio_returns, benchmark_returns, window=252):
    """
    Calculate rolling beta of portfolio returns against benchmark returns
    using a fast, numerically stable, one-pass algorithm.

    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns series.
    benchmark_returns : pd.Series
        Benchmark returns series (e.g., SPY returns).
    window : int, default=252
        Rolling window size for calculation (252 = ~1 year of daily data).

    Returns:
    --------
    pd.Series
        Series containing rolling beta values.
    """
    # Align the two series and drop any rows with missing values
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    df.columns = ['portfolio', 'benchmark']

    # Get numpy arrays for performance
    p = df['portfolio'].values
    b = df['benchmark'].values

    # Pre-calculate squared benchmark and product of returns
    b_sq = b * b
    pb = p * b

    # Initialize sums for the first window
    n = len(df)
    if n < window:
        # Not enough data, return series of zeros matching original index
        return pd.Series(0, index=portfolio_returns.index)

    # Use float64 for sums to avoid precision issues
    sum_p = np.sum(p[:window]).astype(np.float64)
    sum_b = np.sum(b[:window]).astype(np.float64)
    sum_b_sq = np.sum(b_sq[:window]).astype(np.float64)
    sum_pb = np.sum(pb[:window]).astype(np.float64)

    # Use one-pass algorithm (rank-one update) to calculate rolling beta
    betas = np.zeros(n)

    for i in range(window - 1, n):
        if i >= window:
            # Subtract the value rolling out and add the new value rolling in
            out_idx = i - window
            sum_p += p[i] - p[out_idx]
            sum_b += b[i] - b[out_idx]
            sum_b_sq += b_sq[i] - b_sq[out_idx]
            sum_pb += pb[i] - pb[out_idx]

        # Calculate beta using stable formula
        # Beta = (n * Σ(PB) - ΣP * ΣB) / (n * Σ(B²) - (ΣB)²)
        numerator = window * sum_pb - sum_p * sum_b
        denominator = window * sum_b_sq - sum_b * sum_b

        # Avoid division by zero
        if denominator != 0:
            betas[i] = numerator / denominator
        else:
            betas[i] = 0.0

    # Create pandas Series with correct index
    beta_series = pd.Series(betas, index=df.index)
    beta_series[:window-1] = np.nan  # Set initial values to NaN

    # Reindex to match original portfolio_returns index
    return beta_series.reindex(portfolio_returns.index).fillna(method='ffill').fillna(0)


def get_performance_metrics(cumulative_series, daily_returns_series):
    """
    Calculate performance metrics for a strategy.

    Parameters:
    -----------
    cumulative_series : pd.Series
        Cumulative returns series.
    daily_returns_series : pd.Series
        Daily returns series (percentage change).

    Returns:
    --------
    tuple
        (annualized_return_percent, sharpe_ratio)
    """
    # Clean daily returns
    daily_rets = daily_returns_series.dropna()

    # Calculate Compound Annual Growth Rate (CAGR)
    total_return = cumulative_series.iloc[-1]
    ann_ret = (1 + total_return) ** (252 / len(cumulative_series)) - 1

    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252))

    return ann_ret * 100, sharpe


def plot_lag_analysis(p_optimal, significance_level, save_path=None, show_plot=True, dates=None):
    """
    Plot optimal lag values over time.

    Parameters:
    -----------
    p_optimal : array-like
        Array of optimal lag values for each day.
    significance_level : float
        Significance level used in the analysis.
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
    show_plot : bool, default=True
        Whether to display the plot.
    dates : pd.DatetimeIndex or array-like, optional
        Date indices corresponding to p_optimal values. If None, uses day indices.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if dates is not None:
        x_values = dates
        x_label = 'Date'
    else:
        x_values = np.arange(len(p_optimal))
        x_label = 'Day'

    plt.figure(figsize=(15, 6))

    # Use step plot for cleaner visualization - shows discrete lag values clearly
    plt.step(x_values, p_optimal, where='mid', linewidth=2, alpha=0.8, color='steelblue')

    # Add scatter points to emphasize individual data points
    plt.scatter(x_values, p_optimal, alpha=0.7, s=30, color='darkblue', zorder=5)

    # Fill area under the step plot for better visual impact
    plt.fill_between(x_values, p_optimal, step='mid', alpha=0.3, color='lightblue')

    plt.xlabel(x_label)
    plt.ylabel('p_optimal Value')
    plt.title(f'p_optimal Values Over Time\n(Significance Level={significance_level})')
    plt.grid(True, alpha=0.3, zorder=0)

    # Set y-axis to show integer ticks (since p_optimal are lag counts)
    from matplotlib.ticker import MaxNLocator
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # Format x-axis for dates
    if dates is not None:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_strategy_comparison(pnl_results, significance_level, include_spy=True,
                           save_path=None, show_plot=True, market_adjusted=False):
    """
    Plot comparison of different trading strategies.

    Parameters:
    -----------
    pnl_results : dict
        Dictionary containing strategy results with keys like 'naive', 'weighted', etc.
        Each value should be a tuple of (cumulative_returns, daily_returns, positions).
    significance_level : float
        Significance level used in analysis.
    include_spy : bool, default=True
        Whether to include SPY benchmark in the plot.
    save_path : str, optional
        Path to save the plot.
    show_plot : bool, default=True
        Whether to display the plot.
    market_adjusted : bool, default=False
        Whether the returns are market-adjusted.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Color scheme for strategies
    colors = {
        'naive': 'blue',
        'weighted': 'green',
        'top_50': 'red',
        'top_25': 'coral',
        'top_75': 'darkred',
        'spy': 'purple'
    }

    # Plot strategy results
    for strategy_name, (cumulative_returns, daily_returns, positions) in pnl_results.items():
        if strategy_name != 'spy':  # Handle SPY separately
            color = colors.get(strategy_name, 'black')
            label = strategy_name.replace('_', ' ').title()
            if strategy_name.startswith('top_'):
                percentile = strategy_name.split('_')[1]
                label = f'Top Strategy, {percentile}% percentile'
            plt.plot(cumulative_returns, color=color, label=label, linewidth=2)

    # Add SPY if requested and available
    if include_spy and not market_adjusted and 'spy' in pnl_results:
        spy_cumulative, _, _ = pnl_results['spy']
        plt.plot(spy_cumulative, color=colors['spy'], label='SPY', linewidth=2)

    # Customize plot
    title = f"{'Market-Adjusted ' if market_adjusted else ''}PnL Comparison Across Different Strategies"
    title += f"\n(Significance Level={significance_level})"
    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('PnL', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    else:
        plt.close()


def print_performance_summary(pnl_results, significance_level):
    """
    Print performance summary for all strategies.

    Parameters:
    -----------
    pnl_results : dict
        Dictionary containing strategy results.
    significance_level : float
        Significance level used in analysis.
    """
    print(f"Results for Significance Level: {significance_level}")
    print("-" * 50)

    strategy_labels = {
        'naive': 'Naive',
        'weighted': 'Weighted',
        'top_50': 'Top 50%',
        'top_25': 'Top 25%',
        'top_75': 'Top 75%',
        'spy': 'SPY Benchmark'
    }

    for strategy_name, (cumulative, daily_returns, _) in pnl_results.items():
        label = strategy_labels.get(strategy_name, strategy_name.title())
        ann_ret, sharpe = get_performance_metrics(cumulative, daily_returns)
        print(f"  {label}: {ann_ret:.4f}% annualized, Sharpe: {sharpe:.4f}")

    print("-" * 50)
