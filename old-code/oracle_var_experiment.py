"""
ICAIF 2025 - Comprehensive VAR Analysis Experiment
==================================================

Complete end-to-end VAR analysis supporting multiple analysis types:
- Plain VAR: Traditional VAR without DML
- OR-VAR: Orthogonal Regression with DML
- ORACLE-VAR: Oracle VAR with significance-level-based lag selection

Modify the global variables below to customize the experiment.

Usage:
    python oracle_var_experiment.py

Results:
    - Generates plots for lag analysis and strategy comparison
    - Prints performance metrics for all strategies
    - Saves results to pickle file
"""
import multiprocessing

# =============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE VARIABLES
# =============================================================================

# Analysis Type Selection
ANALYSIS_TYPE = 'OR-VAR'       # Options: 'Plain VAR', 'OR-VAR', 'ORACLE-VAR'

# Asset Selection
ASSET_SELECTION = 'ETF'            # Options: 'ETF', 'TOP20'

# Confounding Variables Control - Toggle True/False to use these datasets
USE_MACRO = False                   # Federal Funds Rate, 5-Year Breakeven Inflation, Economic Policy Uncertainty, 10-Year Treasury
USE_TRADE = False                   # WTI Crude Oil, USD Indices (Major & Emerging), CBOE Gold ETF Volatility
USE_VIX = True                     # VIX Volatility Index
USE_COVID = False                   # US COVID dataset (filtered to 2020-12-31)

# Data Subset Control
SUBSET_DAYS = None                # Analysis period: None for full dataset, or number of days for subset analysis
# If SUBSET_DAYS is set, the analysis will use the last SUBSET_DAYS + LOOKBACK_DAYS days of data

# Analysis Parameters
SIGNIFICANCE_LEVEL = 0.15           # For ORACLE-VAR only: significance level for hypothesis testing
LOOKBACK_DAYS = 252                 # Lookback window (1 or 4 years of daily data)
DAYS_VALID = 2                      # Validation period for OR-VAR parameter search
P_MAX = 10                           # Maximum number of lags to consider
MAX_THREADS = multiprocessing.cpu_count()  # Use all available CPU cores for parallel processing

# ML Method Configuration
MODEL_Y_NAME = 'extra_trees'      # Outcome model: 'extra_trees', 'random_forest', 'lasso', 'ols'
MODEL_T_NAME = 'extra_trees'      # Treatment model (for DML methods): 'extra_trees', 'random_forest', 'lasso', 'ols'
CV_FOLDS = 5                      # Cross-validation folds for DML methods
# For PlainVAR, MODEL_T_NAME is ignored and only MODEL_Y_NAME is used (which only supports 'ols' and 'lasso')

# ML Parameters (customize based on chosen method - though you can use the defaults, which you can modify
# in the core_utils.py module)
# MODEL_Y_PARAMS = {
#                 'n_estimators': 200,           # More trees for stability
#                 'max_depth': 5,               # Limit depth to prevent overfitting
#                 'min_samples_split': 20,       # Better generalization
#                 'min_samples_leaf': 10,        # Prevent overfitting
#                 'max_features': 0.8,           # Consider subset of features
#                 'bootstrap': False,            # Default for Extra Trees
#                 'random_state': 0,
#                 'n_jobs': -1
# }

MODEL_Y_PARAMS = {}

# MODEL_T_PARAMS = {
#                 'n_estimators': 200,           # More trees for stability
#                 'max_depth': 5,               # Limit depth to prevent overfitting
#                 'min_samples_split': 20,       # Better generalization
#                 'min_samples_leaf': 10,        # Prevent overfitting
#                 'max_features': 0.8,           # Consider subset of features
#                 'bootstrap': False,            # Default for Extra Trees
#                 'random_state': 0,
#                 'n_jobs': -1
# }

MODEL_T_PARAMS = {}

# Predefined ETF List
ETFS = [
    "SPY", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLK", "XLU",
    "EWH", "EWJ", "EWQ", "EWS", "EWU"
]

# PnL Strategy Configuration
PNL_STRATEGIES = {
    'naive': {'pnl_strategy': 'naive'},
    'weighted': {'pnl_strategy': 'weighted'}, 
    'top_50': {'pnl_strategy': 'top', 'percentile': 0.5},
    'top_25': {'pnl_strategy': 'top', 'percentile': 0.25},
    'top_75': {'pnl_strategy': 'top', 'percentile': 0.75}
}

# Market Adjustment Settings
MARKET_ADJUSTMENT = True           # Apply SPY market adjustment
BETA_NEUTRAL = False               # Use rolling beta neutrality (by default, this is false)
BETA_WINDOW = 63                   # Rolling beta calculation window (~3 months, by default, we don't use this)

# Output Configuration
SAVE_PLOTS = True                  # Save plots to files
SHOW_PLOTS = False                 # Display plots (set False for batch processing)
RESULTS_DIR = 'experiment_results' # Directory for saving results

# Create appropriate experiment name based on analysis type
if ANALYSIS_TYPE == 'Plain VAR':
    EXPERIMENT_NAME = f'plain_var_{ASSET_SELECTION.lower()}_{MODEL_Y_NAME}'
elif ANALYSIS_TYPE == 'OR-VAR':
    EXPERIMENT_NAME = f'or_var_{ASSET_SELECTION.lower()}_{MODEL_Y_NAME}_{MODEL_T_NAME}'
elif ANALYSIS_TYPE == 'ORACLE-VAR':
    EXPERIMENT_NAME = f'oracle_var_{ASSET_SELECTION.lower()}_sig_{SIGNIFICANCE_LEVEL}_{MODEL_Y_NAME}_{MODEL_T_NAME}'
else:
    EXPERIMENT_NAME = f'var_experiment_{ASSET_SELECTION.lower()}'

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import sys
import pickle
import time
from functools import reduce

# Add modules path
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(current_dir, 'Modules')
if modules_path not in sys.path:
    sys.path.append(modules_path)

# Import custom modules
from core_utils import (
    make_lags, realign, get_regressor, validate_data_compatibility,
    rolling_beta, get_performance_metrics, plot_lag_analysis,
    plot_strategy_comparison, print_performance_summary
)
from pnl_calculator import calculate_pnl

# Import analysis modules based on type
if ANALYSIS_TYPE == 'Plain VAR':
    from VAR_parallelized import parallel_rolling_window_VAR
elif ANALYSIS_TYPE == 'OR-VAR':
    from DML_parallelized import parallel_rolling_window_OR_VAR
elif ANALYSIS_TYPE == 'ORACLE-VAR':
    from DML_parallelized import parallel_rolling_window_ORACLE_VAR
else:
    raise ValueError(f"Invalid ANALYSIS_TYPE: {ANALYSIS_TYPE}. Must be 'Plain VAR', 'OR-VAR', or 'ORACLE-VAR'")

# =============================================================================
# MAIN EXECUTION GUARD (Required for Windows multiprocessing)
# =============================================================================

if __name__ == '__main__':
    # Create results directory with experiment subfolder
    experiment_folder = os.path.join(RESULTS_DIR, EXPERIMENT_NAME)
    os.makedirs(experiment_folder, exist_ok=True)

    print("="*80)
    print("VAR Analysis Experiment")
    print("="*80)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Analysis Type: {ANALYSIS_TYPE}")
    print(f"Asset Selection: {ASSET_SELECTION}")
    # Check if any confounding variables are enabled
    confounding_enabled = USE_MACRO or USE_TRADE or USE_VIX or USE_COVID
    if confounding_enabled:
        confound_status = []
        if USE_MACRO: confound_status.append("MACRO")
        if USE_TRADE: confound_status.append("TRADE")
        if USE_VIX: confound_status.append("VIX")
        if USE_COVID: confound_status.append("COVID")
        print(f"Confounding Variables: Enabled ({'+'.join(confound_status)})")
    else:
        print(f"Confounding Variables: Disabled")
    print(f"Data Subset: {'Full dataset' if SUBSET_DAYS is None else f'{SUBSET_DAYS} days + {LOOKBACK_DAYS} lookback'}")
    if ANALYSIS_TYPE == 'ORACLE-VAR':
        print(f"Significance Level: {SIGNIFICANCE_LEVEL}")
    print(f"Lookback Days: {LOOKBACK_DAYS}")
    print(f"Model: {MODEL_Y_NAME}" + (f" (Y) / {MODEL_T_NAME} (T)" if ANALYSIS_TYPE != 'Plain VAR' else ""))
    print("="*80)

    # =============================================================================
    # DATA LOADING AND PREPROCESSING
    # =============================================================================

    print("\n🔄 Loading and preprocessing data...")

    # Load asset returns data
    data_folder_path = os.path.join(current_dir, 'Data')
    data_file_path = os.path.join(data_folder_path, "OPCL_20000103_20201231.csv")

    print(f"Loading asset data from: {data_file_path}")
    returns_df = pd.read_csv(data_file_path)
    # Assumes CSV has 'ticker' column and date columns in format 'XYYYYYMMDD'
    returns_df.set_index('ticker', inplace=True)
    # Converts date columns from 'XYYYYYMMDD' format to 'YYYY-MM-DD'
    returns_df.columns = pd.to_datetime(returns_df.columns.str.lstrip('X'), format='%Y%m%d').strftime('%Y-%m-%d')
    returns_df_cleaned = returns_df.dropna().transpose()
    returns_df_cleaned.index = pd.to_datetime(returns_df_cleaned.index)
    returns_df_cleaned.index.name = 'date'

    print(f"✅ Asset data loaded: {returns_df_cleaned.shape}")
    print(f"Date range: {returns_df_cleaned.index.min()} to {returns_df_cleaned.index.max()}")

    # Define confounding variable categories
    macro_files = [
        "DFF_20000103_20201231.csv",         # Federal Funds Effective Rate
        "T5YIE_20030102_20201231.csv",       # 5-Year Breakeven Inflation Rate (from 2003)
        "USEPUINDXD_20000103_20201231.csv",  # Economic Policy Uncertainty Index
        "DFII10_20030102_20201231.csv"       # 10-Year Treasury Constant Maturity Rate
    ]
    
    trade_files = [
        "DCOILWTICO_20000103_20201231.csv",  # West Texas Intermediate (WTI) Crude Oil Prices
        "DTWEXBGS_20060102_20201231.csv",    # Broad U.S. Dollar Index: Major Currencies (from 2006)
        "DTWEXEMEGS_20060102_20201231.csv",  # Broad U.S. Dollar Index: Emerging Markets (from 2006)
        "GVZCLS_20080603_20201231.csv"       # CBOE Gold ETF Volatility Index
    ]
    
    vix_files = [
        "VIX_20000103_20201231.csv"          # VIX Volatility Index
    ]
    
    covid_files = [
        "USCOVIDDATA_20200101_20221231.csv"  # US COVID dataset (filtered to 2020-12-31)
    ]
    
    # Build list of files to load based on configuration
    confounding_files = []
    if USE_MACRO:
        confounding_files.extend(macro_files)
    if USE_TRADE:
        confounding_files.extend(trade_files)
    if USE_VIX:
        confounding_files.extend(vix_files)
    if USE_COVID:
        confounding_files.extend(covid_files)

    print(f"Loading confounding variables based on configuration:")
    print(f"   USE_MACRO: {USE_MACRO} ({'✓' if USE_MACRO else '✗'}) - {len(macro_files)} files")
    print(f"   USE_TRADE: {USE_TRADE} ({'✓' if USE_TRADE else '✗'}) - {len(trade_files)} files")
    print(f"   USE_VIX: {USE_VIX} ({'✓' if USE_VIX else '✗'}) - {len(vix_files)} files")
    print(f"   USE_COVID: {USE_COVID} ({'✓' if USE_COVID else '✗'}) - {len(covid_files)} files")
    print(f"   Total files to load: {len(confounding_files)}")
    
    dfs_with_index = []
    for f in confounding_files:
        try:
            # Assumes each CSV has 'observation_date' column in first position
            df = pd.read_csv(os.path.join(data_folder_path, f), parse_dates=[0])
            # Assumes the date column is named 'observation_date'
            df.set_index('observation_date', inplace=True)
            
            # Special handling for COVID data - filter to 2020-12-31 to match other datasets
            if 'USCOVIDDATA' in f:
                df = df[df.index <= '2020-12-31']
                print(f"   ✅ Loaded: {f} (filtered to 2020-12-31: {df.shape[0]} rows)")
            else:
                print(f"   ✅ Loaded: {f}")
            
            dfs_with_index.append(df)
        except FileNotFoundError:
            print(f"   ⚠️  File not found (skipping): {f}")
        except Exception as e:
            print(f"   ❌ Error loading {f}: {e}")

    # Merge confounding variables (only if any were loaded)
    if dfs_with_index:
        merged_confound_df = reduce(
            lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="outer"),
            dfs_with_index
        )

        # Forward-only interpolation to avoid lookahead bias
        imputed_confound_df = merged_confound_df.interpolate(method='linear', limit_direction='forward')
        imputed_confound_df = imputed_confound_df.fillna(0)
        imputed_confound_df.index.name = 'date'

        # Filter confounding variables to match trading dates
        filtered_confound_df = imputed_confound_df[imputed_confound_df.index.isin(returns_df_cleaned.index)]

        print(f"✅ Confounding data processed: {filtered_confound_df.shape}")
    else:
        filtered_confound_df = pd.DataFrame(index=returns_df_cleaned.index)
        print(f"⚠️  No confounding variables loaded - using empty DataFrame")

    # Prepare analysis datasets based on asset selection
    if ASSET_SELECTION == 'ETF':
        print(f"Using predefined ETF list: {len(ETFS)} assets")
        returns_df_prep = returns_df_cleaned[ETFS]
    elif ASSET_SELECTION == 'TOP20':
        print("Using first 20 stocks from dataset")
        available_stocks = returns_df_cleaned.columns[:20]  # First 20 stocks using :20 slicing
        returns_df_prep = returns_df_cleaned[available_stocks]
        print(f"✅ Selected first 20 stocks: {list(available_stocks)}")
    else:
        raise ValueError(f"Invalid ASSET_SELECTION: {ASSET_SELECTION}. Must be 'ETF' or 'TOP20'")

    # Prepare confounding data based on confounding variable settings
    confounding_enabled = USE_MACRO or USE_TRADE or USE_VIX or USE_COVID
    if confounding_enabled and not filtered_confound_df.empty:
        confound_df_prep = filtered_confound_df.loc[returns_df_prep.index]
        print(f"✅ Analysis datasets prepared:")
        print(f"   Returns: {returns_df_prep.shape}")
        print(f"   Confounding: {confound_df_prep.shape} (enabled with {confound_df_prep.shape[1]} variables)")
    else:
        confound_df_prep = None
        print(f"✅ Analysis datasets prepared:")
        print(f"   Returns: {returns_df_prep.shape}")
        print(f"   Confounding: disabled (no variables enabled or loaded)")

    # Apply data subset if specified
    if SUBSET_DAYS is not None:
        print(f"\n📊 Applying data subset: {SUBSET_DAYS} days + {LOOKBACK_DAYS} lookback days + 'Today'...")
        
        # Calculate total days needed
        total_days_needed = (LOOKBACK_DAYS + 1) + SUBSET_DAYS
        
        # Take the most recent data points
        if len(returns_df_prep) < total_days_needed:
            print(f"⚠️  Warning: Dataset only has {len(returns_df_prep)} days, but {total_days_needed} requested")
            print(f"   Using all available data")
            # Keep all data if insufficient
            subset_start_date = returns_df_prep.index.min()
        else:
            # Calculate subset from the end backwards
            subset_start_idx = len(returns_df_prep) - total_days_needed
            subset_start_date = returns_df_prep.index[subset_start_idx]
            
            print(f"   Original data range: {returns_df_prep.index.min()} to {returns_df_prep.index.max()}")
            print(f"   Subset data range: {subset_start_date} to {returns_df_prep.index.max()}")
            
            # Apply subset to returns data
            returns_df_prep = returns_df_prep.loc[subset_start_date:]
            
            # Apply subset to confounding data if enabled
            confounding_enabled = USE_MACRO or USE_TRADE or USE_VIX or USE_COVID
            if confounding_enabled and confound_df_prep is not None:
                confound_df_prep = confound_df_prep.loc[subset_start_date:]
        
        print(f"✅ Data subset applied:")
        print(f"   Returns: {returns_df_prep.shape}")
        confounding_enabled = USE_MACRO or USE_TRADE or USE_VIX or USE_COVID
        if confounding_enabled and confound_df_prep is not None:
            print(f"   Confounding: {confound_df_prep.shape}")
    else:
        print(f"\n📊 Using full dataset (SUBSET_DAYS = None)")

    # Validate data compatibility
    print("\n🔍 Validating data compatibility...")
    try:
        confounding_enabled = USE_MACRO or USE_TRADE or USE_VIX or USE_COVID
        if confounding_enabled and confound_df_prep is not None:
            validate_data_compatibility(
                returns_df_prep,
                confound_df_prep,
                lookback_days=LOOKBACK_DAYS,
                days_valid=DAYS_VALID
            )
        else:
            # When confounding is disabled, only validate returns data
            min_required = LOOKBACK_DAYS + DAYS_VALID + 1
            if len(returns_df_prep) < min_required:
                raise ValueError(f"Insufficient data: need at least {min_required} days, got {len(returns_df_prep)}")
        print("✅ Data validation passed")
    except ValueError as e:
        print(f"❌ Data validation failed: {e}")
        sys.exit(1)

    # =============================================================================
    # VAR ANALYSIS
    # =============================================================================

    print(f"\n🚀 Running {ANALYSIS_TYPE} analysis...")
    print(f"   Lookback window: {LOOKBACK_DAYS} days")
    print(f"   Maximum lags: {P_MAX}")
    if ANALYSIS_TYPE == 'ORACLE-VAR':
        print(f"   Significance level: {SIGNIFICANCE_LEVEL}")
    elif ANALYSIS_TYPE == 'OR-VAR':
        print(f"   Validation days: {DAYS_VALID}")
    print(f"   Threads: {MAX_THREADS}")

    start_time = time.perf_counter()

    # Call appropriate analysis function based on type
    if ANALYSIS_TYPE == 'Plain VAR':
        # Plain VAR only supports 'ols' and 'lasso' models
        model_for_var = MODEL_Y_NAME if MODEL_Y_NAME in ['ols', 'lasso'] else 'ols'
        if model_for_var != MODEL_Y_NAME:
            print(f"⚠️  Warning: Plain VAR only supports 'ols' and 'lasso', using '{model_for_var}' instead of '{MODEL_Y_NAME}'")
        
        results = parallel_rolling_window_VAR(
            asset_df=returns_df_prep,
            confound_df=confound_df_prep,
            p_max=P_MAX,
            model_type=model_for_var,
            lookback_days=LOOKBACK_DAYS,
            days_valid=DAYS_VALID,
            max_threads=MAX_THREADS
        )
    elif ANALYSIS_TYPE == 'OR-VAR':
        results = parallel_rolling_window_OR_VAR(
            asset_df=returns_df_prep,
            confound_df=confound_df_prep,
            p_max=P_MAX,
            model_y_name=MODEL_Y_NAME,
            model_t_name=MODEL_T_NAME,
            model_y_params=MODEL_Y_PARAMS,
            model_t_params=MODEL_T_PARAMS,
            cv_folds=CV_FOLDS,
            lookback_days=LOOKBACK_DAYS,
            days_valid=DAYS_VALID,
            max_threads=MAX_THREADS
        )
    elif ANALYSIS_TYPE == 'ORACLE-VAR':
        results = parallel_rolling_window_ORACLE_VAR(
            asset_df=returns_df_prep,
            confound_df=confound_df_prep,
            p_max=P_MAX,
            model_y_name=MODEL_Y_NAME,
            model_t_name=MODEL_T_NAME,
            model_y_params=MODEL_Y_PARAMS,
            model_t_params=MODEL_T_PARAMS,
            cv_folds=CV_FOLDS,
            lookback_days=LOOKBACK_DAYS,
            significance_level=SIGNIFICANCE_LEVEL,
            max_threads=MAX_THREADS
        )

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"✅ {ANALYSIS_TYPE} analysis completed!")
    print(f"   Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"   Processed {len(results['p_optimal'])} days")

    # Store timing results
    results['execution_time_seconds'] = execution_time
    results['experiment_config'] = {
        'analysis_type': ANALYSIS_TYPE,
        'asset_selection': ASSET_SELECTION,
        'use_macro': USE_MACRO,
        'use_trade': USE_TRADE,
        'use_vix': USE_VIX,
        'use_covid': USE_COVID,
        'significance_level': SIGNIFICANCE_LEVEL,
        'lookback_days': LOOKBACK_DAYS,
        'p_max': P_MAX,
        'model_y_name': MODEL_Y_NAME,
        'model_t_name': MODEL_T_NAME,
        'etfs': ETFS,
        'market_adjustment': MARKET_ADJUSTMENT,
        'beta_neutral': BETA_NEUTRAL
    }

    # =============================================================================
    # RESULTS ANALYSIS AND VISUALIZATION
    # =============================================================================

    print("\n📊 Analyzing results and generating visualizations...")

    # Extract forecast and actual returns
    forecasted_returns = pd.DataFrame(
        results['Y_hat_next_store'],
        index=returns_df_prep.index[LOOKBACK_DAYS+1:],
        columns=returns_df_prep.columns
    )

    actual_returns = returns_df_prep.iloc[LOOKBACK_DAYS+1:, :]

    # Load SPY separately for market adjustment if needed and not already included
    spy_returns_for_adjustment = None
    if MARKET_ADJUSTMENT and 'SPY' not in actual_returns.columns:
        if 'SPY' in returns_df_cleaned.columns:
            print("📈 Loading SPY separately for market adjustment...")
            # Get SPY data for the same time period as actual_returns
            spy_full_data = returns_df_cleaned['SPY']
            
            # Apply the same data subset if specified
            if SUBSET_DAYS is not None:
                spy_full_data = spy_full_data.loc[returns_df_prep.index]
            
            # Extract SPY returns for the same period as actual returns
            spy_returns_for_adjustment = spy_full_data.iloc[LOOKBACK_DAYS+1:]
            
            # Ensure alignment with actual_returns index
            spy_returns_for_adjustment = spy_returns_for_adjustment.reindex(actual_returns.index)
            print(f"✅ SPY data loaded for market adjustment: {spy_returns_for_adjustment.shape}")
        else:
            print("⚠️  SPY not available in full dataset - market adjustment will be disabled")
            MARKET_ADJUSTMENT = False

    print(f"✅ Prepared forecast data:")
    print(f"   Forecasted returns: {forecasted_returns.shape}")
    print(f"   Actual returns: {actual_returns.shape}")
    if spy_returns_for_adjustment is not None:
        print(f"   SPY for market adjustment: {spy_returns_for_adjustment.shape}")

    # Generate lag analysis plot
    print("📈 Generating lag analysis plot...")
    # Only pass significance level for ORACLE-VAR analysis
    sig_level_for_plot = SIGNIFICANCE_LEVEL if ANALYSIS_TYPE == 'ORACLE-VAR' else None
    # Get the date indices for the analysis period
    analysis_dates = returns_df_prep.index[LOOKBACK_DAYS+1:]
    plot_lag_analysis(
        results['p_optimal'],
        sig_level_for_plot,
        save_path=os.path.join(experiment_folder, f'{EXPERIMENT_NAME}_lag_analysis.png') if SAVE_PLOTS else None,
        show_plot=SHOW_PLOTS,
        dates=analysis_dates
    )

    # Calculate PnL for all strategies
    print("💰 Calculating PnL for all strategies...")
    pnl_results = {}

    for strategy_name, strategy_params in PNL_STRATEGIES.items():
        print(f"   Computing {strategy_name} strategy...")
        
        # Calculate without market adjustment first
        cumulative, daily_pct, positions = calculate_pnl(
            forecasted_returns, 
            actual_returns,
            contrarian=False,
            market_adjustment=False,
            beta_adjustment=False,
            **strategy_params
        )
        
        # Apply market adjustment if requested
        if MARKET_ADJUSTMENT:
            cumulative_adj, daily_pct_adj, positions_adj = calculate_pnl(
                forecasted_returns,
                actual_returns,
                contrarian=False,
                market_adjustment=True,
                beta_adjustment=BETA_NEUTRAL,
                beta_window=BETA_WINDOW,
                benchmark_returns=spy_returns_for_adjustment,
                **strategy_params
            )
            pnl_results[strategy_name] = (cumulative_adj, daily_pct_adj, positions_adj)
        else:
            pnl_results[strategy_name] = (cumulative, daily_pct, positions)

    # Add SPY benchmark for comparison (if not market adjusted and SPY is available)
    if not MARKET_ADJUSTMENT:
        spy_data_for_benchmark = None
        if 'SPY' in actual_returns.columns:
            spy_data_for_benchmark = actual_returns['SPY']
        elif spy_returns_for_adjustment is not None:
            spy_data_for_benchmark = spy_returns_for_adjustment
        
        if spy_data_for_benchmark is not None:
            print("   Adding SPY benchmark...")
            spy_returns = np.exp(spy_data_for_benchmark.cumsum()) - 1
            spy_daily = np.exp(spy_data_for_benchmark) - 1
            spy_positions = pd.Series(1.0, index=spy_data_for_benchmark.index)  # Always fully invested
            pnl_results['spy'] = (spy_returns, spy_daily, spy_positions)
        else:
            print("   ⚠️  SPY not available in dataset - skipping benchmark")

    print(f"✅ PnL calculation completed for {len(pnl_results)} strategies")

    # Generate strategy comparison plot
    print("📈 Generating strategy comparison plot...")
    # Only pass significance level for ORACLE-VAR analysis
    sig_level_for_plot = SIGNIFICANCE_LEVEL if ANALYSIS_TYPE == 'ORACLE-VAR' else None
    plot_strategy_comparison(
        pnl_results,
        sig_level_for_plot,
        include_spy=not MARKET_ADJUSTMENT,
        market_adjusted=MARKET_ADJUSTMENT,
        save_path=os.path.join(experiment_folder, f'{EXPERIMENT_NAME}_strategy_comparison.png') if SAVE_PLOTS else None,
        show_plot=SHOW_PLOTS
    )

    # Print performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    # Only pass significance level for ORACLE-VAR analysis
    sig_level_for_summary = SIGNIFICANCE_LEVEL if ANALYSIS_TYPE == 'ORACLE-VAR' else None
    print_performance_summary(pnl_results, sig_level_for_summary)

    # =============================================================================
    # SAVE RESULTS
    # =============================================================================

    print(f"\n💾 Saving results to {experiment_folder}...")

    # Save complete results
    results_file = os.path.join(experiment_folder, f'{EXPERIMENT_NAME}_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    # Save PnL results separately for easy access
    pnl_file = os.path.join(experiment_folder, f'{EXPERIMENT_NAME}_pnl_results.pkl')
    with open(pnl_file, 'wb') as f:
        pickle.dump(pnl_results, f)

    # Save performance metrics as CSV
    performance_data = []
    for strategy_name, (cumulative, daily_returns, _) in pnl_results.items():
        ann_ret, sharpe = get_performance_metrics(cumulative, daily_returns)
        performance_data.append({
            'strategy': strategy_name,
            'annualized_return_pct': ann_ret,
            'sharpe_ratio': sharpe,
            'total_return': cumulative.iloc[-1] if len(cumulative) > 0 else 0
        })

    performance_df = pd.DataFrame(performance_data)
    performance_file = os.path.join(experiment_folder, f'{EXPERIMENT_NAME}_performance.csv')
    performance_df.to_csv(performance_file, index=False)

    print(f"✅ Results saved:")
    print(f"   Complete results: {results_file}")
    print(f"   PnL results: {pnl_file}")
    print(f"   Performance metrics: {performance_file}")

    if SAVE_PLOTS:
        print(f"   Plots saved to: {experiment_folder}/")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Results saved to: {experiment_folder}")

    # Display key findings
    best_strategy = performance_df.loc[performance_df['sharpe_ratio'].idxmax()]
    print(f"\n🏆 Best performing strategy (by Sharpe ratio):")
    print(f"   Strategy: {best_strategy['strategy']}")
    print(f"   Annualized Return: {best_strategy['annualized_return_pct']:.4f}%")
    print(f"   Sharpe Ratio: {best_strategy['sharpe_ratio']:.4f}")

    print(f"\n📊 Average optimal lag: {np.mean(results['p_optimal']):.2f}")
    print(f"📊 Lag range: {np.min(results['p_optimal']):.0f} - {np.max(results['p_optimal']):.0f}")

    print("\n🎯 Experiment configuration summary:")
    print(f"   Analysis Type: {ANALYSIS_TYPE}")
    print(f"   Asset Selection: {ASSET_SELECTION}")
    print(f"   Data Subset: {'Full dataset' if SUBSET_DAYS is None else f'{SUBSET_DAYS} days + {LOOKBACK_DAYS} lookback'}")
    confounding_enabled = USE_MACRO or USE_TRADE or USE_VIX or USE_COVID
    if confounding_enabled:
        confound_summary = []
        if USE_MACRO: confound_summary.append("MACRO")
        if USE_TRADE: confound_summary.append("TRADE")
        if USE_VIX: confound_summary.append("VIX")
        if USE_COVID: confound_summary.append("COVID")
        print(f"   Confounding Variables: Enabled ({'+'.join(confound_summary)})")
    else:
        print(f"   Confounding Variables: Disabled")
    if ANALYSIS_TYPE == 'ORACLE-VAR':
        print(f"   Significance Level: {SIGNIFICANCE_LEVEL}")
    elif ANALYSIS_TYPE == 'OR-VAR':
        print(f"   Validation Days: {DAYS_VALID}")
    print(f"   Lookback Days: {LOOKBACK_DAYS}")
    if ANALYSIS_TYPE == 'Plain VAR':
        print(f"   Model: {MODEL_Y_NAME}")
    else:
        print(f"   Models: {MODEL_Y_NAME} (Y) / {MODEL_T_NAME} (T)")
    print(f"   Market Adjustment: {MARKET_ADJUSTMENT}")
    print(f"   Beta Neutrality: {BETA_NEUTRAL}")
    print("="*80)