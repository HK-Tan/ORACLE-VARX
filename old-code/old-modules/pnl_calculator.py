import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

def rolling_beta(portfolio_returns, benchmark_returns, window=252):
    """
    Calculates the rolling beta of portfolio returns against benchmark returns
    using a fast, numerically stable, one-pass algorithm (rank-one update).

    Inputs:
    portfolio_returns (pd.Series): A series of portfolio returns.
    benchmark_returns (pd.Series): A series of benchmark returns.
    window (int): The rolling window size for the calculation.

    Output:
    pd.Series: A series containing the rolling beta values.
    """
    # 1. Align the two series and drop any rows with missing values
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    df.columns = ['portfolio', 'benchmark']

    # Get numpy arrays for performance
    p = df['portfolio'].values
    b = df['benchmark'].values

    # 2. Pre-calculate squared benchmark and product of returns
    b_sq = b * b
    pb = p * b

    # 3. Initialize sums for the first window
    n = len(df)
    if n < window:
        # Not enough data, return series of zeros matching original index
        return pd.Series(0, index=portfolio_returns.index)

    # Use float64 for sums to avoid precision issues
    sum_p = np.sum(p[:window]).astype(np.float64)
    sum_b = np.sum(b[:window]).astype(np.float64)
    sum_b_sq = np.sum(b_sq[:window]).astype(np.float64)
    sum_pb = np.sum(pb[:window]).astype(np.float64)

    # 4. Use a one-pass algorithm (rank-one update) to calculate rolling beta
    betas = np.zeros(n)

    for i in range(window - 1, n):
        if i >= window:
            # Subtract the value that is rolling out of the window
            # and add the new value that is rolling in.
            out_idx = i - window
            sum_p += p[i] - p[out_idx]
            sum_b += b[i] - b[out_idx]
            sum_b_sq += b_sq[i] - b_sq[out_idx]
            sum_pb += pb[i] - pb[out_idx]

        # Calculate beta using the stable formula
        # Beta = (n * Σ(PB) - ΣP * ΣB) / (n * Σ(B²) - (ΣB)²)
        numerator = window * sum_pb - sum_p * sum_b
        denominator = window * sum_b_sq - sum_b * sum_b

        # Avoid division by zero
        if denominator != 0:
            betas[i] = numerator / denominator
        else:
            betas[i] = 0.0 # Or np.nan, depending on desired behavior

    # 5. Format the output to match the original function
    # Create a pandas Series with the correct index
    beta_series = pd.Series(betas, index=df.index)
    beta_series[:window-1] = np.nan # Set initial values to NaN as rolling() does

    # Reindex to match the original portfolio_returns index, ffilling NaNs
    return beta_series.reindex(portfolio_returns.index).fillna(method='ffill').fillna(0)

def calculate_pnl(forecast_df, actual_df, pnl_strategy="weighted", percentile=0.5, contrarian=False,
                 beta_neutral=False, market_adjustment=True, beta_window=252, benchmark="SPY",
                 benchmark_returns=None, **kwargs):
    """
    This function calculates the PnL based on the forecasted returns and actual returns.
    Based on the enhanced version from ORACLE_VAR_etf_market_adjusted_testbed.ipynb

    Inputs:
    forecast_df: DataFrame containing the forecasted returns for each asset/cluster.
    actual_df: DataFrame containing the actual returns for each asset/cluster (in terms of log returns).
    pnl_strategy: Strategy for calculating PnL. Options are:
        - "naive": Go long $1 on clusters with positive forecast return, go short $1 on clusters with negative forecast return.
        - "weighted": Weight based on the predicted return of each cluster.
        - "top": Only choose clusters with absolute returns above a certain percentile.
    percentile: If pnl_strategy is "top", this is the percentile threshold for selecting clusters.
    contrarian: If True, inverts the trading signals (bets against forecasts).
    beta_neutral: If True, use rolling beta for market adjustment instead of simple subtraction.
    market_adjustment: If True, adjust returns for market (default True as per user requirement).
    beta_window: Number of days for rolling beta calculation.
    benchmark: Benchmark ticker for market adjustment (default "SPY").
    benchmark_returns: Optional pd.Series of benchmark returns if not in actual_df.

    Remark:
    The dataframes keep daily data as rows, with columns as different assets or clusters.
    We also assume that these df are aligned.

    Output:
    tuple of (pd.Series, pd.Series, pd.Series)
        - cumulative_returns: Cumulative portfolio returns starting from 0
        - daily_portfolio_returns_per: Daily percentage returns
        - positions: Daily position exposure (delta)
    """

    # Convert log returns to simple returns for a "factor"
    # Percentage change is given by exp(log_return) - 1
    simple_returns = np.exp(actual_df) - 1

    # Set trading direction: -1 for contrarian, 1 for normal
    direction = -1 if contrarian else 1
    
    if pnl_strategy == "naive":
        raw_positions = direction * np.sign(forecast_df)
        # Normalize so absolute positions sum to 1 each day
        row_abs_sum = raw_positions.abs().sum(axis=1).replace(0, 1)
        positions = raw_positions.div(row_abs_sum, axis=0)

    elif pnl_strategy == "weighted":
        row_abs_sum = forecast_df.abs().sum(axis=1).replace(0, 1)
        positions = direction * forecast_df.div(row_abs_sum, axis=0)

    elif pnl_strategy == "top":
        positions = pd.DataFrame(0, index=forecast_df.index, columns=forecast_df.columns)
        iter=forecast_df.shape[0]

        for i in range(iter):
            abs_val=forecast_df.iloc[i,:].abs()
            sorted=abs_val.sort_values(ascending=False)
            cutoff_number=int(abs_val.shape[0]*(1-percentile))
            threshold = sorted[cutoff_number]
            positions.iloc[i,forecast_df.iloc[i,:] > threshold] = direction
            positions.iloc[i,forecast_df.iloc[i,:]< -threshold] = -direction

        row_sums = positions.abs().sum(axis=1).replace(0, 1)
        positions = positions.div(row_sums, axis=0)

    # Calculate daily portfolio returns
    daily_pnl = positions * simple_returns
    daily_portfolio_returns_per = daily_pnl.sum(axis=1)

    # Market adjustment (default behavior as per user requirement)
    if market_adjustment:
        # Use benchmark_returns if provided, otherwise try to get from simple_returns
        if benchmark_returns is not None:
            benchmark_simple_returns = np.exp(benchmark_returns) - 1
        elif benchmark in simple_returns.columns:
            benchmark_simple_returns = simple_returns[benchmark]
        else:
            raise KeyError(f"Benchmark '{benchmark}' not found in returns data and benchmark_returns not provided. Available columns: {list(simple_returns.columns)}")
        
        if beta_neutral:
            # Calculate rolling beta of our unhedged portfolio against the benchmark
            betas = rolling_beta(daily_portfolio_returns_per, benchmark_simple_returns, window=beta_window)

            # Adjust the portfolio returns by shorting the benchmark proportional to beta
            # Hedged Return = Portfolio Return - (Beta * Benchmark Return)
            hedge_amount = (betas * benchmark_simple_returns).fillna(0)
            daily_portfolio_returns_per = daily_portfolio_returns_per - positions.sum(axis=1)*hedge_amount
        else:
            daily_portfolio_returns_per = daily_portfolio_returns_per - positions.sum(axis=1)*benchmark_simple_returns

    daily_portfolio_returns = daily_portfolio_returns_per + 1
    cumulative_returns = daily_portfolio_returns.cumprod() - 1

    # return cumulative returns, percentage returns, and the delta of the strategy
    return cumulative_returns, daily_portfolio_returns_per, positions.sum(axis=1)