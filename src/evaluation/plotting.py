"""Plotting utilities for strategy comparison and model analysis.

This module provides visualization functions for:
- Comparing cumulative returns across trading strategies
- Analyzing optimal lag selection over time
- Performance metrics calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from typing import Dict, Tuple, Optional, Union


def get_performance_metrics(
    cumulative_series: pd.Series,
    daily_returns_series: pd.Series,
) -> Tuple[float, float, float]:
    """Calculate performance metrics for a strategy.

    Args:
        cumulative_series: Cumulative returns series.
        daily_returns_series: Daily returns series (percentage change).

    Returns:
        Tuple of (annualized_return_pct, sharpe_ratio, total_return_pct):
            - annualized_return_pct: Compound annual growth rate as percentage
            - sharpe_ratio: Annualized Sharpe ratio (assuming risk-free rate = 0)
            - total_return_pct: Total return over the period as percentage
    """
    # Clean daily returns
    daily_rets = daily_returns_series.dropna()

    # Total return
    total_return = cumulative_series.iloc[-1]

    # Calculate Compound Annual Growth Rate (CAGR)
    # CAGR = (1 + total_return)^(252/n_days) - 1
    n_days = len(cumulative_series)
    ann_ret = (1 + total_return) ** (252 / n_days) - 1

    # Sharpe ratio (assuming risk-free rate = 0)
    # Sharpe = (mean_daily * 252) / (std_daily * sqrt(252))
    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    return ann_ret * 100, sharpe, total_return * 100


def plot_lag_analysis(
    p_optimal: Union[np.ndarray, pd.Series],
    significance_level: float = None,
    dates: Union[pd.DatetimeIndex, np.ndarray] = None,
    title: str = None,
    save_path: str = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (15, 6),
) -> None:
    """Plot optimal lag values over time.

    Creates a step plot showing how the optimal lag order varies across
    the test period, useful for understanding model adaptation.

    Args:
        p_optimal: Array of optimal lag values for each day.
        significance_level: Optional significance level used (for title annotation).
        dates: Date indices corresponding to p_optimal values.
            If None, uses day indices (0, 1, 2, ...).
        title: Custom title. If None, uses default title.
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot interactively.
        figsize: Figure size as (width, height) in inches.

    Example:
        >>> p_optimal = np.array([3, 5, 5, 4, 6, 3, 2, 5])
        >>> dates = pd.date_range("2010-01-01", periods=8)
        >>> plot_lag_analysis(p_optimal, significance_level=0.05, dates=dates)
    """
    # Convert to numpy if pandas Series
    if isinstance(p_optimal, pd.Series):
        p_optimal = p_optimal.values

    # Set x-axis values
    if dates is not None:
        x_values = dates
        x_label = "Date"
    else:
        x_values = np.arange(len(p_optimal))
        x_label = "Day"

    plt.figure(figsize=figsize)

    # Use step plot for cleaner visualization of discrete lag values
    plt.step(x_values, p_optimal, where="mid", linewidth=2, alpha=0.8, color="steelblue")

    # Add scatter points to emphasize individual data points
    plt.scatter(x_values, p_optimal, alpha=0.7, s=30, color="darkblue", zorder=5)

    # Fill area under the step plot
    plt.fill_between(x_values, p_optimal, step="mid", alpha=0.3, color="lightblue")

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("p_optimal Value", fontsize=12)

    # Set title
    if title is not None:
        plt.title(title, fontsize=14)
    elif significance_level is not None:
        plt.title(f"p_optimal Values Over Time\n(Significance Level={significance_level})", fontsize=14)
    else:
        plt.title("p_optimal Values Over Time", fontsize=14)

    plt.grid(True, alpha=0.3, zorder=0)

    # Set y-axis to show integer ticks (since p_optimal are lag counts)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # Format x-axis for dates
    if dates is not None and hasattr(dates, "dtype") and np.issubdtype(dates.dtype, np.datetime64):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_strategy_comparison(
    pnl_results: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    significance_level: float = None,
    include_spy: bool = True,
    market_adjusted: bool = False,
    title: str = None,
    save_path: str = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot comparison of different trading strategies.

    Creates a line plot showing cumulative returns for each strategy,
    useful for comparing performance across different trading approaches.

    Args:
        pnl_results: Dictionary of strategy results. Keys are strategy names
            (e.g., "naive", "weighted", "top_50", "spy").
            Values are tuples of (cumulative_returns, daily_returns, positions).
        significance_level: Optional significance level used (for title annotation).
        include_spy: Whether to include SPY benchmark in the plot.
        market_adjusted: Whether returns are already market-adjusted.
            If True, SPY is not shown (would be flat line at 0).
        title: Custom title. If None, uses default title.
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot interactively.
        figsize: Figure size as (width, height) in inches.

    Example:
        >>> pnl_results = {
        ...     "naive": (cumulative_naive, daily_naive, positions_naive),
        ...     "weighted": (cumulative_weighted, daily_weighted, positions_weighted),
        ...     "spy": (cumulative_spy, daily_spy, positions_spy),
        ... }
        >>> plot_strategy_comparison(pnl_results, significance_level=0.05)
    """
    # Color scheme for strategies
    colors = {
        "naive": "blue",
        "weighted": "green",
        "top_50": "red",
        "top_25": "coral",
        "top_75": "darkred",
        "spy": "purple",
    }

    # Labels for strategies
    labels = {
        "naive": "Naive (Direction)",
        "weighted": "Weighted (Magnitude)",
        "top_50": "Top 50%",
        "top_25": "Top 25%",
        "top_75": "Top 75%",
        "spy": "SPY Benchmark",
    }

    plt.figure(figsize=figsize)

    # Plot strategy results
    for strategy_name, (cumulative_returns, daily_returns, positions) in pnl_results.items():
        if strategy_name == "spy":
            continue  # Handle SPY separately

        color = colors.get(strategy_name, "black")
        label = labels.get(strategy_name, strategy_name.replace("_", " ").title())
        plt.plot(cumulative_returns.values, color=color, label=label, linewidth=2)

    # Add SPY if requested and available (and not market-adjusted)
    if include_spy and not market_adjusted and "spy" in pnl_results:
        spy_cumulative, _, _ = pnl_results["spy"]
        plt.plot(spy_cumulative.values, color=colors["spy"], label=labels["spy"], linewidth=2, linestyle="--")

    # Build title
    if title is not None:
        plot_title = title
    else:
        plot_title = "Market-Adjusted " if market_adjusted else ""
        plot_title += "PnL Comparison Across Different Strategies"
        if significance_level is not None:
            plot_title += f"\n(Significance Level={significance_level})"

    plt.title(plot_title, fontsize=14)
    plt.xlabel("Time (Trading Days)", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add horizontal line at y=0
    plt.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def print_performance_summary(
    pnl_results: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    significance_level: float = None,
) -> pd.DataFrame:
    """Print and return performance summary for all strategies.

    Args:
        pnl_results: Dictionary of strategy results (same format as plot_strategy_comparison).
        significance_level: Optional significance level used (for header).

    Returns:
        DataFrame with performance metrics for each strategy.
    """
    # Labels for strategies
    strategy_labels = {
        "naive": "Naive",
        "weighted": "Weighted",
        "top_50": "Top 50%",
        "top_25": "Top 25%",
        "top_75": "Top 75%",
        "spy": "SPY Benchmark",
    }

    if significance_level is not None:
        print(f"Results for Significance Level: {significance_level}")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Ann. Return %':>15} {'Sharpe':>10} {'Total %':>12}")
    print("-" * 60)

    results = []
    for strategy_name, (cumulative, daily_returns, _) in pnl_results.items():
        label = strategy_labels.get(strategy_name, strategy_name.title())
        ann_ret, sharpe, total_ret = get_performance_metrics(cumulative, daily_returns)

        print(f"{label:<20} {ann_ret:>15.2f} {sharpe:>10.3f} {total_ret:>12.2f}")

        results.append({
            "strategy": strategy_name,
            "label": label,
            "annualized_return_pct": ann_ret,
            "sharpe_ratio": sharpe,
            "total_return_pct": total_ret,
        })

    print("-" * 60)

    return pd.DataFrame(results)
