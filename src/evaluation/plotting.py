"""Plotting utilities for strategy comparison and model analysis.

This module provides visualization functions for:
- Comparing cumulative returns across trading strategies
- Analyzing optimal lag selection over time
- Performance metrics calculation
"""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Tuple, Optional, Union


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


def plot_lag_analysis_with_volatility(
    p_optimal: Union[np.ndarray, pd.Series],
    spy_volatility: pd.Series,
    dates: Union[pd.DatetimeIndex, np.ndarray] = None,
    significance_level: float = None,
    vol_window: int = 21,
    title: str = None,
    save_path: str = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (15, 6),
) -> None:
    """Plot optimal lag values over time with SPY realized volatility overlay.

    Creates a dual-axis plot with p_optimal as a step plot on the left axis
    and SPY realized volatility as a dashed line on the right axis.

    Args:
        p_optimal: Array of optimal lag values for each day.
        spy_volatility: Pre-computed SPY realized volatility with DatetimeIndex.
        dates: Date indices corresponding to p_optimal values.
        significance_level: Optional significance level used (for title annotation).
        vol_window: Rolling window used for volatility (for label only).
        title: Custom title. If None, uses default title.
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot interactively.
        figsize: Figure size as (width, height) in inches.
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

    fig, ax1 = plt.subplots(figsize=figsize)

    # Left axis: p_optimal step plot (matching existing style)
    step_line = ax1.step(
        x_values, p_optimal, where="mid", linewidth=2, alpha=0.8,
        color="steelblue", label=r"$p_{opt}$", zorder=3,
    )
    ax1.scatter(x_values, p_optimal, alpha=0.7, s=30, color="darkblue", zorder=5)
    ax1.fill_between(x_values, p_optimal, step="mid", alpha=0.3, color="lightblue")

    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel("p_optimal Value", fontsize=12)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3, zorder=0)

    # Right axis: SPY realized volatility
    ax2 = ax1.twinx()
    vol_line, = ax2.plot(
        spy_volatility.index, spy_volatility.values,
        linestyle="--", color="tab:orange", linewidth=1.6, alpha=0.9,
        label=f"SPY {vol_window}d Realized Vol", zorder=1,
    )
    ax2.set_ylabel("Market Realized Vol (%)", fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    # Set title
    if title is not None:
        ax1.set_title(title, fontsize=14)
    elif significance_level is not None:
        ax1.set_title(
            f"p_optimal Values Over Time\n(Significance Level={significance_level})",
            fontsize=14,
        )
    else:
        ax1.set_title("p_optimal Values Over Time", fontsize=14)

    # Format x-axis for dates
    if dates is not None and hasattr(dates, "dtype") and np.issubdtype(dates.dtype, np.datetime64):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.xticks(rotation=30, ha="right")

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


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


def plot_coefficient_heatmap(
    heatmap_df: pd.DataFrame,
    title: str = None,
    save_path: str = None,
    show_plot: bool = True,
    figsize: tuple = None,
    cmap: str = "RdBu_r",
    vmax: float = None,
    annotate: bool = True,
    lag_separators: bool = True,
) -> None:
    """Plot a heatmap of regression coefficients across lags.

    Visualizes coefficient magnitudes where rows are outcome assets and
    columns are lagged treatment assets grouped by lag order.

    Args:
        heatmap_df: DataFrame from get_coefficient_heatmap_matrix().
            Index = outcome assets, columns = lagged assets like "XLY(L1)".
        title: Custom title. If None, uses default.
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot interactively.
        figsize: Figure size as (width, height). If None, auto-sized.
        cmap: Matplotlib colormap name (default: "RdBu_r").
        vmax: Maximum absolute value for symmetric color scale.
            If None, uses max absolute value in data.
        annotate: Whether to annotate cells with values.
            Auto-disabled when columns > 45.
        lag_separators: Whether to draw vertical lines between lag groups.
    """
    n_rows, n_cols = heatmap_df.shape
    data = heatmap_df.values

    # Auto-disable annotations for wide heatmaps
    if n_cols > 45:
        annotate = False

    # Auto-size figure
    if figsize is None:
        w = max(8, n_cols * 0.7 + 2)
        h = max(4, n_rows * 0.5 + 2)
        figsize = (w, h)

    # Symmetric color scale centered at 0
    if vmax is None:
        vmax = np.abs(data).max()
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    # Axis labels
    ax.set_xticks(np.arange(n_cols))
    asset_names = [col.split("(")[0] for col in heatmap_df.columns]
    ax.set_xticklabels(asset_names, rotation=90, fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(heatmap_df.index, fontsize=10)

    # Detect lag groups from column names for separators and group labels
    n_assets = n_rows  # assume square per lag block
    n_lags = n_cols // n_assets if n_assets > 0 else 1

    if lag_separators and n_lags > 1:
        for k in range(1, n_lags):
            ax.axvline(x=k * n_assets - 0.5, color="black", linewidth=1.5)

        # Add "Lag k" group labels via secondary x-axis
        sec = ax.secondary_xaxis('top')
        centers = [k * n_assets + (n_assets - 1) / 2 for k in range(n_lags)]
        sec.set_xticks(centers, labels=[f'Lag {k+1}' for k in range(n_lags)])
        sec.tick_params('x', length=0, labelsize=9, pad=2)

    # Annotate cells
    if annotate:
        fontsize = max(5, 8 - n_cols // 15)
        for i in range(n_rows):
            for j in range(n_cols):
                val = data[i, j]
                color = "white" if np.abs(val) > vmax * 0.7 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=fontsize, color=color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coefficient", fontsize=10)

    if title is not None:
        ax.set_title(title, fontsize=13, pad=30)
    else:
        ax.set_title("Coefficient Heatmap", fontsize=13, pad=30)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_coefficient_evolution_per_p(
    per_p_coefs,
    title: str = "Coefficient Evolution",
    save_path: Optional[str] = None,
    show_plot: bool = False,
    cmap: str = "RdBu_r",
    vmax: Optional[float] = None,
) -> None:
    """Plot p* x p* grid showing genuine per-p coefficient evolution.

    Each row shows coefficients from a VAR(row) fit (not truncated VAR(p_max)).
    Each column shows a specific lag's coefficient matrix.
    Cells where lag > model order are left blank (zero).

             Lag 1      Lag 2      Lag 3     ...   Lag p*
    VAR(1)   [A_1]      [zero]     [zero]          [zero]
    VAR(2)   [A_1]      [A_2]      [zero]          [zero]
    VAR(3)   [A_1]      [A_2]      [A_3]           [zero]
    ...
    VAR(p*)  [A_1]      [A_2]      [A_3]     ...   [A_p*]

    Args:
        per_p_coefs: PerPCoefficients instance from refit functions.
        title: Overall figure title.
        save_path: Path to save the plot. If None, plot is not saved.
        show_plot: Whether to display the plot interactively.
        cmap: Matplotlib colormap name.
        vmax: Maximum absolute value for symmetric color scale.
    """
    p_star = per_p_coefs.p_star
    asset_names = per_p_coefs.asset_names
    n_assets = len(asset_names)

    if p_star == 0:
        return

    # Compute global vmax across all coefficients
    if vmax is None:
        all_vals = []
        for p, coefs in per_p_coefs.coefficients.items():
            all_vals.append(coefs.cpu().numpy().ravel())
        if all_vals:
            all_vals = np.concatenate(all_vals)
            vmax = np.abs(all_vals).max()
        else:
            vmax = 1.0
    if vmax == 0:
        vmax = 1.0

    # Figure sizing
    cell_w = max(2.0, 3.5 - 0.1 * p_star)
    cell_h = cell_w
    fig, axes = plt.subplots(
        p_star, p_star,
        figsize=(p_star * cell_w + 2.0, p_star * cell_h + 1.5),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for row in range(p_star):  # row = model order (VAR(row+1))
        model_p = row + 1
        coefs = per_p_coefs.coefficients.get(model_p)

        for col in range(p_star):  # col = lag index
            ax = axes[row, col]
            lag = col + 1

            if lag > model_p or coefs is None:
                # Blank cell: lag exceeds model order
                ax.set_facecolor('#f0f0f0')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#cccccc')
                continue

            # Extract lag block: coefs shape is (model_p, n_assets, n_assets)
            block = coefs[col].cpu().numpy()  # (n_assets, n_assets)

            im = ax.imshow(block, aspect="equal", cmap=cmap, vmin=-vmax, vmax=vmax)

            ax.set_xticks(np.arange(n_assets))
            ax.set_yticks(np.arange(n_assets))

            # X-labels: bottom row only
            if row == p_star - 1:
                ax.set_xticklabels(asset_names, rotation=90, fontsize=7)
            else:
                ax.set_xticklabels([])

            # Y-labels: leftmost column only
            if col == 0:
                ax.set_yticklabels(asset_names, fontsize=7)
            else:
                ax.set_yticklabels([])

        # Row label (left side)
        axes[row, 0].set_ylabel(f"VAR({model_p})", fontsize=9, fontweight='bold')

    # Column labels (top)
    for col in range(p_star):
        axes[0, col].set_title(f"Lag {col + 1}", fontsize=9, fontweight='bold')

    # Shared colorbar
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
        cbar.set_label("Coefficient", fontsize=10)

    fig.suptitle(title, fontsize=13)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
