"""Evaluation module for PnL calculation, backtesting, and visualization.

This module provides tools for:
- Calculating PnL from forecasted returns
- Running backtests on VAR/OR-VARX model results
- Plotting strategy comparisons and lag analysis
"""

from src.evaluation.pnl import calculate_pnl
from src.evaluation.backtest import run_backtest
from src.evaluation.plotting import (
    plot_strategy_comparison,
    plot_lag_analysis,
    plot_lag_analysis_with_volatility,
    plot_coefficient_heatmap,
    plot_coefficient_evolution_per_p,
    get_performance_metrics,
)

__all__ = [
    "calculate_pnl",
    "run_backtest",
    "plot_strategy_comparison",
    "plot_lag_analysis",
    "plot_lag_analysis_with_volatility",
    "plot_coefficient_heatmap",
    "plot_coefficient_evolution_per_p",
    "get_performance_metrics",
]
