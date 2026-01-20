# Plan: Import PnL/Plotting from old-code and Run VARX/OR-VARX Experiments

## Objective
1. Import `plot_strategy_comparison` and `calculate_pnl` from old-code
2. Create visualization script for current VARX and OR-VARX results
3. Run experiments with specified parameters

---

## Part 1: Import PnL and Plotting Functions

### Files to Import From
- `old-code/old-modules/pnl_calculator.py` → `calculate_pnl()` (refactored, no rolling_beta)
- `old-code/old-modules/core_utils.py` → `plot_strategy_comparison()`, `plot_lag_analysis()`

### Create New Module: `src/evaluation/pnl.py`
```python
# calculate_pnl(forecast_df, actual_df, strategy, percentile, ...)
#    - Converts log return forecasts to trading positions
#    - Strategies: "naive" (direction), "weighted" (magnitude), "top" (percentile)
#    - Returns: (cumulative_returns, daily_returns, positions)
#    - NOTE: beta_neutral=False always, rolling_beta code NOT imported
#    - market_adjustment=True subtracts SPY returns directly (no beta calc)
```

### Create New Module: `src/evaluation/plotting.py`
```python
# 1. plot_strategy_comparison(pnl_results, significance_level, include_spy, ...)
#    - Plots cumulative returns for all strategies
#    - Color coding: naive=blue, weighted=green, top_*=red variants, spy=purple
#
# 2. plot_lag_analysis(p_optimal, significance_level, dates, ...)
#    - Step plot showing how optimal lag varies over time
#    - X-axis: Date, Y-axis: p_optimal (1-10)
```

---

## Part 2: Bridge Current Forecasts to PnL

### Current Output Format
- `VARXResult.forecasts`: shape (n_assets, n_output_days)
- `VARXResult.to_dataframe()`: returns DataFrame (n_days × n_assets)

### Conversion Function: `src/evaluation/backtest.py`
```python
def run_backtest(result: VARXResult, actual_returns: pd.DataFrame,
                 strategies: list[str]) -> dict:
    """
    Convert VARXResult forecasts to PnL for multiple strategies.

    Returns: {strategy_name: (cumulative, daily_pct, positions)}
    """
```

---

## Part 3: Experiment Configuration

### Common Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Validation window | 21 days | For optimal p selection |
| Lookback | 766 days | 3 years (756) + 10 days offset |
| p_max | 10 | Maximum lag order |
| Assets | 9 ETFs | XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU |
| Confounder | VIX only | For OR-VARX |

### Test Period
- Data range: 2000-01-03 to 2020-12-31 (~5274 trading days)
- Test days: Total days - lookback (766) - validation (21) ≈ 4487 output days

---

## Part 4: Scripts to Create

### 1. `scripts/run_var_experiment.py`
Run Plain VARX (no confounders):
- Load ETF returns from OPCL file
- Run `fit_var()` with lookback=766, validation_days=21, p_max=10
- Convert forecasts to PnL (all strategies)
- Save results and plots

### 2. `scripts/run_orvarx_experiment.py`
Run OR-VARX with LightGBM:
- Load ETF returns + VIX confounder
- Run `fit_orvarx()` with learner_name='lgbm', same parameters
- Convert forecasts to PnL (all strategies)
- Save results and plots

### 3. `scripts/compare_strategies.py`
Side-by-side comparison:
- Load both VAR and OR-VARX results
- Plot on same figure for comparison
- Calculate performance metrics (Sharpe, max drawdown, hit rate)

---

## Part 5: Saving Mechanism

### Files Saved Per Experiment
Each experiment (VAR or OR-VARX) saves to `results/{experiment_name}/`:

| File | Format | Contents |
|------|--------|----------|
| `{name}_results.pkl` | pickle | Raw model results (forecasts, p_optimal, coefficients) |
| `{name}_pnl_results.pkl` | pickle | PnL tuples for each strategy |
| `{name}_performance.csv` | CSV | Summary metrics table |
| `{name}_lag_analysis.png` | PNG | Optimal lag over time plot |
| `{name}_strategy_comparison.png` | PNG | Cumulative returns plot |

### Performance Metrics CSV
```csv
strategy,annualized_return_pct,sharpe_ratio,total_return
naive,12.5,0.85,156.3
weighted,18.2,1.12,245.7
top_50,15.3,0.95,198.4
...
```

---

## Part 6: Plots Per Individual Experiment

### Plot 1: Lag Analysis (`plot_lag_analysis`)
Shows how optimal lag `p` varies over time for model adaptation.

```
p_optimal Values Over Time
    │
 10 ┤     ▄▄▄           ▄▄
    │    ▄   ▄         ▄  ▄
  7 ┤   ▄     ▄▄      ▄    ▄
    │  ▄        ▄    ▄
  4 ┤ ▄          ▄▄▄▄
    │▄
  1 ┼──────────────────────────────
    2003        2010        2015        2020
```

### Plot 2: Strategy Comparison (`plot_strategy_comparison`)
Shows cumulative PnL for all trading strategies + SPY benchmark.

```
PnL Comparison Across Different Strategies
    │
+300%┤                              ╱ Top 25%
     │                           ╱╱
+200%┤                        ╱╱     Weighted
     │                    ╱╱╱╱
+100%┤               ╱╱╱╱╱           Naive
     │          ╱╱╱╱
   0%┼─────────────────────────────── SPY
     │
-50% ┤
     └────────────────────────────────────
      2003       2010       2015       2020
```

---

## Part 7: Implementation Steps

### Step 1: Create evaluation module structure
```
src/evaluation/
├── __init__.py
├── pnl.py          # calculate_pnl (no rolling_beta)
├── plotting.py     # plot_strategy_comparison, plot_lag_analysis
└── backtest.py     # run_backtest bridging function
```

### Step 2: Port and adapt functions
- Copy calculate_pnl from old-code, remove beta_neutral logic
- Copy plot_strategy_comparison, plot_lag_analysis from old-code
- Adapt to work with VARXResult objects

### Step 3: Create experiment scripts
- run_var_experiment.py
- run_orvarx_experiment.py

### Step 4: Run experiments
- Plain VARX over full dataset
- OR-VARX with LightGBM over full dataset

### Step 5: Generate comparison plots
- Individual strategy plots
- VAR vs OR-VARX comparison

---

## Verification Plan
1. Run VAR experiment → verify forecasts shape matches expected output days
2. Run OR-VARX experiment → verify convergence and forecast generation
3. Check PnL calculation → cumulative returns should be reasonable (-50% to +500% range)
4. Verify plot generation → should show all strategies with proper colors/legend

---

## Confirmed Settings
- **Test period**: Full dataset (2000-2020, ~4500 output forecast days)
- **SPY benchmark**: Yes, include SPY as purple benchmark line
- **Strategies**: All (naive, weighted, top_50, top_25, top_75)

---

## Implementation Status

- [x] Step 1: Create evaluation module structure
- [x] Step 2: Port and adapt functions
- [x] Step 3: Create experiment scripts (run_var_experiment.py, run_orvarx_experiment.py)
- [ ] Step 4: Run experiments
- [ ] Step 5: Generate comparison plots
