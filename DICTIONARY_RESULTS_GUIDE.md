# Dictionary Results Guide

## Overview
`run_fourier_options_backtest()` now returns a **simple dictionary** instead of a dataclass, making it much easier to analyze results programmatically.

## Return Format

```python
results = {
    # Summary Statistics
    'ticker': 'SPY',
    'date_range': ('2025-01-02', '2025-04-30'),
    'initial_capital': 10000.0,
    'final_capital': 8996.09,
    'total_return': -10.04,  # Percentage

    # Trade Statistics
    'total_trades': 2,
    'winning_trades': 0,
    'losing_trades': 0,
    'win_rate': 0.0,  # Percentage
    'average_pnl_percent': 0.0,
    'average_days_held': 0.0,

    # Detailed Data
    'trades': [  # List of trade dictionaries
        {
            'entry_date': datetime(2025, 1, 23),
            'entry_index': 15,
            'option_type': 'put',
            'strike_price': 598,
            'expiration_date': '2025-02-22',
            'entry_price': 6.10,
            'entry_stock_price': 609.75,
            'signal_reason': 'overbought',
            'greeks_at_entry': {
                'delta': -0.311,
                'gamma': 0.012,
                'vega': 0.617,
                'theta': -0.153,
                'rho': -0.161
            },
            'exit_date': None,
            'exit_index': None,
            'exit_price': None,
            'exit_stock_price': None,
            'exit_reason': None,
            'pnl_percent': None,
            'pnl_dollar': None,
            'days_held': None,
            'greeks_at_exit': None
        },
        # ... more trades
    ],

    'capital_history': [  # List of snapshots
        {'date': datetime(...), 'capital': 10000.0, 'open_positions': 0},
        {'date': datetime(...), 'capital': 9390.20, 'open_positions': 1},
        # ... more snapshots
    ],

    'parameters': {  # Backtest configuration
        'contracts_per_trade': 1,
        'stoploss_percent': 50,
        'takeprofit_percent': 50,
        'days_to_expiry': 30,
        'otm_percent': 2.0,
        'max_positions': 2
    },

    # Advanced Data
    'fourier_analysis': FourierAnalysis(...),  # Full analysis object
    'signals': [SignalPoint(...), ...]  # All detected signals
}
```

## Usage Examples

### Basic Access
```python
from fourier import run_fourier_options_backtest

# Run backtest (verbose controls progress output only)
results = run_fourier_options_backtest(
    ticker='SPY',
    start_date='2025-01-01',
    end_date='2025-05-01',
    verbose=True  # Shows progress, NOT final summary
)

# Access results directly
print(f"Return: {results['total_return']:.2f}%")
print(f"Win Rate: {results['win_rate']:.1f}%")
print(f"Trades: {results['total_trades']}")
```

### Analyze Trades
```python
# Iterate through all trades
for trade in results['trades']:
    if trade['exit_reason'] == 'takeprofit':
        print(f"Winner: {trade['option_type']} ${trade['strike_price']}")
        print(f"  Profit: {trade['pnl_percent']:.1f}%")

# Get winning trades only
winners = [t for t in results['trades'] if t['pnl_dollar'] and t['pnl_dollar'] > 0]
print(f"Winners: {len(winners)}")
```

### Extract Greeks Data
```python
# Analyze entry Greeks
for trade in results['trades']:
    greeks = trade['greeks_at_entry']
    print(f"Delta: {greeks['delta']:.3f}")
    print(f"Theta: {greeks['theta']:.3f}")
```

### Capital Curve Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Convert capital history to DataFrame
df = pd.DataFrame(results['capital_history'])

# Plot capital curve
plt.plot(df['date'], df['capital'])
plt.title(f"{results['ticker']} Options Backtest")
plt.ylabel('Capital ($)')
plt.xlabel('Date')
plt.show()
```

### Print Formatted Summary (Optional)
```python
from fourier import print_options_backtest_summary

# Only print summary if you want it
print_options_backtest_summary(results)
```

### Parameter Sweeps
```python
# Easy to run multiple tests and compare
results_list = []

for n in range(10, 25, 2):
    results = run_fourier_options_backtest(
        ticker='SPY',
        start_date='2025-01-01',
        end_date='2025-05-01',
        n_harmonics=n,
        verbose=False  # Suppress output for batch runs
    )
    results_list.append({
        'n_harmonics': n,
        'total_return': results['total_return'],
        'win_rate': results['win_rate']
    })

# Find best parameters
best = max(results_list, key=lambda x: x['total_return'])
print(f"Best n_harmonics: {best['n_harmonics']}")
print(f"Return: {best['total_return']:.2f}%")
```

### Export to CSV
```python
import pandas as pd

# Convert trades to DataFrame
trades_df = pd.DataFrame(results['trades'])

# Export
trades_df.to_csv('backtest_trades.csv', index=False)
```

### Compare Multiple Tickers
```python
tickers = ['SPY', 'QQQ', 'IWM']
comparison = []

for ticker in tickers:
    results = run_fourier_options_backtest(
        ticker=ticker,
        start_date='2025-01-01',
        end_date='2025-05-01',
        verbose=False
    )
    comparison.append({
        'ticker': ticker,
        'return': results['total_return'],
        'win_rate': results['win_rate'],
        'trades': results['total_trades']
    })

# Print comparison
for c in comparison:
    print(f"{c['ticker']}: {c['return']:.2f}% return, {c['win_rate']:.1f}% win rate")
```

## Key Changes from Before

### Before (OptionsBacktestResults dataclass)
```python
results = run_fourier_options_backtest(...)
print(results.total_return)  # Dot notation
print(results.win_rate)
```

### After (Dictionary)
```python
results = run_fourier_options_backtest(...)
print(results['total_return'])  # Dictionary access
print(results['win_rate'])
```

## Verbose Parameter Behavior

```python
# verbose=True (default)
# - Shows progress bar
# - Prints trade entries/exits
# - Shows cache statistics
# - Does NOT print final summary
results = run_fourier_options_backtest(..., verbose=True)

# verbose=False
# - Silent execution
# - Perfect for parameter sweeps
# - No output at all
results = run_fourier_options_backtest(..., verbose=False)

# To print summary (optional, after backtest):
print_options_backtest_summary(results)
```

## Benefits of Dictionary Format

1. **Easy Data Access**: No need to remember dataclass attribute names
2. **JSON Serializable**: Can easily save/load results
3. **Pandas Compatible**: Direct conversion to DataFrame
4. **Flexible Analysis**: Easy to extract and manipulate data
5. **Better for Automation**: Simpler to work with in loops and batch processing

## Saving/Loading Results

```python
import json
import pickle

# Save as JSON (for human-readable storage)
# Note: Need to convert datetime objects first
results_serializable = {
    k: v for k, v in results.items()
    if k not in ['fourier_analysis', 'signals', 'capital_history', 'trades']
}
with open('results.json', 'w') as f:
    json.dump(results_serializable, f, indent=2)

# Save as pickle (preserves all data including objects)
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Load
with open('results.pkl', 'rb') as f:
    loaded_results = pickle.load(f)
```

## Quick Reference - All Dictionary Keys

- `ticker` - Stock symbol
- `date_range` - Tuple (start, end)
- `initial_capital` - Starting $
- `final_capital` - Ending $
- `total_return` - % return
- `total_trades` - Number of trades
- `winning_trades` - Count
- `losing_trades` - Count
- `win_rate` - % wins
- `average_pnl_percent` - Avg % P&L
- `average_days_held` - Avg hold time
- `trades` - List of trade dicts
- `capital_history` - List of snapshots
- `parameters` - Backtest config dict
- `fourier_analysis` - FourierAnalysis object
- `signals` - List of SignalPoint objects
