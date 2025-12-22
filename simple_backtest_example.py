"""
Simple example showing the new dictionary-based backtest results.
Perfect for copying into Jupyter notebooks.
"""
from fourier import run_fourier_options_backtest, print_options_backtest_summary

# Run backtest - returns a dictionary!
results = run_fourier_options_backtest(
    ticker='SPY',
    start_date='2025-01-01',
    end_date='2025-05-01',
    n_harmonics=18,
    smoothing_sigma=0,
    overbought_threshold=9,
    oversold_threshold=-8,
    contracts_per_trade=1,
    max_positions=2,
    tick_size='1d',
    stoploss_percent=50,
    takeprofit_percent=50,
    days_to_expiry=30,
    otm_percent=2.0,
    initial_capital=10000,
    verbose=False  # Set to False for clean output
)

# Access results directly
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Win Rate: {results['win_rate']:.1f}%")
print(f"Total Trades: {results['total_trades']}")
print(f"Final Capital: ${results['final_capital']:,.2f}")

# Optionally print formatted summary
print("\n")
print_options_backtest_summary(results)
