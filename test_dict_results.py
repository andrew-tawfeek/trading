"""
Test the new dictionary return format from run_fourier_options_backtest.
"""
from fourier import run_fourier_options_backtest, print_options_backtest_summary

print("Running Fourier options backtest...")
print("Results will be returned as a dictionary for easy analysis")
print("=" * 80)

# Run backtest with verbose=True to see progress
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
    verbose=True  # Shows progress and trade details
)

# Results is now a dictionary - easy to analyze!
print("\n" + "=" * 80)
print("ACCESSING RESULTS AS DICTIONARY")
print("=" * 80)
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Win Rate: {results['win_rate']:.1f}%")
print(f"Total Trades: {results['total_trades']}")
print(f"Average P&L: {results['average_pnl_percent']:.2f}%")
print(f"Final Capital: ${results['final_capital']:,.2f}")
print(f"Number of Signals: {len(results['signals'])}")

# Print formatted summary if desired
print("\n" + "=" * 80)
print("FORMATTED SUMMARY (optional)")
print("=" * 80)
print_options_backtest_summary(results)

# Easy to access specific trade data
print("=" * 80)
print("ACCESSING SPECIFIC TRADE DATA")
print("=" * 80)
for i, trade in enumerate(results['trades']):
    print(f"Trade {i+1}:")
    print(f"  Type: {trade['option_type'].upper()}")
    print(f"  Entry: ${trade['entry_price']:.2f} on {trade['entry_date']}")
    print(f"  Strike: ${trade['strike_price']}")
    print(f"  Greeks at entry: {trade['greeks_at_entry']}")
