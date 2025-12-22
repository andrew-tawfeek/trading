"""
Example: Stock Backtesting with Fourier Analysis

This script demonstrates how to run a complete stock backtesting using Fourier
signal analysis.
"""

from fourier import run_fourier_backtest, print_backtest_summary, plot_stock

# Example 1: Basic Stock Backtest
# ================================
print("="*80)
print("EXAMPLE 1: Basic Stock Backtest")
print("="*80)

results = run_fourier_backtest(
    ticker='SPY',
    start_date='2025-01-01',
    end_date='2025-12-20',
    n_harmonics=10,
    smoothing_sigma=2.0,
    overbought_threshold=5.0,
    oversold_threshold=-5.0,
    initial_capital=10000.0
)

print_backtest_summary(results)

# Example 2: Parameter Optimization
# ==================================
print("\n" + "="*80)
print("EXAMPLE 2: Parameter Optimization")
print("="*80)

best_return = -100
best_params = None

print("\nSearching for optimal parameters...")

for n_harmonics in [5, 10, 15, 20]:
    for smoothing_sigma in [0, 1, 2, 3]:
        for overbought in [4, 5, 6, 7]:
            for oversold in [-7, -6, -5, -4]:
                results = run_fourier_backtest(
                    ticker='SPY',
                    start_date='2025-01-01',
                    end_date='2025-12-20',
                    n_harmonics=n_harmonics,
                    smoothing_sigma=smoothing_sigma,
                    overbought_threshold=overbought,
                    oversold_threshold=oversold,
                    initial_capital=10000.0
                )

                total_return = results['backtest']['total_return']

                if total_return > best_return:
                    best_return = total_return
                    best_params = {
                        'n_harmonics': n_harmonics,
                        'smoothing_sigma': smoothing_sigma,
                        'overbought': overbought,
                        'oversold': oversold
                    }
                    print(f"New best: {best_return:.2f}% | Params: {best_params}")

print(f"\n{'='*80}")
print(f"OPTIMAL PARAMETERS:")
print(f"{'='*80}")
print(f"Best Return: {best_return:.2f}%")
print(f"Parameters: {best_params}")

# Example 3: Visualize Best Strategy
# ===================================
print("\n" + "="*80)
print("EXAMPLE 3: Visualizing Best Strategy")
print("="*80)

fig = plot_stock(
    ticker='SPY',
    start_date='2025-01-01',
    end_date='2025-12-20',
    fourier=True,
    n_harmonics=best_params['n_harmonics'],
    smoothing_sigma=best_params['smoothing_sigma'],
    use_candlestick=True,
    overbought_threshold=best_params['overbought'],
    oversold_threshold=best_params['oversold'],
    show_signals=True
)

print("\nOpening interactive chart in browser...")
fig.show()

# Example 4: Multi-Stock Comparison
# ==================================
print("\n" + "="*80)
print("EXAMPLE 4: Multi-Stock Comparison")
print("="*80)

tickers = ['SPY', 'QQQ', 'IWM', 'DIA']
results_comparison = {}

for ticker in tickers:
    results = run_fourier_backtest(
        ticker=ticker,
        start_date='2025-01-01',
        end_date='2025-12-20',
        n_harmonics=10,
        smoothing_sigma=2.0,
        overbought_threshold=5.0,
        oversold_threshold=-5.0,
        initial_capital=10000.0
    )
    results_comparison[ticker] = results['backtest']['total_return']

print(f"\n{'Ticker':<10} {'Return':>10}")
print("-" * 25)
for ticker, ret in sorted(results_comparison.items(), key=lambda x: x[1], reverse=True):
    print(f"{ticker:<10} {ret:>9.2f}%")

print("\n" + "="*80)
print("Examples completed!")
print("="*80)
