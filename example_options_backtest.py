"""
Example: Options Backtesting with Fourier Analysis

This script demonstrates how to run options backtesting using Fourier signals.
"""

from fourier import run_fourier_options_backtest, print_options_backtest_summary

# Example 1: Basic Options Backtest
# ==================================
print("="*80)
print("EXAMPLE 1: Basic Options Backtest")
print("="*80)

results = run_fourier_options_backtest(
    ticker='AAPL',
    start_date='2024-01-01',
    end_date='2024-12-20',
    n_harmonics=10,
    smoothing_sigma=2.0,
    overbought_threshold=5.0,
    oversold_threshold=-5.0,
    contracts_per_trade=1,
    stoploss_percent=50,
    takeprofit_percent=50,
    days_to_expiry=30,
    otm_percent=2.0,
    max_positions=1,
    initial_capital=10000,
    verbose=True
)

print(f"\nTotal Return: {results['total_return']:.2f}%")
print(f"Win Rate: {results['win_rate']:.1f}%")
print_options_backtest_summary(results)

# Example 2: Conservative vs Aggressive Strategies
# =================================================
print("\n" + "="*80)
print("EXAMPLE 2: Conservative vs Aggressive Strategies")
print("="*80)

strategies = {
    'Conservative': {
        'stoploss_percent': 30,
        'takeprofit_percent': 30,
        'days_to_expiry': 45,
        'otm_percent': 5.0,
        'max_positions': 1
    },
    'Moderate': {
        'stoploss_percent': 50,
        'takeprofit_percent': 50,
        'days_to_expiry': 30,
        'otm_percent': 2.0,
        'max_positions': 2
    },
    'Aggressive': {
        'stoploss_percent': 70,
        'takeprofit_percent': 100,
        'days_to_expiry': 14,
        'otm_percent': 1.0,
        'max_positions': 3
    }
}

strategy_results = {}

for strategy_name, params in strategies.items():
    print(f"\nTesting {strategy_name} Strategy...")

    results = run_fourier_options_backtest(
        ticker='AAPL',
        start_date='2024-01-01',
        end_date='2024-12-20',
        n_harmonics=10,
        smoothing_sigma=2.0,
        overbought_threshold=5.0,
        oversold_threshold=-5.0,
        contracts_per_trade=1,
        initial_capital=10000,
        verbose=False,  # Suppress detailed output
        **params
    )

    strategy_results[strategy_name] = {
        'return': results['total_return'],
        'win_rate': results['win_rate'],
        'total_trades': results['total_trades']
    }

print(f"\n{'='*80}")
print("STRATEGY COMPARISON")
print(f"{'='*80}")
print(f"{'Strategy':<15} {'Return':>10} {'Win Rate':>10} {'Trades':>10}")
print("-" * 50)
for name, res in strategy_results.items():
    print(f"{name:<15} {res['return']:>9.2f}% {res['win_rate']:>9.1f}% {res['total_trades']:>10}")

# Example 3: Different Expiration Periods
# ========================================
print("\n" + "="*80)
print("EXAMPLE 3: Different Expiration Periods")
print("="*80)

expiration_periods = [7, 14, 30, 45, 60]
expiry_results = {}

for days in expiration_periods:
    print(f"\nTesting {days}-day expiration...")

    results = run_fourier_options_backtest(
        ticker='AAPL',
        start_date='2024-01-01',
        end_date='2024-12-20',
        n_harmonics=10,
        smoothing_sigma=2.0,
        overbought_threshold=5.0,
        oversold_threshold=-5.0,
        contracts_per_trade=1,
        stoploss_percent=50,
        takeprofit_percent=50,
        days_to_expiry=days,
        otm_percent=2.0,
        max_positions=1,
        initial_capital=10000,
        verbose=False
    )

    expiry_results[days] = results['total_return']

print(f"\n{'='*80}")
print("EXPIRATION PERIOD COMPARISON")
print(f"{'='*80}")
print(f"{'Days to Expiry':<20} {'Return':>10}")
print("-" * 35)
for days, ret in sorted(expiry_results.items()):
    print(f"{days:<20} {ret:>9.2f}%")

# Example 4: Multiple Position Sizing
# ====================================
print("\n" + "="*80)
print("EXAMPLE 4: Multiple Position Sizing")
print("="*80)

max_positions_list = [1, 2, 3, 5]
position_results = {}

for max_pos in max_positions_list:
    print(f"\nTesting max {max_pos} position(s)...")

    results = run_fourier_options_backtest(
        ticker='AAPL',
        start_date='2024-01-01',
        end_date='2024-12-20',
        n_harmonics=10,
        smoothing_sigma=2.0,
        overbought_threshold=5.0,
        oversold_threshold=-5.0,
        contracts_per_trade=1,
        stoploss_percent=50,
        takeprofit_percent=50,
        days_to_expiry=30,
        otm_percent=2.0,
        max_positions=max_pos,
        initial_capital=10000,
        verbose=False
    )

    position_results[max_pos] = {
        'return': results['total_return'],
        'trades': results['total_trades']
    }

print(f"\n{'='*80}")
print("POSITION SIZING COMPARISON")
print(f"{'='*80}")
print(f"{'Max Positions':<20} {'Return':>10} {'Total Trades':>15}")
print("-" * 50)
for max_pos, res in sorted(position_results.items()):
    print(f"{max_pos:<20} {res['return']:>9.2f}% {res['trades']:>15}")

print("\n" + "="*80)
print("Examples completed!")
print("="*80)
