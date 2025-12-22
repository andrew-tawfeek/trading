"""
Diagnostic script to identify bottlenecks in Fourier options backtesting.
This will instrument the code to show exactly where time is being spent.
"""
import time
from datetime import datetime, timedelta
from functools import wraps
from technical_retrievals import get_rates, get_historical_volatility
from functions import option_price_historical, greeks_historical, get_stock_price_historical

# Timing decorator
def time_it(func_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            if elapsed > 0.1:  # Only print if > 100ms
                print(f"  [{elapsed:.2f}s] {func_name}")
            return result
        return wrapper
    return decorator

# Test parameters
ticker = 'SPY'
test_date = '2025-03-15'
expiration_date = '2025-04-15'
strike_price = 500

print("=" * 80)
print("DIAGNOSTIC TEST - First-time API calls (cache misses)")
print("=" * 80)

# Test 1: Stock price
print("\n1. Testing get_stock_price_historical()...")
start = time.time()
stock_price = get_stock_price_historical(ticker, test_date)
elapsed = time.time() - start
print(f"   Result: ${stock_price:.2f}")
print(f"   Time: {elapsed:.2f}s")

# Test 2: Risk-free rate
print("\n2. Testing get_rates()...")
start = time.time()
rate = get_rates(expiration_date, test_date)
elapsed = time.time() - start
print(f"   Result: {rate:.4f}")
print(f"   Time: {elapsed:.2f}s")

# Test 3: Historical volatility
print("\n3. Testing get_historical_volatility()...")
start = time.time()
vol = get_historical_volatility(ticker, test_date, lookback_days=30)
elapsed = time.time() - start
print(f"   Result: {vol:.4f}")
print(f"   Time: {elapsed:.2f}s")

# Test 4: Option pricing
print("\n4. Testing option_price_historical()...")
start = time.time()
price = option_price_historical(ticker, expiration_date, strike_price, 'call', test_date)
elapsed = time.time() - start
print(f"   Result: ${price:.2f}")
print(f"   Time: {elapsed:.2f}s")

# Test 5: Greeks calculation
print("\n5. Testing greeks_historical()...")
start = time.time()
greeks = greeks_historical(ticker, expiration_date, strike_price, 'call', test_date)
elapsed = time.time() - start
print(f"   Result: {greeks}")
print(f"   Time: {elapsed:.2f}s")

print("\n" + "=" * 80)
print("CACHED CALLS (should be instant)")
print("=" * 80)

# Test cached calls
print("\n6. Cached stock price...")
start = time.time()
stock_price_2 = get_stock_price_historical(ticker, test_date)
elapsed = time.time() - start
print(f"   Time: {elapsed:.4f}s (should be ~0.00s)")

print("\n7. Cached rate...")
start = time.time()
rate_2 = get_rates(expiration_date, test_date)
elapsed = time.time() - start
print(f"   Time: {elapsed:.4f}s (should be ~0.00s)")

print("\n8. Cached volatility...")
start = time.time()
vol_2 = get_historical_volatility(ticker, test_date, lookback_days=30)
elapsed = time.time() - start
print(f"   Time: {elapsed:.4f}s (should be ~0.00s)")

print("\n9. Cached option price...")
start = time.time()
price_2 = option_price_historical(ticker, expiration_date, strike_price, 'call', test_date)
elapsed = time.time() - start
print(f"   Time: {elapsed:.4f}s (should be ~0.00s)")

print("\n" + "=" * 80)
print("CACHE INFO")
print("=" * 80)
print(f"\nStock price cache: {get_stock_price_historical.cache_info()}")
print(f"Rates cache: {get_rates.cache_info()}")
print(f"Volatility cache: {get_historical_volatility.cache_info()}")

print("\n" + "=" * 80)
print("SIMULATING BACKTEST LOOP (10 different dates)")
print("=" * 80)

# Simulate what happens in a backtest - 10 different dates
test_dates = [(datetime(2025, 3, 1) + timedelta(days=i*3)).strftime('%Y-%m-%d') for i in range(10)]

total_start = time.time()
for i, date in enumerate(test_dates):
    iter_start = time.time()

    # This is what happens on each signal in the backtest
    stock_price = get_stock_price_historical(ticker, date)
    rate = get_rates(expiration_date, date)
    vol = get_historical_volatility(ticker, date, lookback_days=30)
    price = option_price_historical(ticker, expiration_date, strike_price, 'call', date)
    greeks = greeks_historical(ticker, expiration_date, strike_price, 'call', date)

    iter_elapsed = time.time() - iter_start
    print(f"Iteration {i+1}/10: {iter_elapsed:.2f}s")

total_elapsed = time.time() - total_start
print(f"\nTotal time for 10 dates: {total_elapsed:.2f}s")
print(f"Average per date: {total_elapsed/10:.2f}s")
print(f"\nProjected time for 100 signals: {(total_elapsed/10)*100:.0f}s ({(total_elapsed/10)*100/60:.1f} minutes)")

print("\n" + "=" * 80)
print("FINAL CACHE STATS")
print("=" * 80)
print(f"Stock price cache: {get_stock_price_historical.cache_info()}")
print(f"Rates cache: {get_rates.cache_info()}")
print(f"Volatility cache: {get_historical_volatility.cache_info()}")
