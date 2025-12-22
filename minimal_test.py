"""Minimal test of caching"""
print("Step 1: Importing yfinance_cache...")
from yfinance_cache import download_cached

print("Step 2: Downloading SPY data...")
data = download_cached('SPY', '2025-03-10', '2025-03-16')
print(f"Got {len(data)} rows")

print("\nStep 3: Importing technical_retrievals...")
from technical_retrievals import get_rates, get_historical_volatility

print("Step 4: Testing get_rates...")
rate = get_rates('2025-04-15', '2025-03-15')
print(f"Rate: {rate}")

print("\nStep 5: Testing get_historical_volatility...")
vol = get_historical_volatility('SPY', '2025-03-15', lookback_days=30)
print(f"Volatility: {vol}")

print("\nStep 6: Importing get_stock_price_historical...")
from functions import get_stock_price_historical

print("Step 7: Testing get_stock_price_historical...")
price = get_stock_price_historical('SPY', '2025-03-15')
print(f"Price: ${price:.2f}")

print("\n✓ ALL TESTS PASSED!")
