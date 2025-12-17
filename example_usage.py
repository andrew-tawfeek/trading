"""
Example usage of the NASDAQ ticker CSV functions.

This file demonstrates how to:
1. Import ticker data from CSV
2. Get information for a specific ticker
3. Filter tickers by sector
4. Update all ticker data with latest prices
"""

from functions import (
    import_tickers_from_csv,
    get_ticker_info,
    filter_tickers_by_sector,
    update
)

# Example 1: Import all tickers from CSV
print("=" * 60)
print("Example 1: Import all tickers")
print("=" * 60)
df = import_tickers_from_csv()
print(f"\nFirst 5 tickers:")
print(df.head()[['Ticker', 'Company Name', 'Sector', 'Last Price', 'Market Cap']])

# Example 2: Get info for a specific ticker
print("\n" + "=" * 60)
print("Example 2: Get info for AAPL")
print("=" * 60)
aapl_info = get_ticker_info('AAPL')
if aapl_info:
    print(f"\nTicker: {aapl_info['Ticker']}")
    print(f"Company: {aapl_info['Company Name']}")
    print(f"Sector: {aapl_info['Sector']}")
    print(f"Last Price: ${aapl_info['Last Price']}")
    print(f"Market Cap: ${aapl_info['Market Cap']:,.0f}")
    print(f"P/E Ratio: {aapl_info['P/E Ratio']}")
    print(f"52 Week Range: ${aapl_info['52 Week Low']} - ${aapl_info['52 Week High']}")

# Example 3: Filter by sector
print("\n" + "=" * 60)
print("Example 3: Filter by Technology sector")
print("=" * 60)
tech_stocks = filter_tickers_by_sector('Technology')
print(f"\nTop 10 Technology stocks by Market Cap:")
tech_sorted = tech_stocks.sort_values('Market Cap', ascending=False)
print(tech_sorted.head(10)[['Ticker', 'Company Name', 'Last Price', 'Market Cap']])

# Example 4: Update ticker data (commented out - uncomment to run)
# WARNING: This will fetch fresh data for all tickers and may take several minutes
# print("\n" + "=" * 60)
# print("Example 4: Update all ticker data")
# print("=" * 60)
# updated_df = update()
# print("\nData has been updated with latest prices!")

print("\n" + "=" * 60)
print("Examples complete!")
print("=" * 60)
