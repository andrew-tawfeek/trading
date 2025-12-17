# NASDAQ Tickers CSV Database

This project includes a comprehensive CSV database of NASDAQ-traded stocks with functions to import, query, and update the data.

## Files

- **nasdaq_tickers.csv** - CSV file containing 295 NASDAQ tickers with current market data
- **fetch_nasdaq_tickers.py** - Script to initially create the CSV file
- **functions.py** - Contains all the import/query/update functions
- **example_usage.py** - Example code showing how to use the functions

## CSV File Contents

The CSV file includes the following data for each ticker:

- **Ticker** - Stock symbol (e.g., AAPL, NVDA)
- **Company Name** - Full company name
- **Sector** - Business sector (Technology, Healthcare, etc.)
- **Industry** - Specific industry classification
- **Market Cap** - Market capitalization in dollars
- **Last Price** - Most recent closing price
- **Volume** - Trading volume from last trading day
- **52 Week High** - Highest price in last 52 weeks
- **52 Week Low** - Lowest price in last 52 weeks
- **Average Volume** - Average daily trading volume
- **P/E Ratio** - Price to earnings ratio
- **Dividend Yield** - Dividend yield percentage
- **Beta** - Stock volatility measure
- **Last Updated** - Timestamp of last data update

## Sector Breakdown

The database includes stocks from the following sectors:

- Technology: 88 stocks
- Healthcare: 51 stocks
- Consumer Cyclical: 46 stocks
- Industrials: 28 stocks
- Communication Services: 24 stocks
- Consumer Defensive: 15 stocks
- Financial Services: 12 stocks
- Utilities: 10 stocks
- Real Estate: 10 stocks
- Basic Materials: 9 stocks
- Energy: 1 stock

## Available Functions

### 1. import_tickers_from_csv(csv_file='nasdaq_tickers.csv')

Import all ticker data from the CSV file into a pandas DataFrame.

```python
from functions import import_tickers_from_csv

df = import_tickers_from_csv()
print(df.head())
```

### 2. update(csv_file='nasdaq_tickers.csv', save_backup=True)

Update all ticker data with the latest market prices and information. This function:
- Fetches current data for each ticker using yfinance
- Updates all price and volume fields
- Saves a backup of the old file (by default)
- Saves the updated data back to the CSV

```python
from functions import update

# Update all tickers (will take several minutes)
updated_df = update()
```

### 3. get_ticker_info(ticker_symbol, csv_file='nasdaq_tickers.csv')

Get detailed information for a specific ticker.

```python
from functions import get_ticker_info

aapl = get_ticker_info('AAPL')
print(f"Price: ${aapl['Last Price']}")
print(f"Market Cap: ${aapl['Market Cap']:,}")
```

### 4. filter_tickers_by_sector(sector, csv_file='nasdaq_tickers.csv')

Filter tickers by sector.

```python
from functions import filter_tickers_by_sector

tech_stocks = filter_tickers_by_sector('Technology')
print(tech_stocks[['Ticker', 'Company Name', 'Last Price']].head(10))
```

## Usage Examples

### Example 1: Get all tech stocks under $50

```python
from functions import import_tickers_from_csv

df = import_tickers_from_csv()
tech_cheap = df[(df['Sector'] == 'Technology') & (df['Last Price'] < 50)]
print(tech_cheap[['Ticker', 'Company Name', 'Last Price']])
```

### Example 2: Find highest dividend yields

```python
from functions import import_tickers_from_csv

df = import_tickers_from_csv()
high_dividend = df.nlargest(10, 'Dividend Yield')
print(high_dividend[['Ticker', 'Company Name', 'Dividend Yield', 'Last Price']])
```

### Example 3: Get all stocks in healthcare sector

```python
from functions import filter_tickers_by_sector

healthcare = filter_tickers_by_sector('Healthcare')
print(f"Found {len(healthcare)} healthcare stocks")
```

### Example 4: Update data weekly

```python
from functions import update

# Run this weekly to keep data fresh
# Creates automatic backup before updating
updated_df = update()
```

## Updating the Database

To refresh all ticker data with the latest prices:

```bash
python -c "from functions import update; update()"
```

This will:
1. Create a backup of the current CSV file
2. Fetch latest data for all tickers
3. Update prices, volumes, market caps, etc.
4. Save the updated data back to the CSV file

## Notes

- The initial CSV was created on 2025-12-16
- Some tickers may be skipped if they've been delisted or have no available data
- The update() function includes rate limiting to avoid API restrictions
- Market data is fetched from Yahoo Finance via the yfinance library
- Updating all ~300 tickers takes approximately 5-10 minutes

## Top 10 Stocks by Market Cap

1. NVDA - NVIDIA Corporation - $4.3T
2. AAPL - Apple Inc. - $4.1T
3. GOOG - Alphabet Inc. - $3.7T
4. GOOGL - Alphabet Inc. - $3.7T
5. MSFT - Microsoft Corporation - $3.5T
6. AMZN - Amazon.com, Inc. - $2.4T
7. META - Meta Platforms, Inc. - $1.7T
8. TSLA - Tesla, Inc. - $1.6T
9. AVGO - Broadcom Inc. - $1.6T
10. WMT - Walmart Inc. - $920B
