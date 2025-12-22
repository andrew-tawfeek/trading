# Comprehensive API Documentation

Complete reference documentation for all modules and functions in the Stock & Options Backtesting Framework.

## Table of Contents

- [functions.py](#functionspy)
  - [Options Pricing](#options-pricing)
  - [Greeks Calculation](#greeks-calculation)
  - [Historical Analysis](#historical-analysis)
  - [Position Simulation](#position-simulation)
  - [Ticker Management](#ticker-management)
- [fourier.py](#fourierpy)
  - [Data Classes](#data-classes)
  - [Core Fourier Functions](#core-fourier-functions)
  - [Signal Detection](#signal-detection)
  - [Backtesting](#backtesting)
  - [Visualization](#visualization)
- [technical_retrievals.py](#technical_retrievalspy)
- [yfinance_cache.py](#yfinance_cachepy)
- [black_scholes_compat.py](#black_scholes_compatpy)

---

## functions.py

Main module for options pricing, Greeks calculation, and analysis.

### Options Pricing

#### `option_price(ticker_symbol, strike_date, strike_price, option_type)`

Get the current market price of an option.

**Parameters:**
- `ticker_symbol` (str): Stock ticker (e.g., 'AAPL')
- `strike_date` (str): Expiration date in 'YYYY-MM-DD' format
- `strike_price` (float): Strike price (e.g., 150.0)
- `option_type` (str): 'call' or 'put'

**Returns:**
- `dict`: Dictionary with keys:
  - `lastPrice` (float): Last traded price
  - `bid` (float): Current bid price
  - `ask` (float): Current ask price
  - `volume` (int): Trading volume

**Example:**
```python
from functions import option_price

price_data = option_price('AAPL', '2026-01-16', 150.0, 'call')
print(f"Last Price: ${price_data['lastPrice']:.2f}")
print(f"Bid: ${price_data['bid']:.2f}")
print(f"Ask: ${price_data['ask']:.2f}")
print(f"Volume: {price_data['volume']}")
```

---

#### `option_price_historical(ticker_symbol, strike_date, strike_price, option_type, historical_date, iv=None)`

Calculate theoretical option price at a historical date using Black-Scholes-Merton model.

**Parameters:**
- `ticker_symbol` (str): Stock ticker symbol
- `strike_date` (str): Option expiration date in 'YYYY-MM-DD' format
- `strike_price` (float): Strike price
- `option_type` (str): 'call' or 'put'
- `historical_date` (str): Date to price option at in 'YYYY-MM-DD' format
- `iv` (float, optional): Implied volatility as decimal (e.g., 0.25 for 25%). If None, historical volatility will be calculated.

**Returns:**
- `float`: Theoretical option price using BSM model

**Example:**
```python
from functions import option_price_historical

# Price a call option on January 15, 2024
price = option_price_historical(
    ticker_symbol='AAPL',
    strike_date='2024-03-15',
    strike_price=150.0,
    option_type='call',
    historical_date='2024-01-15'
)
print(f"Theoretical price on 2024-01-15: ${price:.2f}")

# With custom implied volatility
price_custom_iv = option_price_historical(
    ticker_symbol='AAPL',
    strike_date='2024-03-15',
    strike_price=150.0,
    option_type='call',
    historical_date='2024-01-15',
    iv=0.30  # 30% implied volatility
)
print(f"Price with 30% IV: ${price_custom_iv:.2f}")
```

---

#### `option_price_intraday(ticker_symbol, strike_date, strike_price, option_type, historical_date, time, iv=None)`

Calculate theoretical option price at a specific intraday time.

**Parameters:**
- `ticker_symbol` (str): Stock ticker symbol
- `strike_date` (str): Option expiration date in 'YYYY-MM-DD' format
- `strike_price` (float): Strike price
- `option_type` (str): 'call' or 'put'
- `historical_date` (str): Date in 'YYYY-MM-DD' format (within last 60 days)
- `time` (str): Time in 'HH:MM' format (24-hour, Eastern Time, market hours only: 09:30-16:00)
- `iv` (float, optional): Implied volatility. If None, historical volatility will be calculated.

**Returns:**
- `float`: Theoretical option price

**Note:** Only works for last 60 days due to yfinance intraday data limitations.

**Example:**
```python
from functions import option_price_intraday

# Price an option at 2:30 PM on December 1st
price = option_price_intraday(
    ticker_symbol='AAPL',
    strike_date='2026-01-16',
    strike_price=150.0,
    option_type='call',
    historical_date='2025-12-01',
    time='14:30'  # 2:30 PM ET
)
print(f"Price at 2:30 PM: ${price:.2f}")
```

---

### Greeks Calculation

#### `greeks(ticker_symbol, strike_date, strike_price, option_type, status=False, silent=False)`

Calculate option Greeks with comprehensive analysis and trading recommendations.

**Parameters:**
- `ticker_symbol` (str): Stock ticker symbol
- `strike_date` (str): Option expiration date in 'YYYY-MM-DD' format
- `strike_price` (float): Strike price
- `option_type` (str): 'call' or 'put'
- `status` (bool, optional): If True, includes buy recommendation analysis. Default False.
- `silent` (bool, optional): If True, suppresses detailed output. Default False.

**Returns:**
- `dict`: Dictionary containing:
  - `delta` (float): Delta value (-1 to 1)
  - `gamma` (float): Gamma value
  - `vega` (float): Vega value (P&L per 1% IV change)
  - `theta` (float): Theta value (daily time decay)
  - `rho` (float): Rho value (P&L per 1% rate change)
  - `buy_score` (int, if status=True): Buy recommendation score 0-100
  - `recommendation` (str, if status=True): Buy recommendation ('STRONG BUY', 'BUY', 'NEUTRAL', 'AVOID', etc.)

**Example:**
```python
from functions import greeks

# Basic Greeks calculation
result = greeks('AAPL', '2026-01-16', 150.0, 'call')
print(f"Delta: {result['delta']:.4f}")
print(f"Gamma: {result['gamma']:.4f}")
print(f"Theta: {result['theta']:.4f}")

# With detailed analysis and recommendation
result_full = greeks(
    ticker_symbol='AAPL',
    strike_date='2026-01-16',
    strike_price=150.0,
    option_type='call',
    status=True,  # Include buy recommendation
    silent=False  # Print detailed analysis
)

print(f"\nBuy Score: {result_full['buy_score']}/100")
print(f"Recommendation: {result_full['recommendation']}")
```

**Analysis Output:**

When `status=True` and `silent=False`, the function prints comprehensive analysis including:

1. **Current Market Data**: Stock price, strike, DTE, IV, moneyness
2. **Raw Greeks Values**: All five Greeks
3. **Delta Analysis**: Probability of expiring ITM, equivalent shares, directional exposure
4. **Gamma Analysis**: Delta sensitivity, gamma risk level, distance from ATM
5. **Vega Analysis**: IV sensitivity, volatility exposure classification
6. **Theta Analysis**: Daily/weekly/monthly decay, acceleration warnings, magnitude classification
7. **Rho Analysis**: Interest rate sensitivity
8. **Risk Metrics**: Delta/theta ratio, gamma/vega ratio, composite risk score
9. **Sensitivity Scenarios**: Price move scenarios, IV change scenarios, time decay scenarios
10. **Stop-Loss & Take-Profit Recommendations**: Conservative/moderate/aggressive levels
11. **Purchase Recommendation**: Detailed scoring breakdown and final recommendation

---

#### `greeks_historical(ticker_symbol, strike_date, strike_price, option_type, historical_date, iv=None)`

Calculate option Greeks at a historical date.

**Parameters:**
- `ticker_symbol` (str): Stock ticker symbol
- `strike_date` (str): Option expiration date in 'YYYY-MM-DD' format
- `strike_price` (float): Strike price
- `option_type` (str): 'call' or 'put'
- `historical_date` (str): Date to calculate Greeks at in 'YYYY-MM-DD' format
- `iv` (float, optional): Implied volatility. If None, historical volatility will be calculated.

**Returns:**
- `dict`: Greeks dictionary with delta, gamma, vega, theta, rho

**Example:**
```python
from functions import greeks_historical

greeks_data = greeks_historical(
    ticker_symbol='AAPL',
    strike_date='2024-03-15',
    strike_price=150.0,
    option_type='call',
    historical_date='2024-01-15'
)

print(f"Historical Greeks on 2024-01-15:")
print(f"  Delta: {greeks_data['delta']:.4f}")
print(f"  Theta: {greeks_data['theta']:.4f}")
```

---

#### `greeks_intraday(ticker_symbol, strike_date, strike_price, option_type, historical_date, time, iv=None)`

Calculate option Greeks at a specific intraday time.

**Parameters:**
- Same as `option_price_intraday()`

**Returns:**
- `dict`: Greeks dictionary

**Note:** Only works for last 60 days.

**Example:**
```python
from functions import greeks_intraday

greeks = greeks_intraday(
    ticker_symbol='AAPL',
    strike_date='2026-01-16',
    strike_price=150.0,
    option_type='call',
    historical_date='2025-12-01',
    time='14:30'
)

print(f"Greeks at 2:30 PM:")
for greek, value in greeks.items():
    print(f"  {greek}: {value:.4f}")
```

---

### Historical Analysis

#### `get_stock_price_historical(ticker_symbol, date)`

Get historical stock price for a specific date.

**Parameters:**
- `ticker_symbol` (str): Stock ticker symbol
- `date` (str): Date in 'YYYY-MM-DD' format

**Returns:**
- `float`: Close price for that date

**Caching:** This function uses LRU caching (maxsize=2000) and persistent disk caching for optimal performance.

**Example:**
```python
from functions import get_stock_price_historical

price = get_stock_price_historical('AAPL', '2024-01-15')
print(f"AAPL price on 2024-01-15: ${price:.2f}")
```

---

#### `get_stock_price_intraday(ticker_symbol, date, time)`

Get intraday stock price for a specific date and time.

**Parameters:**
- `ticker_symbol` (str): Stock ticker symbol
- `date` (str): Date in 'YYYY-MM-DD' format
- `time` (str): Time in 'HH:MM' format (24-hour, Eastern Time, market hours: 09:30-16:00)

**Returns:**
- `float`: Stock price at that timestamp

**Note:** Only works for last 60 days.

**Example:**
```python
from functions import get_stock_price_intraday

price = get_stock_price_intraday('AAPL', '2025-12-01', '14:30')
print(f"AAPL at 2:30 PM: ${price:.2f}")
```

---

### Position Simulation

#### `options_purchase(ticker_symbol, strike_date, strike_price, date, time, option_type, stoploss=20, takeprofit=50, iv=None)`

Simulate buying an option and monitoring it until stop-loss, take-profit, or expiration.

**Parameters:**
- `ticker_symbol` (str): Stock ticker symbol
- `strike_date` (str): Option expiration date in 'YYYY-MM-DD' format
- `strike_price` (float): Strike price
- `date` (str): Purchase date in 'YYYY-MM-DD' format (within last 60 days for intraday)
- `time` (str): Purchase time in 'HH:MM' format (24-hour, Eastern Time, market hours)
- `option_type` (str): 'call' or 'put'
- `stoploss` (float, optional): Stop-loss percentage. Default 20%
- `takeprofit` (float, optional): Take-profit percentage. Default 50%
- `iv` (float, optional): Implied volatility. If None, calculated from history.

**Returns:**
- `dict`: Dictionary containing:
  - `entry_price` (float): Initial option price
  - `exit_price` (float): Final option price when limit triggered
  - `exit_time` (str): Time when position was closed
  - `exit_reason` (str): 'stoploss', 'takeprofit', 'expiration', or 'position_open'
  - `pnl_percent` (float): Profit/loss percentage
  - `pnl_dollar` (float): Profit/loss in dollars (per contract, $100 multiplier)
  - `days_held` (int): Number of days position was held

**Monitoring Behavior:**
- Uses intraday (hourly) monitoring for dates within last 60 days
- Uses end-of-day prices for older dates
- If monitoring reaches current date with position still open, returns with `exit_reason='position_open'`

**Example:**
```python
from functions import options_purchase

result = options_purchase(
    ticker_symbol='AAPL',
    strike_date='2026-01-16',
    strike_price=150.0,
    date='2025-12-10',
    time='10:30',
    option_type='call',
    stoploss=20,     # Exit if price drops 20%
    takeprofit=50    # Exit if price rises 50%
)

print(f"Entry: ${result['entry_price']:.2f}")
print(f"Exit: ${result['exit_price']:.2f} at {result['exit_time']}")
print(f"Reason: {result['exit_reason']}")
print(f"P&L: {result['pnl_percent']:.2f}% (${result['pnl_dollar']:.2f} per contract)")
print(f"Days Held: {result['days_held']}")
```

---

### Ticker Management

#### `import_tickers_from_csv(csv_file='nasdaq_tickers.csv')`

Import ticker data from CSV file.

**Parameters:**
- `csv_file` (str, optional): Path to the CSV file. Default 'nasdaq_tickers.csv'

**Returns:**
- `pd.DataFrame`: DataFrame containing all ticker information

**Example:**
```python
from functions import import_tickers_from_csv

df = import_tickers_from_csv()
print(f"Loaded {len(df)} tickers")
```

---

#### `update(csv_file='nasdaq_tickers.csv', save_backup=True)`

Update all ticker data in the CSV file with latest market prices and information.

**Parameters:**
- `csv_file` (str, optional): Path to the CSV file to update
- `save_backup` (bool, optional): Whether to save a backup before updating. Default True

**Returns:**
- `pd.DataFrame`: Updated DataFrame with current market data

**Updates:**
- Last Price, Volume, 52 Week High/Low
- Market Cap, P/E, Dividend Yield, Beta

**Example:**
```python
from functions import update

df = update(csv_file='nasdaq_tickers.csv', save_backup=True)
```

---

#### `get_ticker_info(ticker_symbol, csv_file='nasdaq_tickers.csv')`

Get information for a specific ticker from the CSV file.

**Parameters:**
- `ticker_symbol` (str): The ticker symbol to look up
- `csv_file` (str, optional): Path to the CSV file

**Returns:**
- `dict` or `None`: Dictionary containing all information for the ticker

**Example:**
```python
from functions import get_ticker_info

info = get_ticker_info('AAPL')
print(f"Company: {info['Name']}")
print(f"Sector: {info['Sector']}")
print(f"Price: ${info['Last Price']}")
```

---

#### `filter_tickers_by_sector(sector, csv_file='nasdaq_tickers.csv')`

Filter tickers by sector.

**Parameters:**
- `sector` (str): Sector name (e.g., 'Technology', 'Healthcare')
- `csv_file` (str, optional): Path to the CSV file

**Returns:**
- `pd.DataFrame`: DataFrame containing tickers from the specified sector

**Example:**
```python
from functions import filter_tickers_by_sector

tech_stocks = filter_tickers_by_sector('Technology')
print(f"Found {len(tech_stocks)} technology stocks")
```

---

## fourier.py

Fourier-based technical analysis and backtesting framework.

### Data Classes

#### `FourierAnalysis`

Container for Fourier analysis results.

**Attributes:**
- `dates` (np.ndarray): Date array
- `prices` (np.ndarray): Original price series
- `fourier_prices` (np.ndarray): Reconstructed Fourier curve
- `detrended_fourier` (np.ndarray): Detrended (centered) Fourier component
- `trend_coeffs` (np.ndarray): Linear trend coefficients
- `n_harmonics` (int): Number of harmonics used
- `smoothing_sigma` (float): Smoothing parameter used
- `rmse` (float): Root mean squared error

---

#### `SignalPoint`

Represents a buy/sell signal point.

**Attributes:**
- `date` (datetime): Signal date
- `index` (int): Array index
- `price` (float): Stock price at signal
- `fourier_value` (float): Fourier curve value
- `detrended_value` (float): Detrended Fourier value
- `signal_type` (str): 'buy' or 'sell'
- `reason` (str): 'oversold', 'overbought', 'peak', or 'trough'

---

#### `FourierPeak`

Represents a peak or trough in the Fourier curve.

**Attributes:**
- `date` (datetime): Peak/trough date
- `index` (int): Array index
- `value` (float): Detrended Fourier value
- `is_peak` (bool): True for peak, False for trough
- `price_at_point` (float): Stock price at peak/trough

---

#### `OptionTrade`

Represents a single option trade in backtesting.

**Attributes:**
- Entry data: `entry_date`, `entry_index`, `entry_price`, `entry_stock_price`
- Option details: `option_type`, `strike_price`, `expiration_date`
- Signal info: `signal_reason`, `greeks_at_entry`
- Exit data (optional): `exit_date`, `exit_index`, `exit_price`, `exit_stock_price`
- Performance: `exit_reason`, `pnl_percent`, `pnl_dollar`, `days_held`
- Exit Greeks: `greeks_at_exit`

---

### Core Fourier Functions

#### `fit_fourier(data, n_harmonics=10, smoothing_sigma=0.0, return_detrended=False)`

Fit a Fourier series to the data with optional smoothing.

**Parameters:**
- `data` (array-like): Time series data to fit
- `n_harmonics` (int, optional): Number of Fourier harmonics to use. Default 10
- `smoothing_sigma` (float, optional): Gaussian smoothing parameter. Higher = smoother. Default 0.0
- `return_detrended` (bool, optional): If True, also return detrended component. Default False

**Returns:**
- `np.ndarray` or `tuple`:
  - If `return_detrended=False`: Reconstructed signal
  - If `return_detrended=True`: (reconstructed signal, detrended component)

**Example:**
```python
from fourier import fit_fourier
import numpy as np

# Generate sample price data
prices = np.array([100, 102, 101, 103, 105, 104, 106, 108])

# Fit Fourier curve
fourier_curve = fit_fourier(prices, n_harmonics=3, smoothing_sigma=1.0)

# With detrended component
fourier_curve, detrended = fit_fourier(
    prices,
    n_harmonics=3,
    smoothing_sigma=1.0,
    return_detrended=True
)
```

---

#### `analyze_fourier(prices, dates, n_harmonics=10, smoothing_sigma=2.0)`

Perform comprehensive Fourier analysis on price data.

**Parameters:**
- `prices` (np.ndarray): Price time series
- `dates` (pd.DatetimeIndex): Corresponding dates
- `n_harmonics` (int, optional): Number of harmonics. Default 10
- `smoothing_sigma` (float, optional): Smoothing parameter. Default 2.0

**Returns:**
- `FourierAnalysis`: Complete analysis results

**Example:**
```python
from fourier import analyze_fourier, get_stock_data

# Get stock data
stock_data = get_stock_data('AAPL', '2025-01-01', '2025-12-20')
prices = stock_data['Close'].values
dates = stock_data.index

# Analyze
analysis = analyze_fourier(prices, dates, n_harmonics=10, smoothing_sigma=2.0)

print(f"RMSE: ${analysis.rmse:.2f}")
print(f"Harmonics: {analysis.n_harmonics}")
```

---

#### `find_fourier_peaks(analysis, prominence=0.5)`

Find peaks and troughs in the detrended Fourier curve.

**Parameters:**
- `analysis` (FourierAnalysis): Fourier analysis results
- `prominence` (float, optional): Minimum prominence for peak detection. Default 0.5

**Returns:**
- `tuple`: (peaks, troughs) where each is a list of FourierPeak objects

**Example:**
```python
from fourier import analyze_fourier, find_fourier_peaks, get_stock_data

# Get data and analyze
stock_data = get_stock_data('AAPL', '2025-01-01', '2025-12-20')
analysis = analyze_fourier(stock_data['Close'].values, stock_data.index)

# Find peaks and troughs
peaks, troughs = find_fourier_peaks(analysis, prominence=1.0)

print(f"Found {len(peaks)} peaks and {len(troughs)} troughs")

for peak in peaks[:3]:
    print(f"Peak on {peak.date}: ${peak.price_at_point:.2f}")
```

---

### Signal Detection

#### `detect_overbought_oversold(analysis, overbought_threshold, oversold_threshold)`

Detect buy/sell signals based on overbought/oversold thresholds.

**Parameters:**
- `analysis` (FourierAnalysis): Fourier analysis results
- `overbought_threshold` (float): Detrended value above which is overbought (sell signal)
- `oversold_threshold` (float): Detrended value below which is oversold (buy signal)

**Returns:**
- `list`: List of SignalPoint objects in chronological order

**Example:**
```python
from fourier import analyze_fourier, detect_overbought_oversold, get_stock_data

# Get data and analyze
stock_data = get_stock_data('AAPL', '2025-01-01', '2025-12-20')
analysis = analyze_fourier(stock_data['Close'].values, stock_data.index)

# Detect signals
signals = detect_overbought_oversold(
    analysis,
    overbought_threshold=5.0,
    oversold_threshold=-5.0
)

print(f"Detected {len(signals)} signals")
print(f"Buy signals: {len([s for s in signals if s.signal_type == 'buy'])}")
print(f"Sell signals: {len([s for s in signals if s.signal_type == 'sell'])}")

# Show first few signals
for signal in signals[:5]:
    print(f"{signal.date.strftime('%Y-%m-%d')}: {signal.signal_type.upper()} "
          f"at ${signal.price:.2f} ({signal.reason})")
```

---

#### `detect_peak_signals(analysis, prominence=0.5)`

Detect buy/sell signals based on Fourier peaks and troughs.

**Parameters:**
- `analysis` (FourierAnalysis): Fourier analysis results
- `prominence` (float, optional): Minimum prominence for peak detection. Default 0.5

**Returns:**
- `list`: List of SignalPoint objects (troughs = buy, peaks = sell)

**Example:**
```python
from fourier import analyze_fourier, detect_peak_signals, get_stock_data

stock_data = get_stock_data('AAPL', '2025-01-01', '2025-12-20')
analysis = analyze_fourier(stock_data['Close'].values, stock_data.index)

signals = detect_peak_signals(analysis, prominence=1.0)

for signal in signals:
    print(f"{signal.date.strftime('%Y-%m-%d')}: {signal.signal_type.upper()} "
          f"({signal.reason})")
```

---

### Backtesting

#### `run_fourier_backtest(ticker, start_date, end_date, n_harmonics=10, smoothing_sigma=2.0, overbought_threshold=5.0, oversold_threshold=-5.0, initial_capital=10000.0, tick_size='1d')`

Run a complete Fourier-based stock backtest.

**Parameters:**
- `ticker` (str): Stock ticker
- `start_date`, `end_date` (str): Date range in 'YYYY-MM-DD' format
- `n_harmonics` (int, optional): Number of harmonics. Default 10
- `smoothing_sigma` (float, optional): Smoothing parameter. Default 2.0
- `overbought_threshold` (float, optional): Sell signal threshold. Default 5.0
- `oversold_threshold` (float, optional): Buy signal threshold. Default -5.0
- `initial_capital` (float, optional): Starting capital. Default $10,000
- `tick_size` (str, optional): Data interval. Default '1d'

**Returns:**
- `dict`: Complete backtesting results containing:
  - `ticker`: Stock ticker
  - `date_range`: (start_date, end_date)
  - `fourier_analysis`: FourierAnalysis object
  - `signals`: List of SignalPoint objects
  - `backtest`: Dictionary with:
    - `initial_capital`, `final_value`, `total_return`
    - `trades`, `portfolio_values`, `num_trades`
  - `parameters`: Dict of backtest parameters

**Example:**
```python
from fourier import run_fourier_backtest, print_backtest_summary

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

# Access specific results
print(f"\nReturn: {results['backtest']['total_return']:.2f}%")
print(f"Trades: {results['backtest']['num_trades']}")
print(f"RMSE: ${results['fourier_analysis'].rmse:.2f}")
```

---

#### `run_fourier_options_backtest(ticker, start_date, end_date, n_harmonics=10, smoothing_sigma=2.0, overbought_threshold=5.0, oversold_threshold=-5.0, initial_capital=10000.0, contracts_per_trade=1, stoploss_percent=50.0, takeprofit_percent=50.0, days_to_expiry=30, otm_percent=2.0, max_positions=1, tick_size='1d', verbose=True)`

Run a complete Fourier-based options backtest.

**Parameters:**
- Basic parameters: Same as `run_fourier_backtest()`
- `contracts_per_trade` (int, optional): Number of contracts per trade. Default 1
- `stoploss_percent` (float, optional): Stop-loss %. Default 50%
- `takeprofit_percent` (float, optional): Take-profit %. Default 50%
- `days_to_expiry` (int, optional): Days to option expiration. Default 30
- `otm_percent` (float, optional): Out-of-the-money %. Default 2%
- `max_positions` (int, optional): Maximum open positions. Default 1
- `verbose` (bool, optional): Print progress. Default True

**Returns:**
- `dict`: Dictionary containing:
  - Basic info: `ticker`, `date_range`
  - Capital: `initial_capital`, `final_capital`, `total_return`
  - Performance: `total_trades`, `winning_trades`, `losing_trades`, `win_rate`
  - Metrics: `average_pnl_percent`, `average_days_held`
  - Data: `trades` (list of trade dicts), `capital_history`, `parameters`
  - Analysis: `fourier_analysis`, `signals`

**Example:**
```python
from fourier import run_fourier_options_backtest, print_options_backtest_summary

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

print(f"Total Return: {results['total_return']:.2f}%")
print(f"Win Rate: {results['win_rate']:.1f}%")
print_options_backtest_summary(results)
```

---

#### `print_backtest_summary(results)`

Print a formatted summary of stock backtest results.

**Parameters:**
- `results` (dict): Results from `run_fourier_backtest()`

**Example:**
```python
from fourier import run_fourier_backtest, print_backtest_summary

results = run_fourier_backtest(ticker='SPY', ...)
print_backtest_summary(results)
```

---

#### `print_options_backtest_summary(results)`

Print a formatted summary of options backtest results.

**Parameters:**
- `results` (dict): Results from `run_fourier_options_backtest()`

**Example:**
```python
from fourier import run_fourier_options_backtest, print_options_backtest_summary

results = run_fourier_options_backtest(ticker='AAPL', ...)
print_options_backtest_summary(results)
```

---

### Visualization

#### `plot_stock(ticker, start_date, end_date, tick_size='1d', fourier=False, n_harmonics=10, smoothing_sigma=2.0, use_candlestick=True, overbought_threshold=None, oversold_threshold=None, show_signals=False)`

Create an interactive plot of stock data with optional Fourier curve fitting.

**Parameters:**
- `ticker` (str): Stock ticker symbol
- `start_date`, `end_date` (str): Date range in 'YYYY-MM-DD' format
- `tick_size` (str, optional): Data interval. Default '1d'
- `fourier` (bool, optional): If True, display Fourier curves. Default False
- `n_harmonics` (int or list, optional): Harmonics to use. Can be single int or list of ints. Default 10
- `smoothing_sigma` (float, optional): Smoothing parameter. Default 2.0
- `use_candlestick` (bool, optional): Use candlestick chart. Default True
- `overbought_threshold` (float, optional): Overbought threshold for detrended plot
- `oversold_threshold` (float, optional): Oversold threshold for detrended plot
- `show_signals` (bool, optional): Show buy/sell signals on chart. Default False

**Returns:**
- `plotly.graph_objects.Figure`: Interactive plotly figure

**Example:**
```python
from fourier import plot_stock

# Basic plot
fig = plot_stock('AAPL', '2025-01-01', '2025-12-20')
fig.show()

# With Fourier analysis
fig = plot_stock(
    ticker='AAPL',
    start_date='2025-01-01',
    end_date='2025-12-20',
    fourier=True,
    n_harmonics=[5, 10, 20],  # Multiple harmonics
    smoothing_sigma=2.0,
    use_candlestick=True,
    overbought_threshold=5.0,
    oversold_threshold=-5.0,
    show_signals=True
)
fig.show()
```

**Chart Features:**
- Interactive zoom, pan, hover
- Candlestick or line chart
- Fourier curves with multiple harmonics
- Detrended plot showing oscillations
- Buy/sell signal markers
- Weekend gaps hidden
- Range selector buttons (1m, 3m, 6m, 1y, All)

---

## technical_retrievals.py

Market data retrieval and technical calculations.

### `countdown(date)`

Calculate days until a date.

**Parameters:**
- `date` (str): Target date in 'YYYY-MM-DD' format

**Returns:**
- `int`: Number of days until date

**Example:**
```python
from technical_retrievals import countdown

days = countdown('2026-01-16')
print(f"Days until expiration: {days}")
```

---

### `get_rates(expiration_date, historical_date=None)`

Determine appropriate risk-free rate based on time to expiration.

**Parameters:**
- `expiration_date` (str): Expiration date in 'YYYY-MM-DD' format
- `historical_date` (str, optional): Historical date in 'YYYY-MM-DD' format. If None, uses current date.

**Returns:**
- `float`: Risk-free rate as decimal (e.g., 0.043 for 4.3%)

**Methodology:**
- Uses Treasury yields: 13-week T-bill (0-6 months), 5-year (6-12 months), 10-year (1+ years)
- Falls back to reasonable estimates if data unavailable
- Cached with LRU cache (maxsize=1000)

**Example:**
```python
from technical_retrievals import get_rates

# Current risk-free rate for 90-day option
rate = get_rates('2026-03-31')
print(f"Current rate: {rate*100:.2f}%")

# Historical rate for specific date
rate_historical = get_rates('2024-03-31', historical_date='2024-01-15')
print(f"Historical rate on 2024-01-15: {rate_historical*100:.2f}%")
```

---

### `get_historical_volatility(ticker_symbol, date, lookback_days=30)`

Calculate historical volatility using past returns.

**Parameters:**
- `ticker_symbol` (str): Stock ticker symbol
- `date` (str): Reference date in 'YYYY-MM-DD' format
- `lookback_days` (int, optional): Number of days to look back. Default 30

**Returns:**
- `float`: Annualized historical volatility as decimal (e.g., 0.25 for 25%)

**Methodology:**
- Calculates log returns from closing prices
- Annualizes using 252 trading days
- Uses cached yfinance data
- Falls back to 30% if calculation fails

**Caching:** LRU cached (maxsize=1000) + disk cached via yfinance_cache

**Example:**
```python
from technical_retrievals import get_historical_volatility

# 30-day historical volatility
vol = get_historical_volatility('AAPL', '2024-01-15', lookback_days=30)
print(f"30-day volatility: {vol*100:.1f}%")

# 60-day volatility
vol_60 = get_historical_volatility('AAPL', '2024-01-15', lookback_days=60)
print(f"60-day volatility: {vol_60*100:.1f}%")
```

---

## yfinance_cache.py

Persistent caching system for yfinance market data.

### `YFinanceCache` Class

File-based cache for YFinance data with automatic expiration.

**Constructor:**
```python
YFinanceCache(cache_dir=".yfinance_cache", cache_expiry_hours=24)
```

**Parameters:**
- `cache_dir` (str, optional): Directory to store cache files. Default '.yfinance_cache'
- `cache_expiry_hours` (int, optional): Hours before cache expires. Default 24

**Methods:**

#### `download(ticker, start, end, interval='1d', progress=False, auto_adjust=False)`

Download stock data with caching.

**Returns:**
- `pd.DataFrame`: Stock data

#### `clear()`

Clear all cache files.

---

### Convenience Functions

#### `download_cached(ticker, start, end, interval='1d', progress=False, auto_adjust=False)`

Convenience function using global cache instance.

**Parameters:**
- `ticker` (str): Stock ticker symbol
- `start`, `end` (str): Date range in 'YYYY-MM-DD' format
- `interval` (str, optional): Data interval. Default '1d'
- `progress` (bool, optional): Show download progress. Default False
- `auto_adjust` (bool, optional): Auto adjust OHLC. Default False

**Returns:**
- `pd.DataFrame`: Stock data

**Example:**
```python
from yfinance_cache import download_cached

# Download with caching
data = download_cached('AAPL', '2025-01-01', '2025-12-20')
print(f"Downloaded {len(data)} rows")

# Second call hits cache (much faster)
data2 = download_cached('AAPL', '2025-01-01', '2025-12-20')
```

---

#### `clear_cache()`

Clear all cached data.

**Example:**
```python
from yfinance_cache import clear_cache

clear_cache()
```

---

## black_scholes_compat.py

Python 3.13-compatible Black-Scholes implementation.

### `black_scholes(flag, S, K, t, r, sigma)`

Calculate Black-Scholes option price.

**Parameters:**
- `flag` (str): 'c' for call, 'p' for put
- `S` (float): Current stock price
- `K` (float): Strike price
- `t` (float): Time to expiration in years
- `r` (float): Risk-free rate (annual)
- `sigma` (float): Volatility (annual)

**Returns:**
- `float`: Option price

**Example:**
```python
from black_scholes_compat import black_scholes

price = black_scholes('c', S=100, K=100, t=0.25, r=0.05, sigma=0.2)
print(f"Call price: ${price:.2f}")
```

---

### Greeks Functions

All Greeks functions have the same signature as `black_scholes()`.

#### `delta(flag, S, K, t, r, sigma)`
Returns option delta (-1 to 1).

#### `gamma(flag, S, K, t, r, sigma)`
Returns option gamma.

#### `vega(flag, S, K, t, r, sigma)`
Returns option vega (per 1% volatility change).

#### `theta(flag, S, K, t, r, sigma)`
Returns option theta (per day).

#### `rho(flag, S, K, t, r, sigma)`
Returns option rho (per 1% rate change).

#### `greeks(flag, S, K, t, r, sigma)`
Returns all Greeks as a dictionary.

**Example:**
```python
from black_scholes_compat import greeks

all_greeks = greeks('c', S=100, K=100, t=0.25, r=0.05, sigma=0.2)

print(f"Delta: {all_greeks['delta']:.4f}")
print(f"Gamma: {all_greeks['gamma']:.4f}")
print(f"Vega: {all_greeks['vega']:.4f}")
print(f"Theta: {all_greeks['theta']:.4f}")
print(f"Rho: {all_greeks['rho']:.4f}")
```

---

## Error Handling

### Common Exceptions

**ValueError**
- Raised when invalid parameters are provided
- Raised when historical data is unavailable
- Raised when dates are out of range

**Example Error Handling:**
```python
from functions import option_price_historical

try:
    price = option_price_historical(
        'AAPL', '2024-03-15', 150.0, 'call', '2024-01-15'
    )
except ValueError as e:
    print(f"Error: {e}")
```

---

## Best Practices

### 1. Use Caching for Performance
```python
# Pre-cache data before running multiple backtests
from yfinance_cache import download_cached

data = download_cached('SPY', '2025-01-01', '2025-12-20')
```

### 2. Check Cache Statistics
```python
from functions import get_stock_price_historical

# After backtesting
info = get_stock_price_historical.cache_info()
print(f"Cache hits: {info.hits}, misses: {info.misses}")
```

### 3. Handle Errors Gracefully
```python
try:
    result = options_purchase(...)
except ValueError as e:
    print(f"Trade simulation failed: {e}")
```

### 4. Use Verbose Mode for Debugging
```python
results = run_fourier_options_backtest(
    ticker='AAPL',
    ...,
    verbose=True  # See detailed progress
)
```

### 5. Start with Conservative Parameters
```python
# Good starting point for options backtesting
results = run_fourier_options_backtest(
    ticker='SPY',
    n_harmonics=10,
    smoothing_sigma=2.0,
    overbought_threshold=5.0,
    oversold_threshold=-5.0,
    stoploss_percent=50,
    takeprofit_percent=50,
    days_to_expiry=30,
    otm_percent=2.0,
    max_positions=1
)
```

---

## Performance Tips

1. **Use `verbose=False`** for faster backtests when you don't need detailed output
2. **Pre-cache data** for the date range you'll be testing
3. **Use larger tick_size** ('1d' instead of '1h') for faster data retrieval
4. **Limit harmonics** to 10-20 for most applications
5. **Clear cache periodically** if disk space is limited

---

## Appendix: Function Summary Table

| Function | Module | Purpose | Key Parameters |
|----------|--------|---------|----------------|
| `greeks()` | functions.py | Calculate Greeks + analysis | ticker, strike_date, strike_price, option_type |
| `option_price()` | functions.py | Get current option price | ticker, strike_date, strike_price, option_type |
| `option_price_historical()` | functions.py | Historical option pricing | + historical_date, iv |
| `options_purchase()` | functions.py | Simulate option trade | + date, time, stoploss, takeprofit |
| `run_fourier_backtest()` | fourier.py | Stock backtesting | ticker, dates, thresholds, capital |
| `run_fourier_options_backtest()` | fourier.py | Options backtesting | + contracts, stoploss, takeprofit, expiry |
| `plot_stock()` | fourier.py | Interactive visualization | ticker, dates, fourier, harmonics |
| `get_rates()` | technical_retrievals.py | Risk-free rate | expiration_date, historical_date |
| `get_historical_volatility()` | technical_retrievals.py | Historical volatility | ticker, date, lookback_days |
| `download_cached()` | yfinance_cache.py | Cached data download | ticker, start, end, interval |

---

*For examples and tutorials, see the [examples/](examples/) directory and [README.md](README.md).*
