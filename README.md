# Stock & Options Backtesting Framework

A comprehensive Python-based backtesting framework for stocks and options trading using Fourier analysis and Black-Scholes pricing models.

## Overview

This framework provides powerful tools for:
- **Stock Backtesting**: Test trading strategies using Fourier signal analysis
- **Options Backtesting**: Simulate options trading with real-time Greeks calculations
- **Historical Analysis**: Price options and calculate Greeks for any historical date
- **Intraday Precision**: Support for intraday option pricing (last 60 days)
- **Data Caching**: Persistent caching to minimize API calls and improve performance
- **Live Trading Integration**: Ready-to-deploy Alpaca trading integration

## Key Features

### 1. Fourier-Based Technical Analysis
- Fit Fourier series to price data for trend extraction
- Detect overbought/oversold conditions
- Identify peaks and troughs for entry/exit signals
- Customizable harmonics and smoothing parameters

### 2. Options Pricing & Greeks
- Black-Scholes option pricing for any historical date
- Real-time Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Comprehensive risk analysis and recommendations
- Python 3.13 compatible (no numba dependency)

### 3. Intelligent Backtesting
- Realistic options trading simulation
- Stop-loss and take-profit management
- Position tracking and P&L calculations
- Win rate and performance metrics

### 4. Performance Optimization
- Persistent disk-based data caching
- LRU caching for expensive calculations
- Parallel data fetching support
- Cache statistics and monitoring

## Installation

### Prerequisites
- Python 3.9 or higher (tested up to Python 3.13)
- pip package manager

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd trading

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```
yfinance>=0.2.48
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
plotly>=5.14.0
tqdm>=4.65.0
```

## Quick Start

### 1. Basic Stock Analysis with Greeks

```python
from functions import greeks, option_price

# Get current option Greeks
result = greeks(
    ticker_symbol='AAPL',
    strike_date='2026-01-16',
    strike_price=150.0,
    option_type='call',
    status=True  # Print detailed analysis
)

print(f"Delta: {result['delta']:.4f}")
print(f"Buy Score: {result['buy_score']}/100")
print(f"Recommendation: {result['recommendation']}")
```

### 2. Historical Option Pricing

```python
from functions import option_price_historical, greeks_historical

# Price an option on a historical date
price = option_price_historical(
    ticker_symbol='AAPL',
    strike_date='2025-03-15',
    strike_price=150.0,
    option_type='call',
    historical_date='2025-01-15'
)

# Calculate Greeks for that date
greeks = greeks_historical(
    ticker_symbol='AAPL',
    strike_date='2025-03-15',
    strike_price=150.0,
    option_type='call',
    historical_date='2025-01-15',
    iv=0.25  # Optional: specify implied volatility
)
```

### 3. Simulate an Options Purchase

```python
from functions import options_purchase

# Simulate buying a call option with risk management
result = options_purchase(
    ticker_symbol='AAPL',
    strike_date='2026-01-16',
    strike_price=150.0,
    date='2025-12-10',
    time='10:30',
    option_type='call',
    stoploss=20,     # 20% stop-loss
    takeprofit=50    # 50% take-profit
)

print(f"Entry: ${result['entry_price']:.2f}")
print(f"Exit: ${result['exit_price']:.2f}")
print(f"P&L: {result['pnl_percent']:.2f}% (${result['pnl_dollar']:.2f})")
print(f"Exit Reason: {result['exit_reason']}")
```

### 4. Fourier Stock Backtesting

```python
from fourier import run_fourier_backtest, print_backtest_summary

# Run a complete Fourier-based stock backtest
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

# Print formatted summary
print_backtest_summary(results)
```

### 5. Fourier Options Backtesting

```python
from fourier import run_fourier_options_backtest, print_options_backtest_summary

# Run a complete options backtest
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

### 6. Interactive Plotting

```python
from fourier import plot_stock

# Create interactive Fourier analysis chart
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

fig.show()  # Opens in browser
```

## Project Structure

```
trading/
├── README.md                    # This file
├── DOCUMENTATION.md             # Detailed function reference
├── EXPERIMENTAL_IDEAS.md        # Research & experimental strategies
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
│
├── core/                        # Core functionality modules
│   ├── functions.py             # Options pricing & Greeks
│   ├── fourier.py               # Fourier analysis & backtesting
│   ├── technical_retrievals.py # Market data retrieval
│   ├── black_scholes_compat.py  # BS model implementation
│   ├── yfinance_cache.py        # Data caching system
│   └── option_pricing_cached.py # Cached pricing functions
│
├── examples/                    # Usage examples
│   ├── stock_backtesting.py     # Stock backtest examples
│   ├── options_backtesting.py   # Options backtest examples
│   ├── greeks_analysis.py       # Greeks analysis examples
│   ├── parameter_optimization.py # Parameter tuning
│   └── live_trading_setup.py    # Alpaca integration guide
│
├── notebooks/                   # Jupyter notebooks
│   ├── quick_start.ipynb        # Quick start tutorial
│   ├── fourier_analysis.ipynb   # Fourier deep dive
│   └── options_strategies.ipynb # Options strategies
│
├── data/                        # Data storage (gitignored)
│   └── .yfinance_cache/         # Cached market data
│
└── tests/                       # Unit tests
    ├── test_functions.py
    ├── test_fourier.py
    └── test_caching.py
```

## Core Modules

### [functions.py](core/functions.py)
Options pricing, Greeks calculation, and analysis tools.

**Key Functions:**
- `greeks()` - Calculate option Greeks with detailed analysis
- `option_price()` - Get current option prices
- `option_price_historical()` - Price options on historical dates
- `greeks_historical()` - Historical Greeks calculation
- `option_price_intraday()` - Intraday option pricing
- `options_purchase()` - Simulate option purchase with risk management

### [fourier.py](core/fourier.py)
Fourier-based technical analysis and backtesting framework.

**Key Functions:**
- `fit_fourier()` - Fit Fourier series to price data
- `analyze_fourier()` - Comprehensive Fourier analysis
- `detect_overbought_oversold()` - Signal detection
- `run_fourier_backtest()` - Stock backtesting
- `run_fourier_options_backtest()` - Options backtesting
- `plot_stock()` - Interactive visualization

### [technical_retrievals.py](core/technical_retrievals.py)
Market data retrieval and calculations.

**Key Functions:**
- `get_rates()` - Risk-free rate retrieval
- `get_historical_volatility()` - Historical volatility calculation
- `countdown()` - Days until expiration

### [yfinance_cache.py](core/yfinance_cache.py)
Persistent caching system for market data.

**Key Features:**
- File-based cache with configurable expiry
- Automatic cache invalidation
- Cache statistics and monitoring

## Parameter Guide

### Fourier Parameters

#### n_harmonics (int)
Number of Fourier harmonics to use for curve fitting.
- **Low (1-5)**: Captures only major trends, smoother curves
- **Medium (10-20)**: Balanced approach, good for most stocks
- **High (30+)**: Captures fine details, may overfit

**Recommended:**
- Stable stocks (SPY, AAPL): 10-15
- Volatile stocks: 5-10
- High-frequency strategies: 20-30

#### smoothing_sigma (float)
Gaussian smoothing parameter for the Fourier curve.
- **0**: No smoothing, raw Fourier curve
- **1-3**: Light smoothing, preserves most details
- **4-6**: Moderate smoothing, removes noise
- **7+**: Heavy smoothing, only major trends

**Recommended:**
- Day trading: 0-1
- Swing trading: 2-4
- Position trading: 5-8

#### overbought_threshold (float)
Detrended Fourier value above which generates sell signals.
- **Low (2-4)**: More frequent trades, higher risk
- **Medium (5-7)**: Balanced approach
- **High (8-10)**: Conservative, fewer trades

#### oversold_threshold (float)
Detrended Fourier value below which generates buy signals (negative).
- **Low (-2 to -4)**: More frequent trades
- **Medium (-5 to -7)**: Balanced approach
- **High (-8 to -10)**: Conservative approach

### Options Parameters

#### days_to_expiry (int)
Days until option expiration.
- **Short-term (7-14)**: Higher theta decay, more sensitive
- **Medium-term (30-45)**: Balanced risk/reward
- **Long-term (60-90)**: Lower theta decay, more time value

#### otm_percent (float)
Percentage out-of-the-money for strike selection.
- **ATM (0-1%)**: Higher delta, more expensive
- **Slightly OTM (2-5%)**: Balanced approach
- **Far OTM (5-10%)**: Cheaper, lower probability

#### stoploss_percent / takeprofit_percent (float)
Risk management thresholds.
- **Tight (10-30%)**: Quick exits, lower risk
- **Medium (30-50%)**: Standard approach
- **Wide (50-100%)**: Longer holding periods

## Performance Optimization

### Caching Strategy

The framework uses multiple layers of caching:

1. **Disk Cache**: YFinance data cached to `.yfinance_cache/`
2. **LRU Cache**: Function-level caching for expensive calculations
3. **Session Cache**: In-memory option price caching during backtests

**Cache Statistics:**
```python
from functions import get_stock_price_historical
from technical_retrievals import get_rates, get_historical_volatility

# After running backtests, check cache efficiency
print(get_stock_price_historical.cache_info())
print(get_rates.cache_info())
print(get_historical_volatility.cache_info())
```

### Pre-caching Data

For faster backtests, pre-cache data:

```python
from yfinance_cache import download_cached

# Pre-cache data for your date range
data = download_cached('SPY', '2025-01-01', '2025-12-20', interval='1d')
print(f"Cached {len(data)} days of data")
```

## Advanced Usage

### Parameter Optimization

Find optimal Fourier parameters for a stock:

```python
from fourier import run_fourier_backtest

best_return = -100
best_params = None

for n_harmonics in range(1, 30):
    for smoothing in [0, 2, 4]:
        for overbought in range(5, 10):
            for oversold in range(-10, -5):
                results = run_fourier_backtest(
                    ticker='SPY',
                    start_date='2025-01-01',
                    end_date='2025-12-20',
                    n_harmonics=n_harmonics,
                    smoothing_sigma=smoothing,
                    overbought_threshold=overbought,
                    oversold_threshold=oversold,
                    initial_capital=10000.0
                )

                total_return = results['backtest']['total_return']

                if total_return > best_return:
                    best_return = total_return
                    best_params = {
                        'n_harmonics': n_harmonics,
                        'smoothing_sigma': smoothing,
                        'overbought': overbought,
                        'oversold': oversold
                    }
                    print(f"New best: {best_return:.2f}% with {best_params}")
```

### Multi-Ticker Analysis

Scan multiple tickers for trading opportunities:

```python
from functions import check_option, greeks

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

for ticker in tickers:
    ticker_obj = yf.Ticker(ticker)
    price = ticker_obj.history(period='1d')['Close'].iloc[-1]

    # Find options near current price
    option_info = check_option(ticker, price, date='2026-01-16')

    if option_info:
        # Analyze Greeks
        call_analysis = greeks(
            ticker, '2026-01-16',
            option_info['call']['strike'],
            'call',
            status=True,
            silent=False
        )

        if call_analysis['buy_score'] >= 70:
            print(f"\n*** BUY SIGNAL: {ticker} ***")
            print(f"Buy Score: {call_analysis['buy_score']}/100")
```

### Live Trading with Alpaca

See [fourier.py](core/fourier.py) lines 1610-2011 for comprehensive Alpaca integration guide.

**Quick Setup:**
```python
from alpaca_trade_api import REST
import os

# Set environment variables
os.environ['ALPACA_API_KEY'] = 'your_key'
os.environ['ALPACA_SECRET_KEY'] = 'your_secret'

# Initialize (use paper trading first!)
api = REST(
    os.environ['ALPACA_API_KEY'],
    os.environ['ALPACA_SECRET_KEY'],
    'https://paper-api.alpaca.markets'
)

# Monitor signals and place orders
# See fourier.py for complete implementation
```

## Risk Management

### Important Considerations

1. **Backtesting Limitations**
   - Past performance ≠ future results
   - Slippage and commissions matter
   - Market conditions change

2. **Options Risks**
   - Options can expire worthless
   - High leverage = high risk
   - Theta decay accelerates near expiration

3. **Position Sizing**
   - Never risk more than 1-2% per trade
   - Limit total options allocation to 10-20% of portfolio
   - Use stop-losses religiously

4. **Paper Trading First**
   - Test strategies for at least 2 weeks
   - Verify all systems work correctly
   - Monitor for unexpected behavior

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'yfinance'`**
```bash
pip install yfinance numpy pandas scipy plotly tqdm
```

**Issue: Slow backtests**
- Pre-cache data with `download_cached()`
- Reduce date range or tick size
- Use fewer harmonics

**Issue: Rate limiting errors**
- Caching should prevent this
- If it persists, add delays between requests
- Check `.yfinance_cache/` directory exists

**Issue: `No data available` errors**
- Verify ticker symbol is correct
- Check date range is valid (market days only)
- Ensure internet connection is active

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Disclaimer

**This software is for educational purposes only.**

- Options trading involves significant risk
- Not financial advice
- Test thoroughly before live trading
- Start with paper trading
- Consult a financial advisor
- Monitor positions actively
- Past performance doesn't guarantee future results

## Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Review [DOCUMENTATION.md](DOCUMENTATION.md) for detailed API reference
- Check [EXPERIMENTAL_IDEAS.md](EXPERIMENTAL_IDEAS.md) for research topics

## Changelog

### Version 1.0.0 (2025-12-21)
- Initial release
- Stock and options backtesting
- Fourier analysis framework
- Python 3.13 support
- Comprehensive caching system
- Alpaca integration guide
