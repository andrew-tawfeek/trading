# Trading Workspace Structure & Development Roadmap

## 📋 Table of Contents
1. [Current Architecture](#current-architecture)
2. [File Structure](#file-structure)
3. [Core Functions Reference](#core-functions-reference)
4. [Data Flow](#data-flow)
5. [Future Development Plan](#future-development-plan)
6. [Usage Examples](#usage-examples)
7. [Development Guidelines](#development-guidelines)

---

## Current Architecture

### System Overview
This is an **options backtesting and live trading platform** built on Black-Scholes pricing models with comprehensive Greeks analysis. The system supports both historical analysis and paper/live trading through Alpaca.

```
┌─────────────────────────────────────────────────────────────┐
│                    Trading Platform Stack                    │
├─────────────────────────────────────────────────────────────┤
│  Data Layer:      yfinance, Treasury API, NASDAQ CSV        │
│  Pricing Engine:  Black-Scholes (py_vollib)                 │
│  Analysis:        Greeks, Buy Scores, P/L Simulation        │
│  Execution:       Alpaca API (Paper/Live Trading)           │
│  Interface:       Python Functions + Jupyter Notebooks      │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Market Data | yfinance | Stock prices, option chains, historical data |
| Pricing Model | py_vollib | Black-Scholes Greeks & option valuation |
| Data Processing | pandas, numpy | CSV handling, calculations |
| Broker Integration | Alpaca API | Paper/live trading execution |
| Development | Jupyter | Interactive testing & exploration |

---

## File Structure

### Current Files

```
trading/
├── 📄 functions.py                    # Core engine (1420 lines)
│   ├── Options Greeks & Pricing
│   ├── Backtesting Engine
│   ├── Buy Score Analysis
│   └── Ticker Database Functions
│
├── 📄 technical_retrievals.py         # Helper functions (181 lines)
│   ├── Countdown calculations
│   ├── Risk-free rate fetching
│   └── Historical volatility
│
├── 📄 alpaca_trading.py              # Live trading (93 lines)
│   ├── Account management
│   ├── Order submission
│   └── Position tracking
│
├── 📊 nasdaq_tickers.csv             # Stock database (600+ tickers)
├── 📓 testing.ipynb                  # Development notebook
├── 📓 alpaca.ipynb                   # Trading API testing
├── 📚 options_greeks_reference.md    # Educational docs
│
├── 🔒 .env                           # API credentials (gitignored)
├── 📋 .gitignore                     # Git exclusions
└── ⚙️  .claude/settings.local.json   # IDE configuration
```

### File Responsibilities

#### **functions.py** - Main Engine
**Purpose:** Core options analysis and backtesting functionality

**Key Functions:**
- `greeks()` - Comprehensive Greeks analysis with buy scores
- `option_price()` - Current option pricing
- `options_purchase()` - Trade simulation with stop-loss/take-profit
- Stock price retrieval (current, historical, intraday)
- Ticker database management

**Dependencies:** yfinance, py_vollib, technical_retrievals

#### **technical_retrievals.py** - Calculation Helpers
**Purpose:** Low-level calculations and data retrieval

**Key Functions:**
- `countdown()` - Days to expiration
- `get_rates()` - Risk-free Treasury rates
- `get_historical_volatility()` - Volatility calculation
- Intraday volatility variants

**Dependencies:** yfinance, numpy, pandas

#### **alpaca_trading.py** - Broker Interface
**Purpose:** Live trading execution via Alpaca API

**Key Functions:**
- `get_account()` - Account info
- `submit_market_order()` - Market orders
- `submit_limit_order()` - Limit orders
- Position & order management

**Dependencies:** alpaca.trading, dotenv

---

## Core Functions Reference

### 🎯 Options Analysis

#### `greeks(ticker, strike_date, strike_price, option_type, status=False, silent=False)`
**Comprehensive Greeks analysis**

**Inputs:**
- `ticker` (str): Stock symbol (e.g., 'AAPL')
- `strike_date` (str): Expiration date 'YYYY-MM-DD'
- `strike_price` (float): Strike price
- `option_type` (str): 'call' or 'put'
- `status` (bool): Print detailed analysis
- `silent` (bool): Suppress all output

**Returns:** Dictionary with Greeks and buy score

**Output Metrics:**
- **Delta (Δ)**: Price sensitivity, ITM probability
- **Gamma (Γ)**: Delta acceleration
- **Theta (Θ)**: Time decay per day
- **Vega (ν)**: IV sensitivity
- **Rho (ρ)**: Interest rate sensitivity
- **Buy Score (0-100)**: Multi-factor recommendation

**Buy Score Ranges:**
- 75-100: STRONG BUY
- 60-74: BUY
- 50-59: NEUTRAL/SLIGHT BUY
- 40-49: NEUTRAL/SLIGHT AVOID
- 25-39: AVOID
- 0-24: STRONG AVOID

**Example:**
```python
result = greeks("AAPL", "2026-01-16", 150.0, "call", status=True)
# Returns: Buy score, Greeks, recommendations
```

---

#### `option_price(ticker, strike_date, strike_price, option_type)`
**Current market option price**

**Returns:** Current bid/ask prices from market

**Example:**
```python
price = option_price("TSLA", "2026-01-16", 200.0, "put")
```

---

#### `option_price_historical(ticker, strike_date, strike_price, option_type, historical_date, iv=None)`
**Historical option pricing using Black-Scholes**

**Use Case:** Backtest what option cost on a specific past date

**Returns:** Theoretical option price at historical date

---

### 📈 Backtesting Engine

#### `options_purchase(ticker, strike_date, strike_price, date, time, option_type, stoploss=20, takeprofit=50, iv=None)`
**Simulate option trade with realistic monitoring**

**Trade Flow:**
1. Enter position at specified date/time
2. Monitor daily (hourly for recent dates)
3. Exit on stop-loss (-20% default)
4. Exit on take-profit (+50% default)
5. Force exit at expiration

**Inputs:**
- Entry date/time
- Stop-loss % (default: 20% loss)
- Take-profit % (default: 50% gain)
- Optional IV override

**Returns:** Dictionary with:
- `entry_price` - Entry option price
- `exit_price` - Exit option price
- `exit_reason` - 'stoploss', 'takeprofit', 'expiration', 'position_open'
- `pl_percent` - P/L percentage
- `pl_dollar` - P/L in dollars per contract
- `days_held` - Position duration

**Example:**
```python
result = options_purchase(
    'AAPL', '2026-01-16', 150.0,
    '2025-12-10', '10:30', 'call',
    stoploss=20, takeprofit=50
)
print(f"P/L: {result['pl_percent']:.1f}%")
print(f"Exit: {result['exit_reason']}")
```

---

### 📊 Stock Price Functions

#### `get_stock_price_historical(ticker, date)`
**Historical closing price**

**Returns:** Stock close price on given date

---

#### `get_stock_price_intraday(ticker, date, time)`
**Intraday hourly price**

**Limitation:** Only available for last 60 days

**Returns:** Stock price at specific hour

---

### 🗃️ Ticker Database

#### `import_tickers_from_csv(csv_file)`
**Load ticker database**

**CSV Columns:** Symbol, Name, Last Sale, Net Change, % Change, Market Cap, Country, IPO Year, Volume, Sector, Industry

**Returns:** List of ticker dictionaries

---

#### `filter_tickers_by_sector(sector)`
**Filter by sector**

**Example:**
```python
tech_stocks = filter_tickers_by_sector("Technology")
```

---

### 🔧 Helper Functions (technical_retrievals.py)

#### `countdown(date)`
**Calculate days to expiration**

**Input:** Date string 'YYYY-MM-DD'

**Returns:** Float days (accounts for market hours)

---

#### `get_rates(expiration_date, historical_date=None)`
**Fetch risk-free rate for Black-Scholes**

**Returns:** Treasury rate matching option duration

---

#### `get_historical_volatility(ticker, date, lookback_days=30)`
**Calculate annualized volatility**

**Method:** Standard deviation of daily log returns

**Returns:** Annualized volatility (σ)

---

## Data Flow

### Greeks Calculation Pipeline

```
┌─────────────────┐
│  User Input     │
│  (ticker, K, T) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Data Retrieval                     │
│  - Stock price (S) via yfinance     │
│  - Option chain (IV, bid/ask)       │
│  - Risk-free rate (r) from Treasury │
│  - Time to expiration (t)           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Black-Scholes Calculation          │
│  - py_vollib.black_scholes()        │
│  - Greeks: delta, gamma, vega, etc. │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Buy Score Analysis                 │
│  - Multi-factor weighted scoring    │
│  - Risk level classification        │
│  - Recommendations (stop/target)    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Output         │
│  - Greeks dict  │
│  - Buy score    │
│  - Analysis     │
└─────────────────┘
```

### Backtesting Flow

```
Entry → Daily/Hourly Monitoring → Exit Trigger → P/L Report
  │              │                      │
  │         ┌────┴─────┐           ┌────┴────┐
  │         │ Pricing  │           │ -20% SL │
  │         │ Greeks   │           │ +50% TP │
  │         │ Tracking │           │ Expire  │
  │         └──────────┘           └─────────┘
  │
  └─ Realistic option pricing via Black-Scholes
```

---

## Future Development Plan

### Phase 1: Strategy Enhancement (Priority: HIGH)

#### 1.1 Multi-Leg Strategies
**File:** `strategies.py` (NEW)

**Functions to Add:**
- `vertical_spread()` - Bull/bear call/put spreads
- `iron_condor()` - Premium collection strategy
- `butterfly()` - Limited risk/reward
- `straddle()` - Volatility plays
- `calendar_spread()` - Time decay arbitrage

**Implementation:**
```python
def vertical_spread(ticker, strike_date, long_strike, short_strike,
                   option_type, status=False):
    """
    Analyze vertical spread strategy

    Returns:
        - Net debit/credit
        - Max profit/loss
        - Breakeven points
        - Greeks for spread position
    """
    pass
```

---

#### 1.2 Advanced Greeks
**File:** `advanced_greeks.py` (NEW)

**Functions to Add:**
- `vanna()` - Delta sensitivity to IV changes
- `charm()` - Delta decay over time
- `vomma()` - Vega sensitivity to IV
- `color()` - Gamma decay over time

**Purpose:** Second-order Greeks for sophisticated risk management

---

#### 1.3 Portfolio Greeks
**File:** `portfolio.py` (NEW)

**Functions to Add:**
- `portfolio_greeks()` - Aggregate Greeks across positions
- `portfolio_risk()` - VaR, stress testing
- `hedge_analysis()` - Suggest offsetting positions
- `correlation_matrix()` - Multi-ticker correlation

---

### Phase 2: Backtesting Improvements (Priority: HIGH)

#### 2.1 Strategy Backtester
**File:** `backtest_engine.py` (NEW)

**Features:**
- **Walk-forward optimization** - Avoid overfitting
- **Rolling window analysis** - Test across market regimes
- **Commission/slippage modeling** - Realistic costs
- **Margin requirements** - Account for buying power
- **Multiple exit strategies** - Trailing stops, technical signals

**Core Function:**
```python
def backtest_strategy(ticker, strategy_func, start_date, end_date,
                     capital=10000, params={}):
    """
    Run strategy across date range

    Returns:
        - Sharpe ratio
        - Max drawdown
        - Win rate
        - Average P/L
        - Equity curve
        - Trade log
    """
    pass
```

---

#### 2.2 Performance Metrics
**File:** `metrics.py` (NEW)

**Functions to Add:**
- `sharpe_ratio()` - Risk-adjusted return
- `sortino_ratio()` - Downside deviation
- `max_drawdown()` - Peak-to-trough decline
- `calmar_ratio()` - Return/max drawdown
- `win_rate()` - Percentage profitable
- `profit_factor()` - Gross profit / gross loss
- `expectancy()` - Average trade expectancy

---

#### 2.3 Trade Journal
**File:** `trade_log.py` (NEW)

**Database:** SQLite or CSV

**Schema:**
```python
{
    'trade_id': int,
    'ticker': str,
    'strategy': str,
    'entry_date': datetime,
    'exit_date': datetime,
    'entry_price': float,
    'exit_price': float,
    'quantity': int,
    'pl_dollar': float,
    'pl_percent': float,
    'exit_reason': str,
    'greeks_at_entry': dict,
    'greeks_at_exit': dict,
    'notes': str
}
```

**Functions:**
- `log_trade()` - Record trade
- `get_trades()` - Query by ticker/date/strategy
- `analyze_trades()` - Performance breakdown

---

### Phase 3: Data & Infrastructure (Priority: MEDIUM)

#### 3.1 Data Management
**File:** `data_manager.py` (NEW)

**Features:**
- **Local caching** - Reduce API calls
- **Data validation** - Check for gaps/errors
- **Alternative data sources** - Polygon.io, IBKR
- **Options chain storage** - Historical option chains

**Functions:**
```python
def cache_option_chain(ticker, date):
    """Store option chain snapshot"""
    pass

def get_cached_data(ticker, date):
    """Retrieve from cache if available"""
    pass
```

---

#### 3.2 Volatility Models
**File:** `volatility.py` (NEW)

**Current:** Simple 30-day historical volatility

**Enhancements:**
- **GARCH models** - Conditional heteroskedasticity
- **Implied volatility surface** - Skew/smile analysis
- **Historical IV percentiles** - Context for current IV
- **ATM IV tracking** - Cleaner IV metric

---

#### 3.3 Earnings & Events
**File:** `events.py` (NEW)

**Features:**
- **Earnings calendar** - Avoid/target earnings
- **Dividend tracking** - Adjust pricing models
- **Ex-dividend dates** - Early assignment risk
- **Economic calendar** - Fed meetings, CPI, etc.

---

### Phase 4: Live Trading (Priority: MEDIUM)

#### 4.1 Order Management
**File:** Enhance `alpaca_trading.py`

**Add:**
- `submit_bracket_order()` - Entry with stop/target
- `trailing_stop()` - Dynamic stop-loss
- `modify_order()` - Update pending orders
- `get_fills()` - Execution history

---

#### 4.2 Position Management
**File:** `position_manager.py` (NEW)

**Features:**
- **Position sizing** - Kelly criterion, fixed fractional
- **Risk per trade** - Never risk more than X%
- **Correlation limits** - Max exposure to correlated positions
- **Daily loss limits** - Circuit breaker

---

#### 4.3 Monitoring & Alerts
**File:** `monitor.py` (NEW)

**Features:**
- **Real-time Greeks tracking** - Alert on threshold breaches
- **Price alerts** - Email/SMS notifications
- **P/L tracking** - Live position updates
- **Auto-exit triggers** - Stop-loss automation

---

### Phase 5: Analysis & Optimization (Priority: LOW)

#### 5.1 Scanner
**File:** `scanner.py` (NEW)

**Features:**
- **High IV rank** - Find volatility opportunities
- **Unusual options activity** - Large volume spikes
- **Technical setups** - Support/resistance + options
- **Earnings plays** - Pre-earnings straddles

**Example:**
```python
def scan_high_iv(min_iv_percentile=80, min_volume=1000):
    """Find high IV opportunities"""
    pass
```

---

#### 5.2 Visualization
**File:** `visualization.py` (NEW)

**Charts:**
- **Equity curves** - Strategy performance over time
- **P/L distribution** - Histogram of returns
- **Greeks over time** - Delta/theta decay
- **Profit zones** - Payoff diagrams
- **IV surface** - 3D volatility smile

**Library:** matplotlib or plotly

---

#### 5.3 Machine Learning
**File:** `ml_models.py` (NEW)

**Applications:**
- **Price prediction** - LSTM for stock movement
- **IV forecasting** - Predict volatility expansion
- **Strategy selection** - Recommend strategy based on regime
- **Exit optimization** - ML-based exit timing

---

### Phase 6: Testing & Documentation (Priority: ONGOING)

#### 6.1 Unit Tests
**Directory:** `tests/`

**Files:**
- `test_greeks.py` - Validate Greeks calculations
- `test_pricing.py` - Option pricing accuracy
- `test_backtesting.py` - Trade simulation logic
- `test_strategies.py` - Multi-leg strategies

**Framework:** pytest

---

#### 6.2 Documentation
**Files:**
- `API_REFERENCE.md` - Complete function documentation
- `STRATEGY_GUIDE.md` - Strategy explanations
- `BACKTESTING_GUIDE.md` - How to backtest
- `LIVE_TRADING_GUIDE.md` - Safe live trading practices

---

## Usage Examples

### Example 1: Analyze Option
```python
from functions import greeks

# Get comprehensive analysis
result = greeks("AAPL", "2026-01-16", 150.0, "call", status=True)

# Access specific Greeks
print(f"Delta: {result['delta']:.3f}")
print(f"Theta: ${result['theta']:.2f}/day")
print(f"Buy Score: {result['buy_score']}/100")
```

---

### Example 2: Backtest Single Trade
```python
from functions import options_purchase

# Simulate trade entered on Dec 10
result = options_purchase(
    ticker='TSLA',
    strike_date='2026-01-16',
    strike_price=200.0,
    date='2025-12-10',
    time='10:30',
    option_type='call',
    stoploss=20,
    takeprofit=50
)

print(f"Entry: ${result['entry_price']:.2f}")
print(f"Exit: ${result['exit_price']:.2f}")
print(f"P/L: {result['pl_percent']:.1f}%")
print(f"Exit Reason: {result['exit_reason']}")
print(f"Days Held: {result['days_held']}")
```

---

### Example 3: Screen Multiple Stocks
```python
from functions import import_tickers_from_csv

# Load ticker database
tickers = import_tickers_from_csv("nasdaq_tickers.csv")

# Filter by price and volume
candidates = [
    [ticker['Symbol'], ticker['Last Sale']]
    for ticker in tickers
    if 3 * ticker['Last Sale'] < 120  # Affordable options
    and ticker['Volume'] > 500000      # Liquid stocks
]

# Scan for opportunities (custom function)
for symbol, price in candidates[:10]:
    result = greeks(symbol, "2026-01-16", price, "call", silent=True)
    if result['buy_score'] >= 70:
        print(f"{symbol}: Buy Score {result['buy_score']}")
```

---

### Example 4: Walk-Forward Backtest (FUTURE)
```python
from backtest_engine import backtest_strategy
from strategies import vertical_spread

# Backtest bull call spread
results = backtest_strategy(
    ticker='SPY',
    strategy_func=vertical_spread,
    start_date='2024-01-01',
    end_date='2025-12-01',
    capital=10000,
    params={
        'long_strike_delta': 0.60,
        'short_strike_delta': 0.40,
        'dte': 45
    }
)

print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_dd']:.1f}%")
print(f"Win Rate: {results['win_rate']:.1f}%")
```

---

## Development Guidelines

### Code Standards

#### Function Documentation
```python
def example_function(param1, param2, optional=None):
    """
    Brief description of function

    Args:
        param1 (type): Description
        param2 (type): Description
        optional (type, optional): Description. Defaults to None.

    Returns:
        type: Description of return value

    Example:
        >>> example_function("foo", 42)
        Expected output
    """
    pass
```

---

#### Error Handling
- Always validate inputs (ticker format, dates, strikes)
- Handle API failures gracefully (rate limits, network errors)
- Return default values or None on failure
- Log errors for debugging

---

#### Performance
- Cache API calls when possible
- Use vectorized operations (numpy/pandas)
- Avoid unnecessary loops
- Profile slow functions

---

### Git Workflow

**Branch Strategy:**
- `main` - Production-ready code
- `dev` - Integration branch
- `feature/feature-name` - New features
- `bugfix/bug-name` - Bug fixes

**Commit Messages:**
```
type: Brief description

Detailed explanation if needed

Examples:
feat: Add vertical spread strategy
fix: Correct theta calculation for deep ITM
docs: Update API reference
test: Add unit tests for greeks()
```

---

### Testing Before Live Trading

**Checklist:**
1. ✅ Backtest strategy on 1+ year historical data
2. ✅ Paper trade for 30+ days
3. ✅ Verify Greeks calculations against broker platform
4. ✅ Test stop-loss/take-profit triggers
5. ✅ Confirm commission/slippage assumptions
6. ✅ Review max drawdown and risk per trade
7. ✅ Start with small position sizes

---

### Recommended Development Order

**Week 1-2: Strategy Foundation**
1. Implement vertical spreads (strategies.py)
2. Add portfolio Greeks (portfolio.py)
3. Create basic backtesting framework

**Week 3-4: Backtesting Infrastructure**
1. Build backtest engine with walk-forward
2. Add performance metrics (Sharpe, drawdown)
3. Create trade journal database

**Week 5-6: Data & Volatility**
1. Implement data caching
2. Add GARCH volatility models
3. Integrate earnings calendar

**Week 7-8: Live Trading Prep**
1. Enhance Alpaca integration
2. Add position management
3. Build monitoring system

**Week 9-10: Analysis Tools**
1. Create scanner for opportunities
2. Add visualization charts
3. Paper trade testing

**Ongoing:**
- Write unit tests
- Document functions
- Refine based on real-world usage

---

## Current Limitations & Solutions

| Limitation | Impact | Solution |
|------------|--------|----------|
| yfinance 15-min delay | Stale prices for fast markets | Consider paid API (Polygon, IBKR) |
| Simple historical volatility | May miss regime changes | Implement GARCH models |
| No dividend modeling | Incorrect pricing for dividend stocks | Add dividend adjustments |
| Intraday data only 60 days | Limited recent backtesting | Store historical option chains |
| Single-leg only | Can't test spreads | Build multi-leg support |
| No commission modeling | Overstates returns | Add realistic cost assumptions |

---

## Resources

### Documentation
- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Options Greeks Reference](options_greeks_reference.md)
- [yfinance Docs](https://pypi.org/project/yfinance/)
- [Alpaca API Docs](https://alpaca.markets/docs/)

### Books
- *Option Volatility and Pricing* - Sheldon Natenberg
- *Options as a Strategic Investment* - Lawrence McMillan
- *Trading Options Greeks* - Dan Passarelli

### Communities
- r/options - Reddit community
- r/algotrading - Algorithmic trading
- Elite Trader Forums

---

## Quick Start Checklist

**For New Users:**
1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Set up Alpaca account (paper trading)
3. ✅ Create `.env` file with API keys
4. ✅ Test basic Greeks: `greeks("AAPL", "2026-01-16", 150, "call", status=True)`
5. ✅ Run sample backtest: `options_purchase()`
6. ✅ Explore testing.ipynb for examples

**For Development:**
1. ✅ Read this structure document
2. ✅ Choose Phase 1 feature to implement
3. ✅ Create feature branch
4. ✅ Write function + docstring
5. ✅ Add unit test
6. ✅ Test manually in notebook
7. ✅ Commit and merge to dev

---

## Appendix: Function Quick Reference

### Greeks & Pricing
| Function | Purpose | Returns |
|----------|---------|---------|
| `greeks()` | Full Greeks analysis | Dict with delta, gamma, theta, vega, rho, buy_score |
| `option_price()` | Current option price | Float (market price) |
| `option_price_historical()` | Historical pricing | Float (BS theoretical price) |

### Backtesting
| Function | Purpose | Returns |
|----------|---------|---------|
| `options_purchase()` | Simulate trade | Dict with entry, exit, P/L, reason |

### Stock Prices
| Function | Purpose | Returns |
|----------|---------|---------|
| `get_stock_price_historical()` | EOD historical price | Float |
| `get_stock_price_intraday()` | Hourly price | Float |

### Helpers
| Function | Purpose | Returns |
|----------|---------|---------|
| `countdown()` | Days to expiration | Float |
| `get_rates()` | Risk-free rate | Float |
| `get_historical_volatility()` | Volatility calculation | Float (annualized σ) |

### Database
| Function | Purpose | Returns |
|----------|---------|---------|
| `import_tickers_from_csv()` | Load tickers | List of dicts |
| `filter_tickers_by_sector()` | Sector filter | List of dicts |

---

**Last Updated:** 2025-12-20
**Version:** 1.0
**Maintainer:** Trading Platform Development Team
