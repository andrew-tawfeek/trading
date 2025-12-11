# Options Trading Reference Guide

## Table of Contents
1. [The Greeks - Quick Reference](#the-greeks---quick-reference)
2. [Other Key Statistics](#other-key-statistics)
3. [Case Study: TLT Put Options](#case-study-tlt-put-options)
4. [Python Packages for Options Data](#python-packages-for-options-data)
5. [Trading Guidelines](#trading-guidelines)

---

## The Greeks - Quick Reference

### Delta (Δ): Direction & Speed
- **Range**: 0 to ±1.00 per share (±100 per contract)
- **Calls**: Positive (0 to +1.00). Shows how much option price rises when stock goes up $1
- **Puts**: Negative (0 to -1.00). Shows how much option price rises when stock goes down $1
- **Example**: Delta of -0.50 means a $1 drop in stock = ~$50 gain per put contract

### Theta (Θ): Time Decay Cost
- Always negative for long options
- **Dollar amount lost per day** from passage of time
- **Example**: Theta of -0.05 = losing $5/day per contract
- Accelerates as expiration approaches (especially final 30 days)
- **Key insight**: Theta decay is your enemy as a buyer, friend as a seller

### Vega (ν): Volatility Sensitivity
- How much option price changes per 1% move in implied volatility
- Higher when stock gets scary/uncertain → option premiums inflate
- **Example**: Vega of 0.10 = gain $10 per contract if IV increases 1%
- Works for you in uncertain markets, against you when volatility crushes

### Gamma (Γ): Acceleration
- How much Delta changes as stock moves $1
- High gamma = delta changes rapidly (position accelerates fast)
- **Example**: Gamma of 0.08 means your delta increases by 0.08 with each $1 stock move
- Highest near-the-money, especially close to expiration

### Rho (ρ): Interest Rate Sensitivity
- Usually negligible for short-term trades
- How much option price changes per 1% change in interest rates
- More relevant for long-dated options (LEAPS)

---

## Other Key Statistics

### Implied Volatility (IV)
- **What it is**: Market's expectation of future volatility (annualized)
- **Example**: 0.1117 = 11.17% expected annual volatility
- **Higher IV** = more expensive options (both calls and puts)
- **Lower IV** = cheaper options, but less movement expected
- **Usage**: Compare to historical volatility to gauge if options are "expensive"

### Open Interest
- **What it is**: Total number of contracts currently held by all traders
- **Higher = better liquidity** (easier to enter/exit at fair prices)
- **Low open interest** = wider bid-ask spreads, harder to trade
- **Rule of thumb**: Look for >1,000 for reasonable liquidity

### Volume
- **What it is**: How many contracts traded today
- **High volume** relative to open interest = active/liquid market
- **Volume > Open Interest** suggests lots of day trading or position turnover
- **Warning**: Very low volume = difficult to get fair prices

### Break Even
- **What it is**: Stock price where you profit at expiration
- **For calls**: Strike + Premium paid
- **For puts**: Strike - Premium paid
- **Must account for premium** - not just about being in-the-money

### Break Even Rate
- **What it is**: Percentage move needed in underlying to break even
- Measures how "realistic" your profit target is
- Negative number for puts (stock must fall), positive for calls (stock must rise)

### Bid-Ask Spread (Mid Price)
- **Mid**: Midpoint between bid and ask
- Use this for estimating "fair value"
- **Wide spreads** = illiquid, expensive to trade
- Try to trade near mid for best price

---

## Case Study: TLT Put Options

### Position Details
- **Underlying**: TLT (iShares 20+ Year Treasury Bond ETF)
- **Position**: 3 contracts of Dec 25 $100 Put
- **Entry Price**: $0.16 per share ($48.09 total premium for 3 contracts)
- **Account Size**: $200
- **Position Size**: 24% of account (high risk)
- **Expiration**: December 25, 2025 (~14 days remaining)

### Greeks Analysis

| Greek | Value | What It Means |
|-------|-------|---------------|
| **Delta** | -0.1392 | For every $1 TLT drops, gain ~$13.92 per contract ($41.76 total) |
| **Theta** | -0.0125 | Losing $1.25/day per contract = **$3.75/day total** |
| **Vega** | 0.0479 | If IV increases 1%, gain $4.79 per contract |
| **Gamma** | 0.0960 | Delta accelerates by 0.096 for each $1 move in TLT |

### Other Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Implied Vol** | 11.17% | Very low - TLT is expected to be calm |
| **Open Interest** | 17,371 | Good liquidity |
| **Volume** | 16.52K | Active trading today |
| **Break Even** | $85.85 | TLT must fall to $85.85 by expiration |
| **Break Even Rate** | -3.2% | TLT needs to fall 3.2% to profit |
| **Current TLT** | ~$88.69 | About $2.84 above break even |

### The Reality Check

**Theta Decay Analysis:**
- Daily theta loss: $3.75
- Days to expiration: 14
- **Total theta decay**: ~$52.50 over remaining life
- Current premium paid: $48.09
- **Problem**: Theta will consume more than your entire investment!

**What You Need:**
1. **Quick move down**: TLT needs to drop $2-3 quickly (2-3% move)
2. **OR volatility spike**: IV increase would help offset theta
3. **Ideally both**: Fast drop + market uncertainty

**Risk Assessment:**
- ⚠️ Position size too large (24% of account)
- ⚠️ Theta decay very aggressive (only 14 days left)
- ⚠️ Low IV means limited vega help
- ✓ Good liquidity (can exit easily)
- ✓ Delta reasonable for directional bet

### Stop-Loss and Take-Profit Recommendations

**Conservative Approach:**
- Stop Loss: -20% ($0.13) = -$9.60 loss
- Take Profit: +40% ($0.22) = +$18.00 gain
- Risk/Reward: 1:1.87

**Time-Based Exit:**
- If no meaningful move in 3-4 days, exit regardless
- Theta will destroy value even if TLT slightly down

### Lessons from This Trade

1. **Position Sizing**: Never risk >15% of account on single options trade
2. **Time Selection**: 30-45 days minimum for breathing room
3. **Theta Awareness**: Calculate total theta cost before entering
4. **Break Even Reality**: TLT must move 3.2% in 14 days (unlikely for bonds)
5. **IV Context**: Low IV = cheap options but also quiet markets

---

## Python Packages for Options Data

### 1. **yfinance** (Free, Easy)
```python
import yfinance as yf

# Get options chain
ticker = yf.Ticker("TLT")
options = ticker.options  # List of expiration dates
chain = ticker.option_chain('2025-12-25')

# Access puts
puts = chain.puts
print(puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 
            'openInterest', 'impliedVolatility']])

# Greeks are NOT provided by yfinance
```
**Pros**: Free, simple, no API key needed
**Cons**: No Greeks calculation, delayed data, limited historical data

### 2. **cboe-data** (Free, Official)
```python
# For VIX and CBOE-specific products
import requests

# Get delayed quotes
url = "https://cdn.cboe.com/api/global/delayed_quotes/options/TLT.json"
response = requests.get(url)
data = response.json()
```
**Pros**: Free, official CBOE data
**Cons**: Delayed (15 min), no Greeks

### 3. **py_vollib** (Greeks Calculation)
```python
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho

# You provide the inputs
S = 88.69  # Stock price
K = 100    # Strike
t = 14/365 # Time to expiration (years)
r = 0.05   # Risk-free rate
sigma = 0.1117  # Implied volatility

# Calculate Greeks
d = delta('p', S, K, t, r, sigma)  # 'p' for put, 'c' for call
g = gamma('p', S, K, t, r, sigma)
v = vega('p', S, K, t, r, sigma)
th = theta('p', S, K, t, r, sigma)
rh = rho('p', S, K, t, r, sigma)

print(f"Delta: {d:.4f}")
print(f"Gamma: {g:.4f}")
print(f"Vega: {v:.4f}")
print(f"Theta: {th:.4f}")
print(f"Rho: {rh:.4f}")
```
**Pros**: Accurate Greeks calculation
**Cons**: You need to provide IV and other inputs

### 4. **QuantLib** (Advanced)
```python
import QuantLib as ql

# Setup
calculation_date = ql.Date(11, 12, 2025)
ql.Settings.instance().evaluationDate = calculation_date

# Option parameters
option_type = ql.Option.Put
strike = 100
expiry_date = ql.Date(25, 12, 2025)

# Market data
spot = 88.69
volatility = 0.1117
risk_free_rate = 0.05
dividend_yield = 0.0

# Create option
payoff = ql.PlainVanillaPayoff(option_type, strike)
exercise = ql.EuropeanExercise(expiry_date)
option = ql.VanillaOption(payoff, exercise)

# Setup Black-Scholes process
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
volatility_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(calculation_date, ql.NullCalendar(), volatility, ql.Actual365Fixed())
)
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, ql.Actual365Fixed())
)
dividend_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, dividend_yield, ql.Actual365Fixed())
)

process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, volatility_handle)
engine = ql.AnalyticEuropeanEngine(process)
option.setPricingEngine(engine)

# Get Greeks
print(f"Option Price: {option.NPV():.4f}")
print(f"Delta: {option.delta():.4f}")
print(f"Gamma: {option.gamma():.4f}")
print(f"Vega: {option.vega():.4f}")
print(f"Theta: {option.theta():.4f}")
print(f"Rho: {option.rho():.4f}")
```
**Pros**: Industry-standard, accurate, feature-rich
**Cons**: Complex setup, steep learning curve

### 5. **Schwab/TD Ameritrade API** (Best for Live Data)
```python
# Requires account and API access
# TD Ameritrade example (being migrated to Schwab)
import requests

# After OAuth setup
headers = {'Authorization': f'Bearer {access_token}'}
url = f'https://api.tdameritrade.com/v1/marketdata/chains'
params = {
    'symbol': 'TLT',
    'contractType': 'PUT',
    'includeQuotes': 'TRUE',
    'strategy': 'SINGLE',
    'range': 'OTM'
}

response = requests.get(url, headers=headers, params=params)
options_data = response.json()

# Greeks included in response!
```
**Pros**: Real-time data, Greeks included, official broker API
**Cons**: Requires brokerage account, API approval

### 6. **Interactive Brokers API (ibapi)** (Professional)
```python
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        
    def tickOptionComputation(self, reqId, tickType, tickAttrib,
                             impliedVol, delta, optPrice, pvDividend,
                             gamma, vega, theta, undPrice):
        print(f"Delta: {delta}, Gamma: {gamma}, Vega: {vega}, Theta: {theta}")

app = IBapi()
app.connect('127.0.0.1', 7497, 123)

# Define contract
contract = Contract()
contract.symbol = "TLT"
contract.secType = "OPT"
contract.exchange = "SMART"
contract.currency = "USD"
contract.lastTradeDateOrContractMonth = "20251225"
contract.strike = 100
contract.right = "P"

app.reqMktData(1, contract, "", False, False, [])
```
**Pros**: Real-time, professional-grade, Greeks included
**Cons**: Requires IB account, complex API, TWS/Gateway must run

### 7. **Tradier API** (Developer-Friendly)
```python
import requests

headers = {
    'Authorization': f'Bearer {access_token}',
    'Accept': 'application/json'
}

# Get options chain with Greeks
url = 'https://api.tradier.com/v1/markets/options/chains'
params = {
    'symbol': 'TLT',
    'expiration': '2025-12-25',
    'greeks': 'true'
}

response = requests.get(url, headers=headers, params=params)
data = response.json()

# Greeks included in response
for option in data['options']['option']:
    if option['option_type'] == 'put' and option['strike'] == 100:
        print(f"Delta: {option['greeks']['delta']}")
        print(f"Gamma: {option['greeks']['gamma']}")
        print(f"Theta: {option['greeks']['theta']}")
        print(f"Vega: {option['greeks']['vega']}")
```
**Pros**: Easy API, Greeks included, good documentation
**Cons**: Requires paid subscription for real-time

### Recommended Approach

**For Learning/Paper Trading:**
```python
# Combine yfinance + py_vollib
import yfinance as yf
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta

ticker = yf.Ticker("TLT")
chain = ticker.option_chain('2025-12-25')
put = chain.puts[chain.puts['strike'] == 100].iloc[0]

# Extract data
S = ticker.info['currentPrice']
K = put['strike']
IV = put['impliedVolatility']
t = 14/365  # Calculate from expiration

# Calculate Greeks
d = delta('p', S, K, t, 0.05, IV)
g = gamma('p', S, K, t, 0.05, IV)
v = vega('p', S, K, t, 0.05, IV)
th = theta('p', S, K, t, 0.05, IV)
```

**For Real Trading:**
- Use your broker's API (Schwab, Interactive Brokers, Tradier)
- Greeks are calculated and provided in real-time
- No need to compute manually

---

## Trading Guidelines

### Before Entering a Trade

**1. Check Liquidity**
- Open Interest > 1,000 (minimum)
- Volume > 100 today
- Bid-Ask spread < 5% of option price

**2. Calculate Costs**
- Total theta decay over holding period
- Break even percentage required
- Maximum potential loss

**3. Position Sizing**
- Risk no more than 2-5% per trade
- For $200 account: $4-10 per trade (not $48!)
- Account for total theta cost, not just entry price

**4. Time Selection**
- Minimum 30 days to expiration
- Ideal: 45-60 days for theta breathing room
- Avoid last 2 weeks unless day trading

### While in Position

**Monitor Daily:**
- Delta: Is underlying moving your direction?
- Theta: How much am I losing per day?
- IV: Is volatility helping or hurting?
- Break even: How much room do I have?

**Exit Signals:**
- Thesis invalidated (price moved opposite direction)
- Theta exceeding gains (time decay winning)
- Hit stop loss or take profit
- 3-4 days with no progress toward target

### Position Management

**Stop Loss Rules:**
- Options: -20% to -30% for moderate risk
- Tighter stops = higher probability of stop-out from noise
- Wider stops = more capital at risk

**Take Profit Rules:**
- Options: +30% to +50% for moderate risk
- Don't be greedy - theta is always working against you
- Consider trailing stop after +20% gain

**Time-Based Exits:**
- Exit 7-10 days before expiration if not profitable
- Theta accelerates dramatically in final days
- Better to take small loss than watch decay to zero

### Risk Management for Small Accounts

**With $200-500:**
- Consider paper trading options until $1,000+
- If trading: max 1-2 contracts at a time
- Focus on high probability setups only
- One loss shouldn't devastate account

**With $500-2,000:**
- Max 2-3 positions simultaneously
- Risk 2-5% per trade
- Mix of 30-45 day options
- Track win rate and adjust

**With $2,000+:**
- Can properly diversify options trades
- Multiple timeframes and strategies
- Room for mistakes and learning
- Consider spreads to reduce capital requirements

---

## Quick Reference Card

### Greeks Cheat Sheet
| Greek | Meaning | Good For You | Bad For You |
|-------|---------|--------------|-------------|
| **Delta** | Direction | Stock moves your way | Stock moves against you |
| **Theta** | Time decay | Selling options | Buying options |
| **Vega** | Volatility | Rising volatility | Falling volatility |
| **Gamma** | Acceleration | Near-the-money | Far out-of-money |

### Pre-Trade Checklist
- [ ] Open Interest > 1,000
- [ ] Volume > 100 today
- [ ] Calculated total theta cost
- [ ] Position size < 5% of account
- [ ] Minimum 30 days to expiration
- [ ] Clear thesis for directional move
- [ ] Stop loss and take profit planned
- [ ] Break even is realistic

### Red Flags
- ⛔ Low open interest (<500)
- ⛔ Wide bid-ask spread (>5%)
- ⛔ Position size >10% of account
- ⛔ Less than 14 days to expiration
- ⛔ Theta > potential daily gains
- ⛔ No clear exit plan
- ⛔ IV extremely high (>50% for non-meme stocks)

---

## Additional Resources

**Options Calculators:**
- [OptionStrat](https://optionstrat.com/) - Visual options strategies
- [Options Profit Calculator](http://optionsprofitcalculator.com/) - P&L visualization
- [CBOE Options Calculator](https://www.cboe.com/tools/) - Official calculator

**Learning:**
- [Options Playbook](https://www.optionsplaybook.com/) - Strategy reference
- [TastyTrade](https://www.tastytrade.com/learn) - Options education
- [r/options Wiki](https://www.reddit.com/r/options/wiki/) - Community resources

**Market Data:**
- [Barchart Options](https://www.barchart.com/options/) - Free options screener
- [MarketChameleon](https://www.marketchameleon.com/) - Unusual options activity
- [CBOE](https://www.cboe.com/) - Volatility indices (VIX, etc.)
