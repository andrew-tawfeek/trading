# Experimental Research Ideas & Future Development

This document outlines potential research directions, experimental strategies, and advanced features for the Stock & Options Backtesting Framework.

## Table of Contents

- [Signal Generation Experiments](#signal-generation-experiments)
- [Risk Management Enhancements](#risk-management-enhancements)
- [Advanced Options Strategies](#advanced-options-strategies)
- [Machine Learning Integration](#machine-learning-integration)
- [Portfolio Optimization](#portfolio-optimization)
- [Alternative Technical Indicators](#alternative-technical-indicators)
- [Market Regime Detection](#market-regime-detection)
- [Live Trading Enhancements](#live-trading-enhancements)
- [Performance Improvements](#performance-improvements)
- [Data Quality & Validation](#data-quality--validation)

---

## Signal Generation Experiments

### 1. Multi-Timeframe Fourier Analysis

**Concept:** Combine Fourier signals from multiple timeframes for stronger confirmation.

**Implementation:**
```python
def multi_timeframe_signals(ticker, date_range):
    """
    Generate signals from multiple timeframes and combine them.
    """
    # Daily signals
    daily_signals = detect_fourier_signals(ticker, date_range, tick_size='1d')

    # Hourly signals (for confirmation)
    hourly_signals = detect_fourier_signals(ticker, date_range, tick_size='1h')

    # Weekly signals (for trend)
    weekly_signals = detect_fourier_signals(ticker, date_range, tick_size='1wk')

    # Combine: only trade when multiple timeframes align
    confirmed_signals = []
    for daily_sig in daily_signals:
        # Check if hourly and weekly agree
        if has_confirming_signal(daily_sig, hourly_signals, weekly_signals):
            confirmed_signals.append(daily_sig)

    return confirmed_signals
```

**Expected Benefits:**
- Reduced false signals
- Better trend alignment
- Higher win rate (possibly at cost of fewer trades)

**Parameters to Test:**
- Timeframe combinations (1d+1h, 1d+1wk, 1h+15min)
- Confirmation windows (how close signals need to be)
- Weight different timeframes differently

---

### 2. Adaptive Threshold Adjustment

**Concept:** Dynamically adjust overbought/oversold thresholds based on recent volatility.

**Implementation:**
```python
def adaptive_thresholds(analysis, base_threshold=5.0, lookback=20):
    """
    Adjust thresholds based on recent volatility in detrended Fourier.
    """
    recent_std = np.std(analysis.detrended_fourier[-lookback:])

    # Scale thresholds by recent volatility
    adaptive_overbought = base_threshold * (recent_std / 3.0)  # 3.0 is baseline
    adaptive_oversold = -base_threshold * (recent_std / 3.0)

    return adaptive_overbought, adaptive_oversold
```

**Research Questions:**
- Does adaptation improve performance in varying market conditions?
- What lookback period works best?
- Should we use exponential weighting for recent data?

---

### 3. Harmonic Ensemble Methods

**Concept:** Use multiple harmonic levels simultaneously and vote on signals.

**Implementation:**
```python
def ensemble_fourier_signals(prices, dates, harmonic_range=[5, 10, 15, 20]):
    """
    Generate signals from multiple harmonic levels and combine via voting.
    """
    all_signals = []

    for n_harm in harmonic_range:
        analysis = analyze_fourier(prices, dates, n_harmonics=n_harm)
        signals = detect_overbought_oversold(analysis, 5.0, -5.0)
        all_signals.append(signals)

    # Vote: only trade when majority of harmonics agree
    consensus_signals = majority_vote(all_signals)

    return consensus_signals
```

**Parameters to Explore:**
- Harmonic ranges (low: 1-5, medium: 10-15, high: 20-30)
- Voting thresholds (unanimous, majority, weighted)
- Harmonic-specific threshold adjustments

---

### 4. Fourier + RSI Hybrid

**Concept:** Combine Fourier signals with traditional RSI for confirmation.

**Implementation:**
```python
def fourier_rsi_hybrid(ticker, date_range):
    """
    Combine Fourier signals with RSI confirmation.
    """
    # Get Fourier signals
    fourier_signals = detect_fourier_signals(ticker, date_range)

    # Calculate RSI
    prices = get_stock_data(ticker, date_range)
    rsi = calculate_rsi(prices, period=14)

    # Filter: only take buy signals when RSI < 30, sell when RSI > 70
    filtered_signals = []
    for sig in fourier_signals:
        idx = sig.index
        if sig.signal_type == 'buy' and rsi[idx] < 30:
            filtered_signals.append(sig)
        elif sig.signal_type == 'sell' and rsi[idx] > 70:
            filtered_signals.append(sig)

    return filtered_signals
```

---

### 5. Volume-Weighted Signals

**Concept:** Give more weight to Fourier signals that occur on high volume.

**Implementation:**
```python
def volume_weighted_signals(analysis, volume_data, volume_threshold_percentile=75):
    """
    Filter signals by volume - only trade on high-volume signals.
    """
    volume_threshold = np.percentile(volume_data, volume_threshold_percentile)

    signals = detect_overbought_oversold(analysis, 5.0, -5.0)

    # Filter by volume
    high_volume_signals = [
        sig for sig in signals
        if volume_data[sig.index] >= volume_threshold
    ]

    return high_volume_signals
```

---

## Risk Management Enhancements

### 1. Kelly Criterion Position Sizing

**Concept:** Use Kelly Criterion to determine optimal position size based on win rate and average P&L.

**Implementation:**
```python
def kelly_position_size(win_rate, avg_win, avg_loss, capital, max_kelly=0.25):
    """
    Calculate optimal position size using Kelly Criterion.

    Args:
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade P&L %
        avg_loss: Average losing trade P&L % (positive number)
        capital: Available capital
        max_kelly: Maximum Kelly fraction to use (0.25 = quarter Kelly)
    """
    # Kelly formula: f = (bp - q) / b
    # where b = odds (avg_win/avg_loss), p = win_rate, q = 1 - p
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p

    kelly_fraction = (b * p - q) / b

    # Cap at max_kelly to reduce risk
    kelly_fraction = min(kelly_fraction, max_kelly)

    # Don't size if negative Kelly (negative expectancy)
    if kelly_fraction <= 0:
        return 0

    return capital * kelly_fraction
```

**Research:**
- Compare fixed size vs Kelly sizing
- Test fractional Kelly (1/4, 1/2, full)
- Dynamic Kelly based on recent performance

---

### 2. Maximum Drawdown Limits

**Concept:** Automatically reduce position sizes or pause trading after significant drawdowns.

**Implementation:**
```python
class DrawdownManager:
    def __init__(self, max_drawdown=0.20, recovery_threshold=0.10):
        self.max_drawdown = max_drawdown
        self.recovery_threshold = recovery_threshold
        self.peak_capital = 0
        self.in_drawdown = False

    def update(self, current_capital):
        """Update peak and check drawdown status."""
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital

        drawdown = (self.peak_capital - current_capital) / self.peak_capital

        if drawdown >= self.max_drawdown:
            self.in_drawdown = True
            return 'HALT_TRADING'  # Stop trading
        elif self.in_drawdown and drawdown <= self.recovery_threshold:
            self.in_drawdown = False
            return 'RESUME_TRADING'

        return 'CONTINUE'

    def get_position_multiplier(self, current_capital):
        """Reduce position size during drawdown."""
        drawdown = (self.peak_capital - current_capital) / self.peak_capital

        if drawdown < 0.10:
            return 1.0  # Full size
        elif drawdown < 0.15:
            return 0.5  # Half size
        else:
            return 0.25  # Quarter size
```

---

### 3. Correlation-Based Position Limits

**Concept:** Limit total exposure when holding multiple correlated positions.

**Implementation:**
```python
def check_correlation_limit(open_positions, new_ticker, max_correlated_exposure=0.30):
    """
    Check if adding a new position would exceed correlation limits.
    """
    # Get tickers of open positions
    open_tickers = [pos.ticker for pos in open_positions]

    if not open_tickers:
        return True  # No positions, OK to open

    # Calculate correlation matrix
    correlation_matrix = calculate_correlation(open_tickers + [new_ticker])

    # Check correlation of new ticker with existing positions
    avg_correlation = np.mean([
        correlation_matrix[new_ticker][ticker]
        for ticker in open_tickers
    ])

    # If highly correlated, limit total exposure
    if avg_correlation > 0.7:
        total_exposure = sum(pos.capital_at_risk for pos in open_positions)
        if total_exposure > max_correlated_exposure * total_capital:
            return False  # Too much correlated exposure

    return True
```

---

### 4. Time-Based Stop Loss Adjustment

**Concept:** Tighten stop-loss as option approaches expiration due to increasing theta decay.

**Implementation:**
```python
def dynamic_stoploss(entry_price, days_held, days_to_expiry, base_sl_pct=50):
    """
    Adjust stop-loss percentage based on time decay acceleration.
    """
    time_remaining_ratio = (days_to_expiry - days_held) / days_to_expiry

    if time_remaining_ratio > 0.66:
        # More than 2/3 time remaining: use base stop-loss
        return base_sl_pct
    elif time_remaining_ratio > 0.33:
        # 1/3 to 2/3 remaining: tighten to 40%
        return 40
    elif time_remaining_ratio > 0.15:
        # Less than 1/3 remaining: tighten to 30%
        return 30
    else:
        # Less than 15% time remaining: very tight 20%
        return 20
```

---

## Advanced Options Strategies

### 1. Vertical Spreads

**Concept:** Implement bull/bear spreads to reduce capital requirements and define risk.

**Implementation:**
```python
def backtest_vertical_spread(ticker, signals, spread_width=5):
    """
    Backtest bull call spreads (buy signal) or bear put spreads (sell signal).
    """
    for signal in signals:
        if signal.signal_type == 'buy':
            # Bull call spread: buy lower strike call, sell higher strike call
            long_strike = get_strike_near_price(signal.price, otm_pct=2)
            short_strike = long_strike + spread_width

            long_price = option_price_historical(ticker, exp_date, long_strike, 'call', signal.date)
            short_price = option_price_historical(ticker, exp_date, short_strike, 'call', signal.date)

            net_debit = long_price - short_price
            max_profit = spread_width - net_debit
            max_loss = net_debit

        # Similar for bear put spreads on sell signals
```

**Benefits:**
- Lower capital requirement
- Defined maximum risk
- Lower break-even point

**Research:**
- Optimal spread width
- Spread selection based on IV
- Dynamic spread adjustments

---

### 2. Iron Condors

**Concept:** Profit from low volatility / range-bound conditions.

**Implementation:**
```python
def detect_range_bound_conditions(analysis, threshold=3.0):
    """
    Identify when Fourier indicates range-bound market (low oscillations).
    """
    recent_std = np.std(analysis.detrended_fourier[-20:])

    if recent_std < threshold:
        return True  # Range-bound
    return False

def open_iron_condor(ticker, current_price, expiration, wing_width=5):
    """
    Open iron condor when range-bound detected.
    """
    # Sell OTM call and put
    short_call_strike = current_price * 1.05
    short_put_strike = current_price * 0.95

    # Buy further OTM for protection
    long_call_strike = short_call_strike + wing_width
    long_put_strike = short_put_strike - wing_width

    # Price all four legs...
```

---

### 3. Calendar Spreads

**Concept:** Take advantage of different theta decay rates.

**Implementation:**
```python
def calendar_spread_backtest(ticker, signal):
    """
    Buy long-dated option, sell short-dated option at same strike.
    """
    strike = get_strike_near_price(signal.price)

    # Buy 60 DTE
    long_exp = get_expiration_date(signal.date, days=60)
    long_price = option_price_historical(ticker, long_exp, strike, 'call', signal.date)

    # Sell 30 DTE
    short_exp = get_expiration_date(signal.date, days=30)
    short_price = option_price_historical(ticker, short_exp, strike, 'call', signal.date)

    net_debit = long_price - short_price

    # Profit from theta decay differential
```

---

### 4. Delta-Neutral Strategies

**Concept:** Hedge directional risk using underlying stock.

**Implementation:**
```python
def delta_neutral_position(option_position, current_stock_price):
    """
    Calculate shares of stock needed to hedge option delta.
    """
    option_delta = option_position.greeks['delta']
    contracts = option_position.contracts

    # Each contract = 100 shares, so total delta exposure:
    total_delta = option_delta * contracts * 100

    # Hedge: if long calls (positive delta), short stock
    shares_to_short = int(total_delta)

    return shares_to_short
```

---

## Machine Learning Integration

### 1. Supervised Learning for Signal Quality

**Concept:** Train ML model to predict which Fourier signals will be profitable.

**Implementation:**
```python
def extract_signal_features(signal, analysis, price_history):
    """
    Extract features for ML model.
    """
    features = {
        'fourier_value': signal.fourier_value,
        'detrended_value': signal.detrended_value,
        'recent_volatility': np.std(price_history[-20:]),
        'distance_from_peak': ...,  # How far from recent Fourier peak
        'volume_ratio': ...,  # Current volume vs average
        'rsi': ...,  # RSI at signal
        'macd': ...,  # MACD at signal
        'days_to_expiry': ...,
        'iv_rank': ...,  # IV percentile
    }
    return features

# Train model
from sklearn.ensemble import RandomForestClassifier

X_train = [extract_signal_features(sig) for sig in training_signals]
y_train = [1 if sig.profitable else 0 for sig in training_signals]

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on new signals
signal_quality_score = model.predict_proba(extract_features(new_signal))
```

---

### 2. Reinforcement Learning for Strategy Optimization

**Concept:** Use RL agent to learn optimal entry/exit timing.

**Skeleton:**
```python
import gym
from stable_baselines3 import PPO

class TradingEnv(gym.Env):
    """Custom trading environment."""

    def __init__(self, stock_data, fourier_analysis):
        self.data = stock_data
        self.fourier = fourier_analysis
        # Define action space: 0=hold, 1=buy, 2=sell
        # Define observation space: current price, Fourier values, Greeks, etc.

    def step(self, action):
        # Execute action, return reward
        pass

    def reset(self):
        # Reset to beginning of data
        pass

# Train RL agent
env = TradingEnv(stock_data, fourier_analysis)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

---

### 3. LSTM for Price Prediction

**Concept:** Use LSTM to predict next-day price, combine with Fourier signals.

**Idea:**
- Train LSTM on historical price sequences
- Use LSTM predictions to filter Fourier signals
- Only trade when LSTM and Fourier agree on direction

---

## Portfolio Optimization

### 1. Modern Portfolio Theory Integration

**Concept:** Optimize allocation across multiple tickers based on expected returns and covariance.

**Implementation:**
```python
from scipy.optimize import minimize

def optimize_portfolio(expected_returns, covariance_matrix, risk_aversion=1.0):
    """
    Find optimal portfolio weights using mean-variance optimization.
    """
    n_assets = len(expected_returns)

    def portfolio_variance(weights):
        return weights @ covariance_matrix @ weights

    def portfolio_return(weights):
        return weights @ expected_returns

    def objective(weights):
        # Maximize return - risk_aversion * variance
        return -(portfolio_return(weights) - risk_aversion * portfolio_variance(weights))

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
    ]
    bounds = [(0, 1) for _ in range(n_assets)]  # Long only

    result = minimize(objective, x0=np.ones(n_assets)/n_assets,
                     method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x  # Optimal weights
```

---

### 2. Pair Trading with Fourier

**Concept:** Find correlated stocks and trade their Fourier spread.

**Implementation:**
```python
def find_cointegrated_pairs(tickers, lookback_days=252):
    """
    Find pairs of stocks that are cointegrated.
    """
    from statsmodels.tsa.stattools import coint

    pairs = []
    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i+1:]:
            prices1 = get_historical_prices(ticker1, lookback_days)
            prices2 = get_historical_prices(ticker2, lookback_days)

            score, pvalue, _ = coint(prices1, prices2)

            if pvalue < 0.05:  # Cointegrated at 95% confidence
                pairs.append((ticker1, ticker2, pvalue))

    return pairs

def trade_pair_spread(ticker1, ticker2):
    """
    Trade the Fourier spread between cointegrated pairs.
    """
    # Analyze both stocks
    analysis1 = analyze_fourier(get_data(ticker1))
    analysis2 = analyze_fourier(get_data(ticker2))

    # Calculate spread
    spread = analysis1.detrended_fourier - analysis2.detrended_fourier

    # Trade when spread mean-reverts
    if spread[-1] > 2 * np.std(spread):
        # Spread too wide: short ticker1, long ticker2
        pass
    elif spread[-1] < -2 * np.std(spread):
        # Spread too narrow: long ticker1, short ticker2
        pass
```

---

### 3. Sector Rotation Strategy

**Concept:** Use Fourier to identify sector momentum and rotate capital.

**Implementation:**
```python
def sector_rotation_backtest(sector_etfs, rebalance_frequency='monthly'):
    """
    Rotate capital into strongest sector based on Fourier momentum.
    """
    sectors = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Finance': 'XLF',
        'Energy': 'XLE',
        # ... more sectors
    }

    for rebalance_date in get_rebalance_dates(start, end, rebalance_frequency):
        sector_scores = {}

        for sector, etf in sectors.items():
            analysis = analyze_fourier(get_data(etf, lookback=60))

            # Score based on Fourier trend slope
            trend_slope = analysis.trend_coeffs[0]
            sector_scores[sector] = trend_slope

        # Invest in top 3 sectors
        top_sectors = sorted(sector_scores, key=sector_scores.get, reverse=True)[:3]
        allocate_capital(top_sectors, equal_weight=True)
```

---

## Alternative Technical Indicators

### 1. Wavelet Transform Instead of Fourier

**Concept:** Use wavelets for better time-frequency localization.

**Rationale:** Wavelets can capture both frequency and time information, potentially better for non-stationary price series.

**Research:**
```python
import pywt

def wavelet_analysis(prices, wavelet='db4', level=4):
    """
    Decompose price series using wavelet transform.
    """
    coeffs = pywt.wavedec(prices, wavelet, level=level)

    # Reconstruct using only approximation coefficients (low frequency)
    reconstructed = pywt.waverec([coeffs[0]] + [None]*level, wavelet)

    return reconstructed[:len(prices)]
```

---

### 2. Hilbert Transform for Cycle Detection

**Concept:** Use Hilbert transform to detect market cycles.

**Implementation:**
```python
from scipy.signal import hilbert

def hilbert_cycle_detection(prices):
    """
    Detect dominant cycle using Hilbert transform.
    """
    analytic_signal = hilbert(prices)
    amplitude = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal))

    # Instantaneous frequency
    inst_freq = np.diff(phase) / (2 * np.pi)

    # Dominant cycle period
    dominant_period = 1 / np.mean(inst_freq)

    return dominant_period
```

---

### 3. Empirical Mode Decomposition (EMD)

**Concept:** Decompose price series into intrinsic mode functions.

**Benefits:**
- Adaptive, data-driven decomposition
- No need to choose harmonics
- Better for non-linear, non-stationary data

---

## Market Regime Detection

### 1. Hidden Markov Models for Market States

**Concept:** Use HMM to detect bull/bear/sideways market regimes.

**Implementation:**
```python
from hmmlearn.hmm import GaussianHMM

def detect_market_regime(returns, n_states=3):
    """
    Fit HMM to returns to detect market regimes.

    States might represent:
    0: Bull market (positive mean, low variance)
    1: Bear market (negative mean, low variance)
    2: High volatility / sideways (high variance)
    """
    model = GaussianHMM(n_components=n_states, covariance_type='full')
    model.fit(returns.reshape(-1, 1))

    # Predict current regime
    hidden_states = model.predict(returns.reshape(-1, 1))

    return hidden_states

def regime_adaptive_strategy(ticker, date_range):
    """
    Adjust strategy parameters based on detected regime.
    """
    returns = calculate_returns(ticker, date_range)
    regimes = detect_market_regime(returns)

    current_regime = regimes[-1]

    if current_regime == 0:  # Bull market
        # More aggressive: tighter thresholds, more trades
        return {'overbought': 4.0, 'oversold': -4.0, 'position_size': 1.5}
    elif current_regime == 1:  # Bear market
        # Defensive: wider thresholds, smaller positions
        return {'overbought': 7.0, 'oversold': -7.0, 'position_size': 0.5}
    else:  # High volatility
        # Cautious: fewer trades, tight stops
        return {'overbought': 8.0, 'oversold': -8.0, 'position_size': 0.75}
```

---

### 2. Volatility Regime Clustering

**Concept:** Cluster historical periods by volatility characteristics.

**Use:** Adapt stop-loss and position sizing based on volatility regime.

---

## Live Trading Enhancements

### 1. Order Execution Optimization

**Concepts:**
- TWAP (Time-Weighted Average Price) execution
- VWAP (Volume-Weighted Average Price) execution
- Smart order routing

---

### 2. Slippage Modeling

**Concept:** Account for realistic slippage in backtests.

**Implementation:**
```python
def apply_slippage(ideal_price, order_type, volume, avg_volume):
    """
    Model realistic slippage based on order size and liquidity.
    """
    volume_ratio = volume / avg_volume

    if order_type == 'market':
        # Market orders: more slippage
        base_slippage = 0.001  # 0.1%
    else:  # limit
        base_slippage = 0.0005  # 0.05%

    # Scale by volume ratio
    slippage_factor = base_slippage * (1 + volume_ratio)

    if order_type == 'buy':
        return ideal_price * (1 + slippage_factor)
    else:  # sell
        return ideal_price * (1 - slippage_factor)
```

---

### 3. Commission Modeling

**Concept:** Include realistic commissions in backtest P&L.

**Implementation:**
```python
def calculate_commission(price, contracts, commission_per_contract=0.65):
    """
    Calculate total commission for options trade.

    Typical retail commissions:
    - Robinhood: $0
    - TD Ameritrade: $0.65/contract
    - Interactive Brokers: $0.25-$0.65/contract
    - Traditional brokers: $1-$2/contract
    """
    return commission_per_contract * contracts
```

---

## Performance Improvements

### 1. Parallel Backtesting

**Concept:** Run multiple backtests in parallel for parameter optimization.

**Implementation:**
```python
from multiprocessing import Pool
from itertools import product

def run_single_backtest(params):
    """Run a single backtest with given parameters."""
    ticker, n_harm, sigma, ob, os = params
    results = run_fourier_backtest(
        ticker, '2025-01-01', '2025-12-20',
        n_harmonics=n_harm, smoothing_sigma=sigma,
        overbought_threshold=ob, oversold_threshold=os
    )
    return (params, results['backtest']['total_return'])

def parallel_optimization(ticker, param_grid):
    """
    Run backtests in parallel for all parameter combinations.
    """
    # Create parameter combinations
    combinations = list(product(
        [ticker],
        param_grid['n_harmonics'],
        param_grid['smoothing_sigma'],
        param_grid['overbought'],
        param_grid['oversold']
    ))

    # Run in parallel
    with Pool(processes=8) as pool:
        results = pool.map(run_single_backtest, combinations)

    # Find best
    best_params, best_return = max(results, key=lambda x: x[1])
    return best_params, best_return

# Usage
param_grid = {
    'n_harmonics': range(5, 25, 5),
    'smoothing_sigma': [0, 1, 2, 3],
    'overbought': range(4, 10),
    'oversold': range(-10, -4)
}

best_params, best_return = parallel_optimization('SPY', param_grid)
```

---

### 2. Incremental Data Updates

**Concept:** Only download new data since last cached date.

**Implementation:**
```python
def incremental_update(ticker, cache_dir='.yfinance_cache'):
    """
    Update cache with only new data since last update.
    """
    # Get last cached date
    last_cached_date = get_last_cached_date(ticker, cache_dir)

    if last_cached_date:
        # Download only new data
        new_data = yf.download(ticker, start=last_cached_date, end='today')

        # Merge with cached data
        cached_data = load_from_cache(ticker, cache_dir)
        merged = pd.concat([cached_data, new_data]).drop_duplicates()

        # Save back to cache
        save_to_cache(ticker, merged, cache_dir)
    else:
        # No cache, download all
        download_cached(ticker, '2000-01-01', 'today')
```

---

### 3. GPU Acceleration for Large-Scale Backtests

**Concept:** Use GPU for parallel Black-Scholes calculations.

**Libraries:** CuPy, Numba CUDA

---

## Data Quality & Validation

### 1. Outlier Detection

**Concept:** Detect and handle price anomalies.

**Implementation:**
```python
def detect_price_outliers(prices, method='iqr', threshold=3):
    """
    Detect outlier prices using IQR or z-score method.
    """
    if method == 'iqr':
        Q1 = np.percentile(prices, 25)
        Q3 = np.percentile(prices, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (prices < lower_bound) | (prices > upper_bound)

    elif method == 'zscore':
        z_scores = (prices - np.mean(prices)) / np.std(prices)
        outliers = np.abs(z_scores) > threshold

    return outliers

def clean_price_data(prices, outlier_method='median'):
    """
    Clean outliers from price data.
    """
    outliers = detect_price_outliers(prices)

    if outlier_method == 'median':
        # Replace outliers with median of neighbors
        prices_cleaned = prices.copy()
        for i in np.where(outliers)[0]:
            neighbors = prices[max(0, i-2):min(len(prices), i+3)]
            prices_cleaned[i] = np.median([p for j, p in enumerate(neighbors) if j != i])

    return prices_cleaned
```

---

### 2. Data Integrity Checks

**Concept:** Validate data before backtesting.

**Checks:**
```python
def validate_stock_data(data):
    """
    Validate stock data quality.
    """
    issues = []

    # Check for missing dates (gaps > 5 days)
    date_diffs = np.diff(data.index.values).astype('timedelta64[D]')
    if np.any(date_diffs > np.timedelta64(5, 'D')):
        issues.append('WARNING: Large gaps in data detected')

    # Check for zero/negative prices
    if np.any(data['Close'] <= 0):
        issues.append('ERROR: Zero or negative prices detected')

    # Check for suspiciously low volume
    if np.any(data['Volume'] == 0):
        issues.append('WARNING: Zero volume days detected')

    # Check for price discontinuities (splits not adjusted)
    price_changes = np.diff(data['Close']) / data['Close'][:-1]
    if np.any(np.abs(price_changes) > 0.50):  # 50% jump
        issues.append('WARNING: Large price jumps detected (possible split)')

    return issues
```

---

### 3. Data Source Diversification

**Concept:** Use multiple data sources and cross-validate.

**Idea:**
```python
def cross_validate_prices(ticker, date, sources=['yfinance', 'alphavantage', 'iex']):
    """
    Get price from multiple sources and check for discrepancies.
    """
    prices = {}
    for source in sources:
        prices[source] = get_price_from_source(ticker, date, source)

    # Check variance
    price_std = np.std(list(prices.values()))
    price_mean = np.mean(list(prices.values()))

    if price_std / price_mean > 0.01:  # More than 1% variance
        print(f"WARNING: Price discrepancy across sources for {ticker} on {date}")
        print(f"Prices: {prices}")

    return price_mean  # Use mean
```

---

## Implementation Priorities

### High Priority (Implement First)
1. **Adaptive Thresholds** - Easy to implement, likely high impact
2. **Kelly Position Sizing** - Critical risk management improvement
3. **Maximum Drawdown Limits** - Protect capital
4. **Parallel Backtesting** - Essential for parameter optimization
5. **Commission/Slippage Modeling** - More realistic backtests

### Medium Priority
1. **Multi-Timeframe Analysis** - More robust signals
2. **Vertical Spreads** - Reduce capital requirements
3. **Volume-Weighted Signals** - Filter quality
4. **HMM Market Regimes** - Adaptive strategies
5. **Data Validation** - Ensure quality

### Low Priority (Research Projects)
1. **Machine Learning Integration** - Complex, uncertain benefit
2. **Wavelet/EMD Alternatives** - Research phase
3. **GPU Acceleration** - Only needed for massive scale
4. **Pair Trading** - Different strategy class
5. **RL Optimization** - Cutting edge, experimental

---

## Contribution Guidelines

If you implement any of these ideas:

1. **Document thoroughly** - Add docstrings and comments
2. **Add unit tests** - Verify correctness
3. **Benchmark performance** - Compare to baseline
4. **Update README** - Add to examples
5. **Share results** - What worked? What didn't?

---

## Research Questions to Explore

1. **Fourier Harmonics**: Is there an optimal harmonic count for different stock volatility levels?
2. **Holding Periods**: What's the relationship between Fourier smoothing and optimal holding period?
3. **Sector Differences**: Do technology stocks need different parameters than utilities?
4. **Market Cap Effects**: Do small caps behave differently than large caps?
5. **Earnings Proximity**: Should we avoid trading near earnings dates?
6. **IV Rank**: Is there an optimal IV percentile for entering options trades?
7. **Theta Optimization**: What's the sweet spot for days to expiry?
8. **Signal Clustering**: Do multiple signals close together indicate stronger moves?

---

*This is a living document. Add your ideas, experiments, and results!*
