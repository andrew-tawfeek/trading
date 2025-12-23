"""
ReplayFourier_Model.py - Historical Replay of Live Options Trading Model

This module simulates the LiveFourier_Model.py over historical dates, allowing you to
backtest the exact same logic that runs in live trading but against historical data.

Usage:
    python ReplayFourier_Model.py

Or import and use programmatically:
    from ReplayFourier_Model import run_replay_model

    run_replay_model(
        ticker='AAPL',
        start_date='2024-01-01',
        end_date='2024-12-01',
        n_harmonics=18,
        smoothing_sigma=0,
        overbought_threshold=9,
        oversold_threshold=-8,
        tick_size='1d',
        update_interval=60,
        initial_capital=10000
    )
"""

import yfinance as yf
from datetime import datetime, timedelta, date
import time
from typing import Optional, Dict, List
from dataclasses import dataclass
import json
import os
import sys
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import holidays

# NYSE holidays calendar
US_HOLIDAYS = holidays.US()

# Import from fourier
from fourier import (
    FourierAnalysis,
    SignalPoint,
    get_option_strike_price,
    get_option_expiration,
    analyze_fourier,
    detect_overbought_oversold
)

# Import from functions
from functions import option_price_historical, greeks_historical


@dataclass
class SimulatedPosition:
    """Represents a simulated option position"""
    entry_date: datetime
    option_type: str  # 'call' or 'put'
    strike_price: float
    expiration_date: str
    entry_price: float
    contracts: int
    cost: float
    strike_date_obj: datetime

    # Greeks at entry
    greeks_at_entry: Optional[Dict]

    # Exit tracking
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_percent: Optional[float] = None
    pnl_dollar: Optional[float] = None


class ReplayOptionsModel:
    """
    Historical replay of the live options trading model using Fourier signals.
    """

    @staticmethod
    def retry_api_call(func, *args, max_retries=3, **kwargs):
        """
        Retry wrapper for external API calls with exponential backoff.

        Parameters
        ----------
        func : callable
            Function to retry
        max_retries : int
            Maximum number of retry attempts
        *args, **kwargs
            Arguments to pass to the function

        Returns
        -------
        Result of the function call, or None if all retries fail
        """
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, raise the exception
                    raise
                # Wait with exponential backoff
                time.sleep(0.5 * (attempt + 1))
        return None

    def __init__(self,
                 ticker: str,
                 start_date: str,
                 end_date: str,
                 n_harmonics: int = 18,
                 smoothing_sigma: float = 0.0,
                 overbought_threshold: float = 9.0,
                 oversold_threshold: float = -8.0,
                 tick_size: str = '1d',
                 lookback_period: str = '5d',
                 update_interval: int = 60,
                 contracts_per_trade: int = 1,
                 max_positions: int = 2,
                 stoploss_percent: float = 50.0,
                 takeprofit_percent: float = 50.0,
                 days_to_expiry: int = 30,
                 otm_percent: float = 2.0,
                 initial_capital: float = 10000.0):
        """
        Initialize the replay options trading model.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date for replay in 'YYYY-MM-DD' format
        end_date : str
            End date for replay in 'YYYY-MM-DD' format
        n_harmonics : int
            Number of Fourier harmonics
        smoothing_sigma : float
            Smoothing parameter
        overbought_threshold : float
            Sell signal threshold
        oversold_threshold : float
            Buy signal threshold
        tick_size : str
            Data interval
        lookback_period : str
            Historical data period for Fourier analysis
        update_interval : int
            Seconds between updates (simulated)
        contracts_per_trade : int
            Number of contracts per trade
        max_positions : int
            Maximum open positions
        stoploss_percent : float
            Stop-loss percentage
        takeprofit_percent : float
            Take-profit percentage
        days_to_expiry : int
            Days to option expiration
        otm_percent : float
            Out-of-the-money percentage
        initial_capital : float
            Starting capital
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.n_harmonics = n_harmonics
        self.smoothing_sigma = smoothing_sigma
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.tick_size = tick_size
        self.lookback_period = lookback_period
        self.update_interval = update_interval

        # Trading parameters
        self.contracts_per_trade = contracts_per_trade
        self.max_positions = max_positions
        self.stoploss_percent = stoploss_percent
        self.takeprofit_percent = takeprofit_percent
        self.days_to_expiry = days_to_expiry
        self.otm_percent = otm_percent
        self.initial_capital = initial_capital

        # Account tracking
        self.capital = initial_capital
        self.open_positions: List[SimulatedPosition] = []
        self.closed_positions: List[SimulatedPosition] = []

        # Last signal tracking - use date instead of index for reliability
        self.last_signal_date = None

        # Cooldown tracking to prevent overtrading
        self.last_trade_date = None
        self.min_days_between_trades = 3  # Minimum days between opening new positions

        print(f"\n{'='*80}")
        print(f"REPLAY OPTIONS TRADING MODEL: {ticker}")
        print(f"{'='*80}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Contracts per Trade: {contracts_per_trade}")
        print(f"Max Positions: {max_positions}")
        print(f"Stop-Loss: {stoploss_percent}% | Take-Profit: {takeprofit_percent}%")
        print(f"Days to Expiry: {days_to_expiry} | OTM: {otm_percent}%")
        print(f"{'='*80}\n")

    def is_market_open(self, current_date: datetime) -> bool:
        """Check if market is open on given date (9:30 AM - 4:00 PM ET, Mon-Fri, excluding holidays)"""
        # Get time in Eastern Time
        et_tz = ZoneInfo('America/New_York')

        # Convert to ET if not already
        if current_date.tzinfo is None:
            current_et = current_date.replace(tzinfo=et_tz)
        else:
            current_et = current_date.astimezone(et_tz)

        # Check if weekend
        if current_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check if holiday
        if current_et.date() in US_HOLIDAYS:
            return False

        # Skip intraday hour checks for daily data (timestamps are at midnight)
        if self.tick_size == '1d':
            return True

        # Check market hours for intraday data
        market_open = current_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_et.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= current_et <= market_close

    def fetch_data_for_date(self, current_date: datetime) -> pd.DataFrame:
        """Fetch historical data up to current_date for Fourier analysis"""
        # Calculate lookback start date - more robust parsing
        if 'mo' in self.lookback_period:
            lookback_days = int(self.lookback_period.replace('mo', '')) * 30
        elif 'd' in self.lookback_period:
            lookback_days = int(self.lookback_period.replace('d', ''))
        else:
            # Default fallback for unknown formats
            lookback_days = 5

        start = (current_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end = current_date.strftime('%Y-%m-%d')

        try:
            df = yf.download(
                tickers=self.ticker,
                start=start,
                end=end,
                interval=self.tick_size,
                progress=False
            )

            if df.empty:
                return None

            return df
        except Exception as e:
            print(f"Error fetching data for {current_date}: {e}")
            return None

    def extract_prices(self, df: pd.DataFrame) -> tuple:
        """Extract OHLC prices from DataFrame"""
        try:
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                close_prices = df['Close'][self.ticker].values
                open_prices = df['Open'][self.ticker].values
                high_prices = df['High'][self.ticker].values
                low_prices = df['Low'][self.ticker].values
            else:
                close_prices = df['Close'].values
                open_prices = df['Open'].values
                high_prices = df['High'].values
                low_prices = df['Low'].values

            return close_prices, open_prices, high_prices, low_prices, df.index
        except Exception as e:
            print(f"Error extracting prices: {e}")
            return None

    def check_for_new_signal(self, analysis: FourierAnalysis, dates) -> Optional[SignalPoint]:
        """
        Check if current conditions trigger a new buy or sell signal.
        Returns the most recent signal if it's new (not previously seen).
        """
        signals = detect_overbought_oversold(
            analysis,
            self.overbought_threshold,
            self.oversold_threshold
        )

        if signals:
            # Get the most recent signal
            latest_signal = signals[-1]

            # Get signal date from the dates array
            try:
                signal_date = dates[latest_signal.index]
                if hasattr(signal_date, 'to_pydatetime'):
                    signal_date = signal_date.to_pydatetime()
            except (IndexError, AttributeError):
                # Fallback to index-based tracking if date extraction fails
                signal_date = None

            # Only return signal if it's newer than what we've seen
            if signal_date is not None:
                if self.last_signal_date is None or signal_date > self.last_signal_date:
                    self.last_signal_date = signal_date
                    return latest_signal
            else:
                # Fallback to old behavior if we can't get the date
                if self.last_signal_date is None:
                    return latest_signal

        return None

    def get_trend_direction(self, prices: np.ndarray, lookback: int = 20) -> str:
        """
        Determine overall trend to filter signals.

        Parameters
        ----------
        prices : np.ndarray
            Price array
        lookback : int
            Number of periods to look back for trend (default: 20)

        Returns
        -------
        str
            'bullish', 'bearish', or 'neutral'
        """
        if len(prices) < lookback:
            return 'neutral'

        recent = prices[-lookback:]
        slope = (recent[-1] - recent[0]) / lookback

        # Adjust threshold based on typical price magnitude
        # Use 0.5% of average price as threshold
        avg_price = np.mean(recent)
        threshold = avg_price * 0.005

        if slope > threshold:
            return 'bullish'
        elif slope < -threshold:
            return 'bearish'
        return 'neutral'

    def calculate_position_size(self, option_price: float, max_risk_percent: float = 10.0) -> int:
        """
        Calculate number of contracts based on available capital and risk limits.

        Parameters
        ----------
        option_price : float
            Price per share of the option
        max_risk_percent : float
            Maximum percentage of capital to risk on a single trade (default: 10%)

        Returns
        -------
        int
            Number of contracts to trade (0 if cannot afford any within risk limits)
        """
        # Absolute maximum contracts to prevent unrealistic positions
        MAX_CONTRACTS = 100
        
        max_cost = self.capital * (max_risk_percent / 100)
        cost_per_contract = option_price * 100

        if cost_per_contract > max_cost:
            return 0  # Can't afford even 1 contract within risk limits

        calculated_contracts = int(max_cost // cost_per_contract)
        
        # Apply maximum limit
        contracts = min(calculated_contracts, MAX_CONTRACTS)
        
        return max(1, contracts)

    def handle_signal(self, signal: SignalPoint, current_date: datetime, close_prices: np.ndarray):
        """
        Handle a trading signal by opening a new position.

        Parameters
        ----------
        signal : SignalPoint
            The trading signal
        current_date : datetime
            Current date
        close_prices : np.ndarray
            Historical close prices for trend analysis
        """
        # Check if we can open new positions
        if len(self.open_positions) >= self.max_positions:
            print(f"[{current_date.strftime('%Y-%m-%d %H:%M:%S')}] Max positions reached ({self.max_positions}) - cannot open new position")
            return

        # Check cooldown period
        if self.last_trade_date:
            days_since_last = (current_date.date() - self.last_trade_date.date()).days
            if days_since_last < self.min_days_between_trades:
                print(f"[{current_date.strftime('%Y-%m-%d %H:%M:%S')}] Cooldown active - "
                      f"{days_since_last} days since last trade (min: {self.min_days_between_trades})")
                return

        # Determine option type based on signal
        if signal.signal_type == 'buy':
            option_type = 'call'
        elif signal.signal_type == 'sell':
            option_type = 'put'
        else:
            return

        # Filter signals based on trend (avoid trading against strong trends)
        trend = self.get_trend_direction(close_prices)
        if signal.signal_type == 'sell' and trend == 'bullish':
            print(f"[{current_date.strftime('%Y-%m-%d %H:%M:%S')}] Skipping PUT signal - bullish trend detected")
            return
        elif signal.signal_type == 'buy' and trend == 'bearish':
            print(f"[{current_date.strftime('%Y-%m-%d %H:%M:%S')}] Skipping CALL signal - bearish trend detected")
            return

        current_price = signal.price
        strike_price = get_option_strike_price(current_price, option_type, self.otm_percent)
        expiration_date = get_option_expiration(current_date, self.days_to_expiry, self.ticker)

        # Validate expiration date
        if expiration_date is None:
            print(f"  Warning: Could not determine expiration date")
            return

        try:
            # Get historical option price with retry logic
            entry_price = self.retry_api_call(
                option_price_historical,
                self.ticker, expiration_date, strike_price,
                option_type, current_date.strftime('%Y-%m-%d')
            )

            # Validate price with minimum threshold
            MIN_OPTION_PRICE = 0.01  # $0.01 minimum per share
            if entry_price is None or entry_price < MIN_OPTION_PRICE:
                print(f"  Warning: Invalid option price (${entry_price if entry_price else 0:.4f}) - must be >= ${MIN_OPTION_PRICE}")
                return

            # Calculate position size (use dynamic sizing instead of fixed contracts_per_trade)
            # Max 10% of capital per trade
            contracts_to_trade = self.calculate_position_size(entry_price, max_risk_percent=10.0)

            if contracts_to_trade == 0:
                print(f"[{current_date.strftime('%Y-%m-%d %H:%M:%S')}] INSUFFICIENT CAPITAL")
                print(f"  Option price: ${entry_price:.2f}/share (${entry_price * 100:.2f}/contract)")
                print(f"  Available capital: ${self.capital:.2f}")
                print(f"  Max risk per trade (10%): ${self.capital * 0.10:.2f}")
                return

            # Calculate cost
            cost = entry_price * 100 * contracts_to_trade

            # Sanity check: prevent unrealistic positions
            max_reasonable_cost = self.initial_capital * 10  # Never risk more than 10x initial capital
            if cost > max_reasonable_cost:
                print(f"[{current_date.strftime('%Y-%m-%d %H:%M:%S')}] UNREALISTIC POSITION BLOCKED")
                print(f"  Calculated cost: ${cost:,.2f}")
                print(f"  Entry price: ${entry_price:.4f}/share")
                print(f"  Contracts: {contracts_to_trade:,}")
                print(f"  Max reasonable cost: ${max_reasonable_cost:,.2f}")
                return

            # Double-check we have enough capital
            if self.capital < cost:
                print(f"[{current_date.strftime('%Y-%m-%d %H:%M:%S')}] INSUFFICIENT CAPITAL")
                print(f"  Need: ${cost:.2f} | Available: ${self.capital:.2f}")
                return

            # Get Greeks at entry with retry logic
            try:
                greeks_data = self.retry_api_call(
                    greeks_historical,
                    self.ticker, expiration_date, strike_price,
                    option_type, current_date.strftime('%Y-%m-%d')
                )
                # Validate Greeks result
                if isinstance(greeks_data, str):  # Error message
                    print(f"  Warning: Greeks calculation returned: {greeks_data}")
                    greeks_data = None
            except Exception as e:
                print(f"  Warning: Could not calculate Greeks: {e}")
                greeks_data = None

            # Create position
            position = SimulatedPosition(
                entry_date=current_date,
                option_type=option_type,
                strike_price=strike_price,
                expiration_date=expiration_date,
                entry_price=entry_price,
                contracts=contracts_to_trade,
                cost=cost,
                strike_date_obj=datetime.strptime(expiration_date, '%Y-%m-%d'),
                greeks_at_entry=greeks_data
            )

            # Deduct from capital and add to positions
            self.capital -= cost
            self.open_positions.append(position)

            # Update last trade date for cooldown tracking
            self.last_trade_date = current_date

            # Print trade details
            print(f"\n{'='*80}")
            print(f"POSITION OPENED: {option_type.upper()}")
            print(f"{'='*80}")
            print(f"Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Stock Price: ${current_price:.2f}")
            print(f"Strike: ${strike_price:.2f}")
            print(f"Expiration: {expiration_date}")
            print(f"Entry Price: ${entry_price:.2f}")
            print(f"Contracts: {contracts_to_trade}")
            print(f"Cost: ${cost:.2f}")
            print(f"Capital Used: {(cost / self.initial_capital) * 100:.1f}% of initial")
            print(f"Remaining Capital: ${self.capital:,.2f}")

            if greeks_data and isinstance(greeks_data, dict):
                print(f"\nGreeks at Entry:")
                print(f"  Delta: {greeks_data['delta']:>8.4f}")
                print(f"  Gamma: {greeks_data['gamma']:>8.4f}")
                print(f"  Vega:  {greeks_data['vega']:>8.4f}")
                print(f"  Theta: {greeks_data['theta']:>8.4f}")
                print(f"  Rho:   {greeks_data['rho']:>8.4f}")

            print(f"{'='*80}\n")

        except Exception as e:
            print(f"[{current_date.strftime('%Y-%m-%d %H:%M:%S')}] ERROR opening position: {e}")

    def check_positions(self, current_date: datetime):
        """Check all open positions for stop-loss, take-profit, or expiration"""
        positions_to_close = []

        for position in self.open_positions:
            try:
                # Get current option price with retry logic
                current_price = self.retry_api_call(
                    option_price_historical,
                    self.ticker, position.expiration_date,
                    position.strike_price, position.option_type,
                    current_date.strftime('%Y-%m-%d')
                )

                # Validate price with minimum threshold
                MIN_OPTION_PRICE = 0.01
                if current_price is None or current_price < MIN_OPTION_PRICE:
                    print(f"  Warning: Invalid price (${current_price if current_price else 0:.4f}) for {position.option_type} ${position.strike_price}")
                    continue
                
                # Additional check: if entry price was invalid, close the position at minimal loss
                if position.entry_price < MIN_OPTION_PRICE:
                    print(f"  Warning: Position has invalid entry price ${position.entry_price:.4f}, force closing")
                    # Force close at minimal loss to clean up bad position
                    revenue = MIN_OPTION_PRICE * 100 * position.contracts
                    self.capital += revenue
                    position.exit_date = current_date
                    position.exit_price = MIN_OPTION_PRICE
                    position.exit_reason = 'invalid_entry'
                    position.pnl_percent = -99.0
                    position.pnl_dollar = revenue - position.cost
                    positions_to_close.append(position)
                    continue

                # Calculate P&L
                pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100

                # Check stop-loss
                if current_price <= position.entry_price * (1 - self.stoploss_percent / 100):
                    revenue = current_price * 100 * position.contracts
                    self.capital += revenue

                    position.exit_date = current_date
                    position.exit_price = current_price
                    position.exit_reason = 'stoploss'
                    position.pnl_percent = pnl_percent
                    position.pnl_dollar = (current_price - position.entry_price) * 100 * position.contracts

                    positions_to_close.append(position)

                    print(f"\n{'='*80}")
                    print(f"STOP-LOSS TRIGGERED: {position.option_type.upper()}")
                    print(f"{'='*80}")
                    print(f"Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Strike: ${position.strike_price:.2f} exp {position.expiration_date}")
                    print(f"Entry: ${position.entry_price:.2f} → Exit: ${current_price:.2f}")
                    print(f"P&L: {pnl_percent:.1f}% (${position.pnl_dollar:.2f})")
                    print(f"Revenue: ${revenue:.2f}")
                    print(f"Capital: ${self.capital:,.2f}")
                    print(f"{'='*80}\n")

                # Check take-profit
                elif current_price >= position.entry_price * (1 + self.takeprofit_percent / 100):
                    revenue = current_price * 100 * position.contracts
                    self.capital += revenue

                    position.exit_date = current_date
                    position.exit_price = current_price
                    position.exit_reason = 'takeprofit'
                    position.pnl_percent = pnl_percent
                    position.pnl_dollar = (current_price - position.entry_price) * 100 * position.contracts

                    positions_to_close.append(position)

                    print(f"\n{'='*80}")
                    print(f"TAKE-PROFIT TRIGGERED: {position.option_type.upper()}")
                    print(f"{'='*80}")
                    print(f"Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Strike: ${position.strike_price:.2f} exp {position.expiration_date}")
                    print(f"Entry: ${position.entry_price:.2f} → Exit: ${current_price:.2f}")
                    print(f"P&L: {pnl_percent:.1f}% (${position.pnl_dollar:.2f})")
                    print(f"Revenue: ${revenue:.2f}")
                    print(f"Capital: ${self.capital:,.2f}")
                    print(f"{'='*80}\n")

                # Check expiration (close 2 days before to avoid assignment risk)
                # Ensure timezone-naive comparison using date objects
                else:
                    days_to_exp = (position.strike_date_obj.date() - current_date.date()).days
                    if days_to_exp <= 2:
                        revenue = current_price * 100 * position.contracts
                        self.capital += revenue

                        position.exit_date = current_date
                        position.exit_price = current_price
                        position.exit_reason = 'expiration'
                        position.pnl_percent = pnl_percent
                        position.pnl_dollar = (current_price - position.entry_price) * 100 * position.contracts

                        positions_to_close.append(position)

                        print(f"\n{'='*80}")
                        print(f"OPTION EXPIRED: {position.option_type.upper()}")
                        print(f"{'='*80}")
                        print(f"Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Strike: ${position.strike_price:.2f}")
                        print(f"Entry: ${position.entry_price:.2f} → Exit: ${current_price:.2f}")
                        print(f"P&L: {pnl_percent:.1f}% (${position.pnl_dollar:.2f})")
                        print(f"Revenue: ${revenue:.2f}")
                        print(f"Capital: ${self.capital:,.2f}")
                        print(f"{'='*80}\n")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  Warning: Could not check position {position.option_type} ${position.strike_price}: {e}")
                continue

        # Move closed positions
        for position in positions_to_close:
            self.open_positions.remove(position)
            self.closed_positions.append(position)

    def run(self):
        """
        Run the historical replay of the live options trading model.
        """
        print(f"Starting replay from {self.start_date} to {self.end_date}")
        print(f"Press Ctrl+C to stop\n")

        # Calculate lookback for fetching extended data
        if 'mo' in self.lookback_period:
            lookback_days = int(self.lookback_period.replace('mo', '')) * 30
        elif 'd' in self.lookback_period:
            lookback_days = int(self.lookback_period.replace('d', ''))
        else:
            lookback_days = 5

        # Fetch all data once with extra lookback buffer
        extended_start = (datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        print(f"Fetching historical data from {extended_start} to {self.end_date}...")
        full_data = yf.download(
            tickers=self.ticker,
            start=extended_start,
            end=self.end_date,
            interval=self.tick_size,
            progress=False
        )

        if full_data is None or full_data.empty:
            print("No data available for specified date range")
            return

        # Get trading dates for the actual replay range
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')

        # Filter dates to the replay range
        trading_dates = []
        for d in full_data.index:
            dt = d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d
            if start_dt <= dt <= end_dt:
                trading_dates.append(d)

        total_dates = len(trading_dates)
        print(f"Data loaded. Processing {total_dates} dates...\n")

        try:
            for i, current_date in enumerate(trading_dates):
                # Convert to datetime if needed
                if hasattr(current_date, 'to_pydatetime'):
                    current_date = current_date.to_pydatetime()

                # Show progress periodically
                if i % 20 == 0:
                    progress_pct = (i / total_dates) * 100
                    print(f"Progress: {i}/{total_dates} ({progress_pct:.1f}%)")

                # Only process during market hours
                if not self.is_market_open(current_date):
                    continue

                # Use pre-fetched data up to current date (more efficient than re-fetching)
                df = full_data.loc[:current_date]
                if df.empty:
                    continue

                # Extract prices
                result = self.extract_prices(df)
                if result is None:
                    continue

                close_prices, _, _, _, dates = result

                # Perform Fourier analysis
                try:
                    analysis = analyze_fourier(
                        close_prices,
                        dates,
                        self.n_harmonics,
                        self.smoothing_sigma
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Error in Fourier analysis: {e}")
                    continue

                # Check for new signals
                new_signal = self.check_for_new_signal(analysis, dates)
                if new_signal:
                    self.handle_signal(new_signal, current_date, close_prices)

                # Check existing positions
                if self.open_positions:
                    self.check_positions(current_date)

                # Print status
                current_price = close_prices[-1]
                current_fourier = analysis.fourier_prices[-1]
                current_detrended = analysis.detrended_fourier[-1]

                print(f"[{current_date.strftime('%Y-%m-%d')}] {self.ticker}: "
                      f"Price=${current_price:.2f} | "
                      f"Fourier=${current_fourier:.2f} | "
                      f"Detrended=${current_detrended:.2f} | "
                      f"Capital=${self.capital:,.2f}")

        except KeyboardInterrupt:
            print("\n\nReplay stopped by user")
        except Exception as e:
            print(f"\nError during replay: {e}")

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print final trading summary with detailed performance metrics"""
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p.pnl_dollar and p.pnl_dollar > 0])
        losing_trades = len([p for p in self.closed_positions if p.pnl_dollar and p.pnl_dollar <= 0])

        # Value open positions at current market price (end date)
        portfolio_value = self.capital
        open_positions_value = 0

        print(f"\n{'='*80}")
        print(f"VALUING OPEN POSITIONS AT END DATE...")
        print(f"{'='*80}")

        for position in self.open_positions:
            try:
                # Get end date
                end_date = datetime.strptime(self.end_date, '%Y-%m-%d')

                # Get current market price for the option
                current_price = self.retry_api_call(
                    option_price_historical,
                    self.ticker, position.expiration_date,
                    position.strike_price, position.option_type,
                    end_date.strftime('%Y-%m-%d')
                )

                if current_price:
                    current_value = current_price * 100 * position.contracts
                    unrealized_pnl = current_value - position.cost
                    unrealized_pnl_pct = (unrealized_pnl / position.cost) * 100
                    portfolio_value += current_value
                    open_positions_value += current_value

                    print(f"  {position.option_type.upper()} ${position.strike_price} exp {position.expiration_date}: "
                          f"Cost=${position.cost:.2f} → Value=${current_value:.2f} "
                          f"(P&L: ${unrealized_pnl:+.2f} / {unrealized_pnl_pct:+.1f}%)")
                else:
                    # Fallback: use entry cost
                    portfolio_value += position.cost
                    open_positions_value += position.cost
                    print(f"  {position.option_type.upper()} ${position.strike_price}: Could not price, using cost ${position.cost:.2f}")
            except Exception as e:
                # Fallback: use entry cost
                portfolio_value += position.cost
                open_positions_value += position.cost
                print(f"  {position.option_type.upper()} ${position.strike_price}: Error pricing ({str(e)[:50]}), using cost ${position.cost:.2f}")

        total_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100

        # Calculate detailed metrics
        avg_win = 0
        avg_loss = 0
        if winning_trades > 0:
            avg_win = np.mean([p.pnl_dollar for p in self.closed_positions if p.pnl_dollar and p.pnl_dollar > 0])
        if losing_trades > 0:
            avg_loss = np.mean([p.pnl_dollar for p in self.closed_positions if p.pnl_dollar and p.pnl_dollar <= 0])

        # Calculate max drawdown
        equity_curve = [self.initial_capital]
        for p in self.closed_positions:
            equity_curve.append(equity_curve[-1] + (p.pnl_dollar if p.pnl_dollar else 0))

        peak = equity_curve[0]
        max_dd = 0
        for val in equity_curve:
            peak = max(peak, val)
            dd = (peak - val) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Count exit reasons
        exit_reasons = {}
        for p in self.closed_positions:
            reason = p.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print(f"\n{'='*80}")
        print(f"FINAL TRADING SUMMARY")
        print(f"{'='*80}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"\nCapital:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Cash: ${self.capital:,.2f}")
        print(f"  Open Positions Value: ${open_positions_value:,.2f}")
        print(f"  Final Portfolio Value: ${portfolio_value:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")

        print(f"\nTrade Statistics:")
        print(f"  Total Closed Trades: {total_trades}")
        print(f"  Winners: {winning_trades}")
        print(f"  Losers: {losing_trades}")
        if total_trades > 0:
            print(f"  Win Rate: {(winning_trades/total_trades)*100:.1f}%")

        print(f"\nPerformance Metrics:")
        if winning_trades > 0:
            print(f"  Average Win: ${avg_win:.2f}")
        if losing_trades > 0:
            print(f"  Average Loss: ${avg_loss:.2f}")
        if avg_loss != 0 and winning_trades > 0:
            print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x")
        print(f"  Max Drawdown: {max_dd:.1f}%")

        if exit_reasons:
            print(f"\nExit Reasons:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason.capitalize()}: {count}")

        print(f"\nOpen Positions: {len(self.open_positions)}")
        print(f"{'='*80}\n")


def run_replay_model(ticker: str,
                     start_date: str,
                     end_date: str,
                     n_harmonics: int = 18,
                     smoothing_sigma: float = 0.0,
                     overbought_threshold: float = 9.0,
                     oversold_threshold: float = -8.0,
                     tick_size: str = '1d',
                     lookback_period: str = '5d',
                     update_interval: int = 60,
                     contracts_per_trade: int = 1,
                     max_positions: int = 2,
                     stoploss_percent: float = 50.0,
                     takeprofit_percent: float = 50.0,
                     days_to_expiry: int = 30,
                     otm_percent: float = 2.0,
                     initial_capital: float = 10000.0):
    """
    Run the historical replay of the live options trading model with Fourier analysis.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    n_harmonics : int
        Number of Fourier harmonics (default: 18)
    smoothing_sigma : float
        Smoothing parameter (default: 0.0)
    overbought_threshold : float
        Sell signal threshold (default: 9.0)
    oversold_threshold : float
        Buy signal threshold (default: -8.0)
    tick_size : str
        Data interval (default: '1d')
    lookback_period : str
        Historical data period (default: '5d')
    update_interval : int
        Seconds between updates (simulated, default: 60)
    contracts_per_trade : int
        Number of contracts per trade (default: 1)
    max_positions : int
        Maximum open positions (default: 2)
    stoploss_percent : float
        Stop-loss percentage (default: 50%)
    takeprofit_percent : float
        Take-profit percentage (default: 50%)
    days_to_expiry : int
        Days to option expiration (default: 30)
    otm_percent : float
        Out-of-the-money percentage (default: 2%)
    initial_capital : float
        Starting capital (default: $10,000)

    Example:
    --------
    >>> run_replay_model('AAPL', '2024-01-01', '2024-12-01', n_harmonics=18, initial_capital=10000)
    """
    model = ReplayOptionsModel(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        n_harmonics=n_harmonics,
        smoothing_sigma=smoothing_sigma,
        overbought_threshold=overbought_threshold,
        oversold_threshold=oversold_threshold,
        tick_size=tick_size,
        lookback_period=lookback_period,
        update_interval=update_interval,
        contracts_per_trade=contracts_per_trade,
        max_positions=max_positions,
        stoploss_percent=stoploss_percent,
        takeprofit_percent=takeprofit_percent,
        days_to_expiry=days_to_expiry,
        otm_percent=otm_percent,
        initial_capital=initial_capital
    )

    model.run()


if __name__ == "__main__":
    import sys

    # Settings cache file location
    REPLAY_SETTINGS_CACHE_FILE = os.path.expanduser('~/.replay_fourier_model_settings.json')

    def load_replay_cached_settings():
        """Load previously saved replay model settings from cache file."""
        if os.path.exists(REPLAY_SETTINGS_CACHE_FILE):
            try:
                with open(REPLAY_SETTINGS_CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cached settings: {e}")
        return {}

    def save_replay_settings(settings: dict):
        """Save replay model settings to cache file for next run."""
        try:
            with open(REPLAY_SETTINGS_CACHE_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")

    # Load cached settings
    cached = load_replay_cached_settings()

    # Default values (will be overridden by cache if available)
    defaults = {
        'ticker': 'AAPL',
        'start_date': '2024-01-01',
        'end_date': '2024-12-01',
        'n_harmonics': 18,
        'smoothing_sigma': 0.0,
        'overbought_threshold': 9.0,
        'oversold_threshold': -8.0,
        'tick_size': '1d',
        'lookback_period': '5d',
        'update_interval': 60,
        'contracts_per_trade': 1,
        'max_positions': 2,
        'stoploss_percent': 50.0,
        'takeprofit_percent': 50.0,
        'days_to_expiry': 30,
        'otm_percent': 2.0,
        'initial_capital': 10000.0
    }

    # Merge cached settings with defaults
    defaults.update(cached)

    # Get ticker from command line or prompt user
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = input(f"Enter a stock ticker symbol (default: {defaults['ticker']}): ").strip().upper()
        if not ticker:
            ticker = defaults['ticker']

    # Prompt for parameters with cached defaults
    print("\nReplay Options Trading Model Configuration")
    print("=" * 60)
    if cached:
        print("(Previous settings loaded as defaults. Press Enter to use them)")
        print("=" * 60)

    try:
        # Date Range
        print("\n--- Date Range ---")

        start = input(f"Start date YYYY-MM-DD (default: {defaults['start_date']}): ").strip()
        start_date = start if start else defaults['start_date']

        end = input(f"End date YYYY-MM-DD (default: {defaults['end_date']}): ").strip()
        end_date = end if end else defaults['end_date']

        # Fourier Analysis Parameters
        print("\n--- Fourier Analysis Parameters ---")

        n_harm = input(f"Harmonics (default: {defaults['n_harmonics']}): ").strip()
        n_harmonics = int(n_harm) if n_harm else defaults['n_harmonics']

        smooth = input(f"Smoothing sigma (default: {defaults['smoothing_sigma']}): ").strip()
        smoothing_sigma = float(smooth) if smooth else defaults['smoothing_sigma']

        overbought = input(f"Overbought threshold (default: {defaults['overbought_threshold']}): ").strip()
        overbought_threshold = float(overbought) if overbought else defaults['overbought_threshold']

        oversold = input(f"Oversold threshold (default: {defaults['oversold_threshold']}): ").strip()
        oversold_threshold = float(oversold) if oversold else defaults['oversold_threshold']

        tick = input(f"Tick size - 1m/5m/15m/30m/1h/1d (default: {defaults['tick_size']}): ").strip()
        tick_size = tick if tick else defaults['tick_size']

        lookback = input(f"Lookback period - 1d/5d/1mo (default: {defaults['lookback_period']}): ").strip()
        lookback_period = lookback if lookback else defaults['lookback_period']

        interval = input(f"Update interval in seconds (default: {defaults['update_interval']}): ").strip()
        update_interval = int(interval) if interval else defaults['update_interval']

        # Trading Parameters
        print("\n--- Options Trading Parameters ---")

        capital = input(f"Initial capital (default: {defaults['initial_capital']}): ").strip()
        initial_capital = float(capital) if capital else defaults['initial_capital']

        contracts = input(f"Contracts per trade (default: {defaults['contracts_per_trade']}): ").strip()
        contracts_per_trade = int(contracts) if contracts else defaults['contracts_per_trade']

        max_pos = input(f"Max open positions (default: {defaults['max_positions']}): ").strip()
        max_positions = int(max_pos) if max_pos else defaults['max_positions']

        stoploss = input(f"Stop-loss percent (default: {defaults['stoploss_percent']}): ").strip()
        stoploss_percent = float(stoploss) if stoploss else defaults['stoploss_percent']

        takeprofit = input(f"Take-profit percent (default: {defaults['takeprofit_percent']}): ").strip()
        takeprofit_percent = float(takeprofit) if takeprofit else defaults['takeprofit_percent']

        days_exp = input(f"Days to expiry (default: {defaults['days_to_expiry']}): ").strip()
        days_to_expiry = int(days_exp) if days_exp else defaults['days_to_expiry']

        otm = input(f"Out-of-the-money percent (default: {defaults['otm_percent']}): ").strip()
        otm_percent = float(otm) if otm else defaults['otm_percent']

    except ValueError:
        print("Invalid input, using defaults")
        ticker = defaults['ticker']
        start_date = defaults['start_date']
        end_date = defaults['end_date']
        n_harmonics = defaults['n_harmonics']
        smoothing_sigma = defaults['smoothing_sigma']
        overbought_threshold = defaults['overbought_threshold']
        oversold_threshold = defaults['oversold_threshold']
        tick_size = defaults['tick_size']
        lookback_period = defaults['lookback_period']
        update_interval = defaults['update_interval']
        contracts_per_trade = defaults['contracts_per_trade']
        max_positions = defaults['max_positions']
        stoploss_percent = defaults['stoploss_percent']
        takeprofit_percent = defaults['takeprofit_percent']
        days_to_expiry = defaults['days_to_expiry']
        otm_percent = defaults['otm_percent']
        initial_capital = defaults['initial_capital']

    # Save current settings for next run
    current_settings = {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'n_harmonics': n_harmonics,
        'smoothing_sigma': smoothing_sigma,
        'overbought_threshold': overbought_threshold,
        'oversold_threshold': oversold_threshold,
        'tick_size': tick_size,
        'lookback_period': lookback_period,
        'update_interval': update_interval,
        'contracts_per_trade': contracts_per_trade,
        'max_positions': max_positions,
        'stoploss_percent': stoploss_percent,
        'takeprofit_percent': takeprofit_percent,
        'days_to_expiry': days_to_expiry,
        'otm_percent': otm_percent,
        'initial_capital': initial_capital
    }
    save_replay_settings(current_settings)

    # Display configuration summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Fourier: {n_harmonics} harmonics, sigma={smoothing_sigma}")
    print(f"Thresholds: Overbought={overbought_threshold}, Oversold={oversold_threshold}")
    print(f"Trading: {contracts_per_trade} contracts/trade, max {max_positions} positions")
    print(f"Risk: {stoploss_percent}% stop-loss, {takeprofit_percent}% take-profit")
    print(f"Options: {days_to_expiry}d expiry, {otm_percent}% OTM")
    print(f"Updates: Every {update_interval}s, {tick_size} tick size")
    print("=" * 60 + "\n")

    # Confirm before starting
    confirm = input("Start the replay with these settings? (y/n, default: y): ").strip().lower()
    if confirm and confirm != 'y':
        print("Replay cancelled.")
        sys.exit(0)

    # Run the replay model with configured parameters
    run_replay_model(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        n_harmonics=n_harmonics,
        smoothing_sigma=smoothing_sigma,
        overbought_threshold=overbought_threshold,
        oversold_threshold=oversold_threshold,
        tick_size=tick_size,
        lookback_period=lookback_period,
        update_interval=update_interval,
        contracts_per_trade=contracts_per_trade,
        max_positions=max_positions,
        stoploss_percent=stoploss_percent,
        takeprofit_percent=takeprofit_percent,
        days_to_expiry=days_to_expiry,
        otm_percent=otm_percent,
        initial_capital=initial_capital
    )
