"""
LiveFourier_Model.py - Live Options Trading Model using Fourier Analysis

This module combines real-time Fourier analysis with simulated options trading.
It monitors stock prices, detects buy/sell signals, and simulates option purchases
with real-time P&L tracking.

Usage:
    python LiveFourier_Model.py

Or import and use programmatically:
    from LiveFourier_Model import run_model

    run_model(
        ticker='AAPL',
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
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List
from dataclasses import dataclass
import json
import os
import sys
from zoneinfo import ZoneInfo

# Import from LiveFourier
from LiveFourier import LiveFourierMonitor, live_fourier_monitor

# Import from fourier
from fourier import (
    FourierAnalysis,
    SignalPoint,
    get_option_strike_price,
    get_option_expiration
)

# Import from functions
from functions import option_price, greeks


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
    greeks_at_entry: Dict

    # Exit tracking
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_percent: Optional[float] = None
    pnl_dollar: Optional[float] = None


class LiveOptionsModel:
    """
    Live options trading model that uses Fourier signals to simulate option trades.
    """

    def __init__(self,
                 ticker: str,
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
        Initialize the live options trading model.

        Parameters match live_fourier_monitor plus additional trading parameters.
        """
        self.ticker = ticker
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

        # Last capital update time
        self.last_capital_update = datetime.now()

        print(f"\n{'='*80}")
        print(f"LIVE OPTIONS TRADING MODEL: {ticker}")
        print(f"{'='*80}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Contracts per Trade: {contracts_per_trade}")
        print(f"Max Positions: {max_positions}")
        print(f"Stop-Loss: {stoploss_percent}% | Take-Profit: {takeprofit_percent}%")
        print(f"Days to Expiry: {days_to_expiry} | OTM: {otm_percent}%")
        print(f"{'='*80}\n")

    def is_market_open(self) -> bool:
        """Check if market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)"""
        # Get current time in Eastern Time
        et_tz = ZoneInfo('America/New_York')
        now_et = datetime.now(et_tz)

        # Check if weekend
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check market hours (simplified - doesn't account for holidays)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now_et <= market_close

    def handle_signal(self, signal_type: str, signal: SignalPoint, analysis: FourierAnalysis):
        """
        Callback function to handle trading signals from LiveFourierMonitor.

        This function is called when a buy/sell signal is detected.
        """
        current_time = datetime.now()

        # Only trade during market hours
        if not self.is_market_open():
            print(f"\n[{current_time.strftime('%H:%M:%S')}] Signal received outside market hours - skipping")
            return

        # Check if we can open new positions
        if len(self.open_positions) >= self.max_positions:
            print(f"\n[{current_time.strftime('%H:%M:%S')}] Max positions reached ({self.max_positions}) - cannot open new position")
            return

        # Determine option type based on signal
        if signal_type == 'buy':
            option_type = 'call'
        elif signal_type == 'sell':
            option_type = 'put'
        else:
            return

        current_price = signal.price
        strike_price = get_option_strike_price(current_price, option_type, self.otm_percent)
        expiration_date = get_option_expiration(current_time, self.days_to_expiry)

        try:
            # Get option price
            option_data = option_price(self.ticker, expiration_date, strike_price, option_type)
            entry_price = option_data['lastPrice']

            # Calculate cost
            cost = entry_price * 100 * self.contracts_per_trade

            # Check if we have enough capital
            if self.capital < cost:
                print(f"\n[{current_time.strftime('%H:%M:%S')}] INSUFFICIENT CAPITAL")
                print(f"  Need: ${cost:.2f} | Available: ${self.capital:.2f}")
                return

            # Get Greeks at entry
            try:
                greeks_data = greeks(self.ticker, expiration_date, strike_price, option_type,
                                    status=False, silent=True)
            except Exception as e:
                print(f"  Warning: Could not calculate Greeks: {e}")
                greeks_data = None

            # Create position
            position = SimulatedPosition(
                entry_date=current_time,
                option_type=option_type,
                strike_price=strike_price,
                expiration_date=expiration_date,
                entry_price=entry_price,
                contracts=self.contracts_per_trade,
                cost=cost,
                strike_date_obj=datetime.strptime(expiration_date, '%Y-%m-%d'),
                greeks_at_entry=greeks_data
            )

            # Deduct from capital and add to positions
            self.capital -= cost
            self.open_positions.append(position)

            # Print trade details
            print(f"\n{'='*80}")
            print(f"POSITION OPENED: {option_type.upper()}")
            print(f"{'='*80}")
            print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Stock Price: ${current_price:.2f}")
            print(f"Strike: ${strike_price:.2f}")
            print(f"Expiration: {expiration_date}")
            print(f"Entry Price: ${entry_price:.2f}")
            print(f"Contracts: {self.contracts_per_trade}")
            print(f"Cost: ${cost:.2f}")
            print(f"Remaining Capital: ${self.capital:,.2f}")

            if greeks_data:
                print(f"\nGreeks at Entry:")
                print(f"  Delta: {greeks_data['delta']:>8.4f}")
                print(f"  Gamma: {greeks_data['gamma']:>8.4f}")
                print(f"  Vega:  {greeks_data['vega']:>8.4f}")
                print(f"  Theta: {greeks_data['theta']:>8.4f}")
                print(f"  Rho:   {greeks_data['rho']:>8.4f}")

            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n[{current_time.strftime('%H:%M:%S')}] ERROR opening position: {e}")

    def check_positions(self):
        """Check all open positions for stop-loss, take-profit, or expiration"""
        current_time = datetime.now()
        positions_to_close = []

        for position in self.open_positions:
            try:
                # Get current option price
                option_data = option_price(self.ticker, position.expiration_date,
                                          position.strike_price, position.option_type)
                current_price = option_data['lastPrice']

                # Calculate P&L
                pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100

                # Check stop-loss
                if current_price <= position.entry_price * (1 - self.stoploss_percent / 100):
                    revenue = current_price * 100 * position.contracts
                    self.capital += revenue

                    position.exit_date = current_time
                    position.exit_price = current_price
                    position.exit_reason = 'stoploss'
                    position.pnl_percent = pnl_percent
                    position.pnl_dollar = (current_price - position.entry_price) * 100 * position.contracts

                    positions_to_close.append(position)

                    print(f"\n{'='*80}")
                    print(f"STOP-LOSS TRIGGERED: {position.option_type.upper()}")
                    print(f"{'='*80}")
                    print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
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

                    position.exit_date = current_time
                    position.exit_price = current_price
                    position.exit_reason = 'takeprofit'
                    position.pnl_percent = pnl_percent
                    position.pnl_dollar = (current_price - position.entry_price) * 100 * position.contracts

                    positions_to_close.append(position)

                    print(f"\n{'='*80}")
                    print(f"TAKE-PROFIT TRIGGERED: {position.option_type.upper()}")
                    print(f"{'='*80}")
                    print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Strike: ${position.strike_price:.2f} exp {position.expiration_date}")
                    print(f"Entry: ${position.entry_price:.2f} → Exit: ${current_price:.2f}")
                    print(f"P&L: {pnl_percent:.1f}% (${position.pnl_dollar:.2f})")
                    print(f"Revenue: ${revenue:.2f}")
                    print(f"Capital: ${self.capital:,.2f}")
                    print(f"{'='*80}\n")

                # Check expiration (within 1 day)
                elif (position.strike_date_obj - current_time).days <= 0:
                    revenue = current_price * 100 * position.contracts
                    self.capital += revenue

                    position.exit_date = current_time
                    position.exit_price = current_price
                    position.exit_reason = 'expiration'
                    position.pnl_percent = pnl_percent
                    position.pnl_dollar = (current_price - position.entry_price) * 100 * position.contracts

                    positions_to_close.append(position)

                    print(f"\n{'='*80}")
                    print(f"OPTION EXPIRED: {position.option_type.upper()}")
                    print(f"{'='*80}")
                    print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Strike: ${position.strike_price:.2f}")
                    print(f"Entry: ${position.entry_price:.2f} → Exit: ${current_price:.2f}")
                    print(f"P&L: {pnl_percent:.1f}% (${position.pnl_dollar:.2f})")
                    print(f"Revenue: ${revenue:.2f}")
                    print(f"Capital: ${self.capital:,.2f}")
                    print(f"{'='*80}\n")

            except Exception as e:
                print(f"  Warning: Could not check position {position.option_type} ${position.strike_price}: {e}")
                continue

        # Move closed positions
        for position in positions_to_close:
            self.open_positions.remove(position)
            self.closed_positions.append(position)

    def update_capital_display(self):
        """Update and display current capital and portfolio value"""
        current_time = datetime.now()

        # Calculate total portfolio value (capital + open positions current value)
        portfolio_value = self.capital

        for position in self.open_positions:
            try:
                option_data = option_price(self.ticker, position.expiration_date,
                                          position.strike_price, position.option_type)
                current_price = option_data['lastPrice']
                position_value = current_price * 100 * position.contracts
                portfolio_value += position_value
            except:
                # If can't get price, use entry price
                portfolio_value += position.cost

        total_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100

        print(f"\n[{current_time.strftime('%H:%M:%S')}] PORTFOLIO UPDATE")
        print(f"  Cash: ${self.capital:,.2f}")
        print(f"  Open Positions: {len(self.open_positions)}")
        print(f"  Total Portfolio Value: ${portfolio_value:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")

        if self.open_positions:
            print(f"\n  Open Positions Details:")
            for i, pos in enumerate(self.open_positions, 1):
                try:
                    option_data = option_price(self.ticker, pos.expiration_date,
                                              pos.strike_price, pos.option_type)
                    current_price = option_data['lastPrice']
                    pnl = ((current_price - pos.entry_price) / pos.entry_price) * 100
                    print(f"    {i}. {pos.option_type.upper()} ${pos.strike_price} exp {pos.expiration_date}: "
                          f"${current_price:.2f} ({pnl:+.1f}%)")
                except:
                    print(f"    {i}. {pos.option_type.upper()} ${pos.strike_price} exp {pos.expiration_date}: "
                          f"(price unavailable)")
        print()

        self.last_capital_update = current_time

    def run(self):
        """
        Run the live options trading model.

        This starts a continuous loop that:
        1. Monitors Fourier signals via LiveFourierMonitor (in background)
        2. Checks positions every minute
        3. Updates capital display every minute
        4. Pauses when market is closed and resumes when it opens
        """
        print(f"Starting live options trading model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Press Ctrl+C to stop\n")

        # Create LiveFourierMonitor with our callback
        monitor = LiveFourierMonitor(
            ticker=self.ticker,
            n_harmonics=self.n_harmonics,
            smoothing_sigma=self.smoothing_sigma,
            overbought_threshold=self.overbought_threshold,
            oversold_threshold=self.oversold_threshold,
            tick_size=self.tick_size,
            lookback_period=self.lookback_period,
            update_interval=self.update_interval,
            enable_alerts=True,
            alert_sound=False,
            alert_callback=self.handle_signal
        )

        try:
            iteration = 0
            market_was_open = self.is_market_open()

            while True:
                iteration += 1
                current_time = datetime.now()
                market_is_open = self.is_market_open()

                # Check if market status changed
                if market_is_open and not market_was_open:
                    print(f"\n{'='*80}")
                    print(f"[{current_time.strftime('%H:%M:%S')}] Market opened! Resuming trading...")
                    print(f"{'='*80}\n")
                elif not market_is_open and market_was_open:
                    print(f"\n{'='*80}")
                    print(f"[{current_time.strftime('%H:%M:%S')}] Market closed. Pausing trading...")
                    print(f"Will resume when market opens.")
                    print(f"{'='*80}\n")

                market_was_open = market_is_open

                # Only trade and monitor when market is open
                if market_is_open:
                    # Run single Fourier update (this will trigger callback if signal detected)
                    monitor.run_single_update()

                    # Check existing positions
                    if self.open_positions:
                        self.check_positions()

                    # Update capital display every minute
                    time_since_update = (current_time - self.last_capital_update).seconds
                    if time_since_update >= 60:
                        self.update_capital_display()

                    print(f"Next check in 60 seconds...")
                    time.sleep(60)
                else:
                    # Market is closed - check every 5 minutes to see if it opened
                    print(f"[{current_time.strftime('%H:%M:%S')}] Market closed. Next check in 5 minutes...")
                    time.sleep(300)  # 5 minutes

        except KeyboardInterrupt:
            print("\n\nModel stopped by user")
        except Exception as e:
            print(f"\nError during model execution: {e}")

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print final trading summary"""
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p.pnl_dollar and p.pnl_dollar > 0])
        losing_trades = len([p for p in self.closed_positions if p.pnl_dollar and p.pnl_dollar <= 0])

        # Calculate portfolio value
        portfolio_value = self.capital
        for position in self.open_positions:
            portfolio_value += position.cost  # Use cost as estimate

        total_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100

        print(f"\n{'='*80}")
        print(f"FINAL TRADING SUMMARY")
        print(f"{'='*80}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Cash: ${self.capital:,.2f}")
        print(f"Final Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"\nTrades:")
        print(f"  Total: {total_trades}")
        print(f"  Winners: {winning_trades}")
        print(f"  Losers: {losing_trades}")
        if total_trades > 0:
            print(f"  Win Rate: {(winning_trades/total_trades)*100:.1f}%")
        print(f"\nOpen Positions: {len(self.open_positions)}")
        print(f"{'='*80}\n")


def run_model(ticker: str,
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
    Run the live options trading model with Fourier analysis.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
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
        Seconds between Fourier updates (default: 60)
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
    >>> run_model('AAPL', n_harmonics=18, initial_capital=10000)
    """
    model = LiveOptionsModel(
        ticker=ticker,
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
    MODEL_SETTINGS_CACHE_FILE = os.path.expanduser('~/.live_fourier_model_settings.json')

    def load_model_cached_settings():
        """Load previously saved model settings from cache file."""
        if os.path.exists(MODEL_SETTINGS_CACHE_FILE):
            try:
                with open(MODEL_SETTINGS_CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cached settings: {e}")
        return {}

    def save_model_settings(settings: dict):
        """Save model settings to cache file for next run."""
        try:
            with open(MODEL_SETTINGS_CACHE_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")

    # Load cached settings
    cached = load_model_cached_settings()

    # Default values (will be overridden by cache if available)
    defaults = {
        'ticker': 'AAPL',
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
    print("\nLive Options Trading Model Configuration")
    print("=" * 60)
    if cached:
        print("(Previous settings loaded as defaults. Press Enter to use them)")
        print("=" * 60)

    try:
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
    save_model_settings(current_settings)

    # Display configuration summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Fourier: {n_harmonics} harmonics, sigma={smoothing_sigma}")
    print(f"Thresholds: Overbought={overbought_threshold}, Oversold={oversold_threshold}")
    print(f"Trading: {contracts_per_trade} contracts/trade, max {max_positions} positions")
    print(f"Risk: {stoploss_percent}% stop-loss, {takeprofit_percent}% take-profit")
    print(f"Options: {days_to_expiry}d expiry, {otm_percent}% OTM")
    print(f"Updates: Every {update_interval}s, {tick_size} tick size")
    print("=" * 60 + "\n")

    # Confirm before starting
    confirm = input("Start the model with these settings? (y/n, default: y): ").strip().lower()
    if confirm and confirm != 'y':
        print("Model cancelled.")
        sys.exit(0)

    # Run the model with configured parameters
    run_model(
        ticker=ticker,
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
