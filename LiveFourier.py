"""
LiveFourier.py - Real-time Fourier Analysis with Trading Signals

This module provides live stock price monitoring with Fourier analysis and
configurable trading signal alerts. It continuously updates the plot and
can trigger notifications when buy/sell conditions are met.

Usage:
    python LiveFourier.py

Or import and use programmatically:
    from LiveFourier import live_fourier_monitor

    live_fourier_monitor(
        ticker='AAPL',
        n_harmonics=17,
        smoothing_sigma=0,
        overbought_threshold=3.0,
        oversold_threshold=-5.0,
        tick_size='1m',
        update_interval=10,
        enable_alerts=True,
        alert_sound=True
    )
"""

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
import time
from typing import Optional, List, Union
import sys

# Import Fourier analysis functions from fourier.py
from fourier import (
    analyze_fourier,
    detect_overbought_oversold,
    FourierAnalysis,
    SignalPoint
)

# Set default renderer
pio.renderers.default = 'browser'


class LiveFourierMonitor:
    """
    Real-time stock price monitor with Fourier analysis and trading signal alerts.
    """

    def __init__(self,
                 ticker: str,
                 n_harmonics: int = 17,
                 smoothing_sigma: float = 0.0,
                 overbought_threshold: float = 3.0,
                 oversold_threshold: float = -5.0,
                 tick_size: str = '1m',
                 lookback_period: str = '5d',
                 update_interval: int = 10,
                 enable_alerts: bool = True,
                 alert_sound: bool = False,
                 alert_callback: Optional[callable] = None):
        """
        Initialize the live Fourier monitor.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        n_harmonics : int
            Number of Fourier harmonics to use (default: 11)
        smoothing_sigma : float
            Gaussian smoothing parameter (default: 0.0)
        overbought_threshold : float
            Detrended value above which triggers sell signal (default: 3.0)
        oversold_threshold : float
            Detrended value below which triggers buy signal (default: -5.0)
        tick_size : str
            Data interval: '1m', '5m', '15m', '30m', '1h', '1d' (default: '1m')
        lookback_period : str
            How much historical data to fetch: '1d', '5d', '1mo', etc. (default: '5d')
        update_interval : int
            Seconds between updates (default: 60)
        enable_alerts : bool
            Enable trading signal alerts (default: True)
        alert_sound : bool
            Play sound on alerts (default: False)
        alert_callback : callable, optional
            Custom function to call when signal is detected.
            Signature: callback(signal_type: str, signal: SignalPoint, analysis: FourierAnalysis)
        """
        self.ticker = ticker
        self.n_harmonics = n_harmonics
        self.smoothing_sigma = smoothing_sigma
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.tick_size = tick_size
        self.lookback_period = lookback_period
        self.update_interval = update_interval
        self.enable_alerts = enable_alerts
        self.alert_sound = alert_sound
        self.alert_callback = alert_callback

        # Track previous signals to avoid duplicate alerts
        self.last_signal = None
        self.last_signal_time = None

        print(f"\n{'='*80}")
        print(f"LIVE FOURIER MONITOR: {ticker}")
        print(f"{'='*80}")
        print(f"Harmonics: {n_harmonics} | Smoothing: {smoothing_sigma}")
        print(f"Overbought: {overbought_threshold} | Oversold: {oversold_threshold}")
        print(f"Tick Size: {tick_size} | Update Interval: {update_interval}s")
        print(f"Alerts: {'ENABLED' if enable_alerts else 'DISABLED'}")
        print(f"{'='*80}\n")

    def fetch_data(self) -> pd.DataFrame:
        """Fetch latest stock data from yfinance."""
        try:
            df = yf.download(
                tickers=self.ticker,
                period=self.lookback_period,
                interval=self.tick_size,
                progress=False
            )

            if df.empty:
                print(f"Warning: No data received for {self.ticker}")
                return None

            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def extract_prices(self, df: pd.DataFrame) -> tuple:
        """Extract OHLC prices from DataFrame."""
        try:
            # Handle MultiIndex columns (when downloading single ticker, yfinance may still use MultiIndex)
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

    def check_for_signals(self, analysis: FourierAnalysis) -> Optional[SignalPoint]:
        """
        Check if current conditions trigger a buy or sell signal.
        Returns the most recent signal if one exists.
        """
        signals = detect_overbought_oversold(
            analysis,
            self.overbought_threshold,
            self.oversold_threshold
        )

        if signals:
            # Return the most recent signal
            latest_signal = signals[-1]
            return latest_signal

        return None

    def trigger_alert(self, signal: SignalPoint, analysis: FourierAnalysis):
        """Trigger alert when a trading signal is detected."""
        current_time = datetime.now()

        # Avoid duplicate alerts for the same signal type within 5 minutes
        if (self.last_signal == signal.signal_type and
            self.last_signal_time and
            (current_time - self.last_signal_time).seconds < 300):
            return

        # Update last signal tracking
        self.last_signal = signal.signal_type
        self.last_signal_time = current_time

        # Format alert message
        alert_type = "🔴 SELL SIGNAL" if signal.signal_type == 'sell' else "🟢 BUY SIGNAL"

        print(f"\n{'='*80}")
        print(f"{alert_type} DETECTED!")
        print(f"{'='*80}")
        print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ticker: {self.ticker}")
        print(f"Current Price: ${signal.price:.2f}")
        print(f"Fourier Price: ${signal.fourier_value:.2f}")
        print(f"Detrended Value: ${signal.detrended_value:.2f}")
        print(f"Reason: {signal.reason}")
        print(f"Threshold: {self.overbought_threshold if signal.signal_type == 'sell' else self.oversold_threshold}")
        print(f"{'='*80}\n")

        # Play sound if enabled
        if self.alert_sound:
            self._play_alert_sound(signal.signal_type)

        # Call custom callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(signal.signal_type, signal, analysis)
            except Exception as e:
                print(f"Error in alert callback: {e}")

    def _play_alert_sound(self, signal_type: str):
        """Play system beep (cross-platform)."""
        try:
            # Simple cross-platform beep
            import sys
            if sys.platform == 'win32':
                import winsound
                frequency = 1000 if signal_type == 'buy' else 500
                winsound.Beep(frequency, 500)
            else:
                # Unix/Linux/Mac
                print('\a')  # Terminal bell
        except Exception:
            pass  # Silently fail if sound not available

    def create_plot(self, df: pd.DataFrame, analysis: FourierAnalysis,
                   latest_signal: Optional[SignalPoint] = None) -> go.Figure:
        """Create the live Plotly figure with Fourier analysis."""
        close_prices, open_prices, high_prices, low_prices, dates = self.extract_prices(df)

        if close_prices is None:
            return None

        # Convert dates to string labels for x-axis to eliminate gaps
        date_labels = [d.strftime('%Y-%m-%d %H:%M' if self.tick_size in ['1m', '2m', '5m', '15m', '30m', '1h'] else '%Y-%m-%d')
                       for d in dates]

        # Create integer indices for plotting (removes gaps automatically)
        x_indices = list(range(len(dates)))

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.08,
            subplot_titles=(
                f'{self.ticker} Live Price (Updates every {self.update_interval}s)',
                'Detrended Fourier Components'
            )
        )

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=x_indices,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name=self.ticker,
            increasing_line_color='green',
            decreasing_line_color='red',
            customdata=date_labels,
            hovertemplate='<b>Time</b>: %{customdata}<br><b>Open</b>: $%{open:.2f}<br><b>High</b>: $%{high:.2f}<br><b>Low</b>: $%{low:.2f}<br><b>Close</b>: $%{close:.2f}<extra></extra>'
        ), row=1, col=1)

        # Add Fourier curve to main plot
        fig.add_trace(go.Scatter(
            x=x_indices,
            y=analysis.fourier_prices,
            mode='lines',
            name=f'Fourier ({self.n_harmonics}h)',
            line=dict(color='cyan', width=2, dash='dash'),
            customdata=date_labels,
            hovertemplate='<b>Time</b>: %{customdata}<br><b>Fourier</b>: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)

        # Add detrended Fourier to bottom plot
        fig.add_trace(go.Scatter(
            x=x_indices,
            y=analysis.detrended_fourier,
            mode='lines',
            name='Detrended',
            line=dict(color='purple', width=2),
            customdata=date_labels,
            hovertemplate='<b>Time</b>: %{customdata}<br><b>Deviation</b>: $%{y:.2f}<extra></extra>',
            showlegend=False
        ), row=2, col=1)

        # Add threshold lines
        fig.add_hline(
            y=0, line_dash="dot", line_color="gray",
            opacity=0.5, row=2, col=1
        )
        fig.add_hline(
            y=self.overbought_threshold, line_dash="dash",
            line_color="red", opacity=0.7, row=2, col=1,
            annotation_text="Overbought (SELL)", annotation_position="right"
        )
        fig.add_hline(
            y=self.oversold_threshold, line_dash="dash",
            line_color="green", opacity=0.7, row=2, col=1,
            annotation_text="Oversold (BUY)", annotation_position="right"
        )

        # Add signal markers if there's a recent signal
        if latest_signal and len(dates) > 0:
            # Find the index closest to the signal date
            signal_idx = latest_signal.index if latest_signal.index < len(dates) else -1

            if signal_idx >= 0:
                marker_color = 'red' if latest_signal.signal_type == 'sell' else 'green'
                marker_symbol = 'triangle-down' if latest_signal.signal_type == 'sell' else 'triangle-up'

                fig.add_trace(go.Scatter(
                    x=[signal_idx],
                    y=[latest_signal.price],
                    mode='markers',
                    name=f'{latest_signal.signal_type.upper()} Signal',
                    marker=dict(color=marker_color, size=15, symbol=marker_symbol),
                    customdata=[date_labels[signal_idx]],
                    hovertemplate=f'<b>{latest_signal.signal_type.upper()}</b><br>Time: %{{customdata}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ), row=1, col=1)

        # Set up custom tick labels for x-axis
        n_ticks = min(10, len(date_labels))
        tick_step = max(1, len(date_labels) // n_ticks)
        tickvals = list(range(0, len(date_labels), tick_step))
        ticktext = [date_labels[i] for i in tickvals]

        # Update layout
        fig.update_layout(
            title=f'{self.ticker} Live Fourier Analysis',
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True,
            height=900,
            xaxis_rangeslider_visible=False
        )

        # Update axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Deviation ($)", row=2, col=1)

        fig.update_xaxes(
            title_text="Time",
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=-45,
            row=2, col=1
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=-45,
            row=1, col=1
        )

        return fig

    def run_single_update(self):
        """Run a single update cycle and return the figure."""
        # Fetch data
        df = self.fetch_data()
        if df is None or df.empty:
            return None

        # Extract prices
        result = self.extract_prices(df)
        if result is None:
            return None

        close_prices, _, _, _, dates = result

        # Perform Fourier analysis
        try:
            analysis = analyze_fourier(
                close_prices,
                dates,
                self.n_harmonics,
                self.smoothing_sigma
            )
        except Exception as e:
            print(f"Error in Fourier analysis: {e}")
            return None

        # Check for signals
        latest_signal = None
        if self.enable_alerts:
            latest_signal = self.check_for_signals(analysis)
            if latest_signal:
                self.trigger_alert(latest_signal, analysis)

        # Create and return plot
        fig = self.create_plot(df, analysis, latest_signal)

        # Print current status
        current_price = close_prices[-1]
        current_fourier = analysis.fourier_prices[-1]
        current_detrended = analysis.detrended_fourier[-1]

        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.ticker}: "
              f"Price=${current_price:.2f} | "
              f"Fourier=${current_fourier:.2f} | "
              f"Detrended=${current_detrended:.2f}")

        return fig

    def run(self, duration_minutes: Optional[int] = None):
        """
        Run the live monitor continuously.

        Parameters:
        -----------
        duration_minutes : int, optional
            Run for specified minutes, or indefinitely if None
        """
        start_time = datetime.now()
        iteration = 0

        print(f"Starting live monitor at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Press Ctrl+C to stop\n")

        try:
            while True:
                iteration += 1

                # Run update
                fig = self.run_single_update()

                # Show plot (will open in browser on first iteration)
                if fig:
                    if iteration == 1:
                        fig.show()
                    # Note: For continuous updates, you'd need a different approach
                    # like Dash or streamlit for live browser updates

                # Check duration limit
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        print(f"\nDuration limit reached ({duration_minutes} minutes)")
                        break

                # Wait before next update
                print(f"Next update in {self.update_interval} seconds...\n")
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
        except Exception as e:
            print(f"\nError during monitoring: {e}")

        print(f"\nMonitor ran for {iteration} iterations")
        print(f"Stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def live_fourier_monitor(ticker: str,
                         n_harmonics: int = 11,
                         smoothing_sigma: float = 0.0,
                         overbought_threshold: float = 3.0,
                         oversold_threshold: float = -5.0,
                         tick_size: str = '1m',
                         lookback_period: str = '5d',
                         update_interval: int = 60,
                         duration_minutes: Optional[int] = None,
                         enable_alerts: bool = True,
                         alert_sound: bool = False,
                         alert_callback: Optional[callable] = None):
    """
    Convenience function to start live Fourier monitoring.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    n_harmonics : int
        Number of Fourier harmonics (default: 11)
    smoothing_sigma : float
        Smoothing parameter (default: 0.0)
    overbought_threshold : float
        Sell signal threshold (default: 3.0)
    oversold_threshold : float
        Buy signal threshold (default: -5.0)
    tick_size : str
        Data interval: '1m', '5m', '15m', '30m', '1h', '1d' (default: '1m')
    lookback_period : str
        Historical data period: '1d', '5d', '1mo' (default: '5d')
    update_interval : int
        Seconds between updates (default: 60)
    duration_minutes : int, optional
        Run for specified minutes, or indefinitely if None
    enable_alerts : bool
        Enable trading signal alerts (default: True)
    alert_sound : bool
        Play sound on alerts (default: False)
    alert_callback : callable, optional
        Custom callback function for signals

    Example:
    --------
    >>> live_fourier_monitor('AAPL', n_harmonics=11, update_interval=30)
    """
    monitor = LiveFourierMonitor(
        ticker=ticker,
        n_harmonics=n_harmonics,
        smoothing_sigma=smoothing_sigma,
        overbought_threshold=overbought_threshold,
        oversold_threshold=oversold_threshold,
        tick_size=tick_size,
        lookback_period=lookback_period,
        update_interval=update_interval,
        enable_alerts=enable_alerts,
        alert_sound=alert_sound,
        alert_callback=alert_callback
    )

    monitor.run(duration_minutes=duration_minutes)


# Example alert callback
def example_alert_callback(signal_type: str, signal: SignalPoint, analysis: FourierAnalysis):
    """
    Example callback function that's called when a signal is detected.

    You can replace this with your own function to:
    - Send email/SMS notifications
    - Log to a database
    - Execute trades via API
    - Send webhook notifications
    """
    print(f"\n[CALLBACK] Custom alert handler triggered for {signal_type.upper()} signal")
    print(f"[CALLBACK] You could execute a trade here...")
    print(f"[CALLBACK] Signal strength: {abs(signal.detrended_value):.2f}")


if __name__ == "__main__":
    # Get user input or use defaults
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = input("Enter a stock ticker symbol (default: AAPL): ").strip().upper()
        if not ticker:
            ticker = "AAPL"

    # Prompt for parameters
    print("\nLive Fourier Monitor Configuration")
    print("-" * 40)

    try:
        n_harm = input(f"Harmonics (default: 17): ").strip()
        n_harmonics = int(n_harm) if n_harm else 17

        smooth = input(f"Smoothing sigma (default: 0): ").strip()
        smoothing_sigma = float(smooth) if smooth else 0.0

        overbought = input(f"Overbought threshold (default: 3.0): ").strip()
        overbought_threshold = float(overbought) if overbought else 3.0

        oversold = input(f"Oversold threshold (default: -5.0): ").strip()
        oversold_threshold = float(oversold) if oversold else -5.0

        tick = input(f"Tick size - 1m/5m/15m/30m/1h/1d (default: 1m): ").strip()
        tick_size = tick if tick else '1m'

        lookback = input(f"Lookback period - 1d/5d/1mo (default: 5d): ").strip()
        lookback_period = lookback if lookback else '5d'

        interval = input(f"Update interval in seconds (default: 10): ").strip()
        update_interval = int(interval) if interval else 10

        alerts = input(f"Enable alerts? (y/n, default: y): ").strip().lower()
        enable_alerts = alerts != 'n'

        sound = input(f"Alert sound? (y/n, default: n): ").strip().lower()
        alert_sound = sound == 'y'

    except ValueError:
        print("Invalid input, using defaults")
        n_harmonics = 11
        smoothing_sigma = 0.0
        overbought_threshold = 3.0
        oversold_threshold = -5.0
        tick_size = '1m'
        update_interval = 60
        enable_alerts = True
        alert_sound = False

    # Start monitoring
    live_fourier_monitor(
        ticker=ticker,
        n_harmonics=n_harmonics,
        smoothing_sigma=smoothing_sigma,
        overbought_threshold=overbought_threshold,
        oversold_threshold=oversold_threshold,
        tick_size=tick_size,
        update_interval=update_interval,
        enable_alerts=enable_alerts,
        alert_sound=alert_sound,
        alert_callback=None  # Use example_alert_callback for custom handling
    )
