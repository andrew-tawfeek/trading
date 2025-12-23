import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import Union, Tuple, Dict, List, Optional
from dataclasses import dataclass
from yfinance_cache import download_cached

# Set the default renderer to browser for Windows
pio.renderers.default = 'browser'


# ========================================
# Data Classes for Backtesting
# ========================================

@dataclass
class FourierAnalysis:
    """Container for Fourier analysis results"""
    dates: np.ndarray
    prices: np.ndarray
    fourier_prices: np.ndarray
    detrended_fourier: np.ndarray
    trend_coeffs: np.ndarray
    n_harmonics: int
    smoothing_sigma: float
    rmse: float


@dataclass
class SignalPoint:
    """Represents a buy/sell signal point"""
    date: datetime
    index: int
    price: float
    fourier_value: float
    detrended_value: float
    signal_type: str  # 'buy' or 'sell'
    reason: str  # 'oversold', 'overbought', 'peak', 'trough'


@dataclass
class FourierPeak:
    """Represents a peak or trough in the Fourier curve"""
    date: datetime
    index: int
    value: float
    is_peak: bool  # True for peak, False for trough
    price_at_point: float


# ========================================
# Core Fourier Functions
# ========================================

def fit_fourier(data: np.ndarray, n_harmonics: int = 10,
                smoothing_sigma: float = 0.0, return_detrended: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Fit a Fourier series to the data with optional smoothing.

    Parameters:
    -----------
    data : array-like
        Time series data to fit
    n_harmonics : int
        Number of Fourier harmonics to use for fitting
    smoothing_sigma : float
        Gaussian smoothing parameter. Higher values = smoother curve.
        0 = no smoothing, 1-3 = moderate smoothing, >5 = heavy smoothing
    return_detrended : bool, optional
        If True, also return the detrended (centered) Fourier component

    Returns:
    --------
    numpy.ndarray or tuple
        If return_detrended=False: Reconstructed signal using Fourier harmonics
        If return_detrended=True: (reconstructed signal, detrended component)
    """
    # Convert to numpy array to ensure compatibility
    data = np.array(data)

    # Detrend the data
    n = len(data)
    t = np.arange(0, n)

    # Linear detrending
    coeffs = np.polyfit(t, data, 1)
    trend = np.polyval(coeffs, t)
    detrended = data - trend

    # Compute FFT
    fft_vals = fft(detrended)

    # Keep only the specified number of harmonics
    fft_filtered = np.zeros_like(fft_vals)
    fft_filtered[:n_harmonics] = fft_vals[:n_harmonics]
    fft_filtered[-n_harmonics:] = fft_vals[-n_harmonics:]

    # Inverse FFT to get the filtered signal
    filtered_signal = np.real(ifft(fft_filtered))

    # Apply Gaussian smoothing if requested
    if smoothing_sigma > 0:
        filtered_signal = gaussian_filter1d(filtered_signal, sigma=smoothing_sigma)
        # Re-center after smoothing to ensure zero mean (removes any DC shift from smoothing)
        filtered_signal = filtered_signal - np.mean(filtered_signal)

    # Add the trend back
    reconstructed = filtered_signal + trend

    if return_detrended:
        return reconstructed, filtered_signal
    else:
        return reconstructed


def analyze_fourier(prices: np.ndarray, dates: pd.DatetimeIndex, n_harmonics: int = 10,
                   smoothing_sigma: float = 2.0) -> FourierAnalysis:
    """
    Perform comprehensive Fourier analysis on price data.

    Parameters:
    -----------
    prices : np.ndarray
        Price time series
    dates : pd.DatetimeIndex
        Corresponding dates
    n_harmonics : int
        Number of harmonics for Fourier fit
    smoothing_sigma : float
        Smoothing parameter for the Fourier curve

    Returns:
    --------
    FourierAnalysis
        Complete Fourier analysis results
    """
    # Fit Fourier
    fourier_prices, detrended_fourier = fit_fourier(
        prices, n_harmonics, smoothing_sigma, return_detrended=True
    )

    # Calculate trend coefficients
    n = len(prices)
    t = np.arange(0, n)
    trend_coeffs = np.polyfit(t, prices, 1)

    # Calculate RMSE
    residuals = prices - fourier_prices
    rmse = np.sqrt(np.mean(residuals**2))

    return FourierAnalysis(
        dates=np.array(dates),
        prices=prices,
        fourier_prices=fourier_prices,
        detrended_fourier=detrended_fourier,
        trend_coeffs=trend_coeffs,
        n_harmonics=n_harmonics,
        smoothing_sigma=smoothing_sigma,
        rmse=rmse
    )


# ========================================
# Peak Detection Functions
# ========================================

def find_fourier_peaks(analysis: FourierAnalysis,
                       prominence: float = 0.5) -> Tuple[List[FourierPeak], List[FourierPeak]]:
    """
    Find peaks and troughs in the detrended Fourier curve.

    Parameters:
    -----------
    analysis : FourierAnalysis
        Fourier analysis results
    prominence : float
        Minimum prominence for peak detection (helps filter noise)

    Returns:
    --------
    Tuple[List[FourierPeak], List[FourierPeak]]
        (peaks, troughs)
    """
    detrended = analysis.detrended_fourier

    # Find peaks (local maxima)
    peak_indices, _ = find_peaks(detrended, prominence=prominence)

    # Find troughs (local minima) by inverting the signal
    trough_indices, _ = find_peaks(-detrended, prominence=prominence)

    # Create FourierPeak objects
    peaks = [
        FourierPeak(
            date=analysis.dates[idx],
            index=idx,
            value=detrended[idx],
            is_peak=True,
            price_at_point=analysis.prices[idx]
        )
        for idx in peak_indices
    ]

    troughs = [
        FourierPeak(
            date=analysis.dates[idx],
            index=idx,
            value=detrended[idx],
            is_peak=False,
            price_at_point=analysis.prices[idx]
        )
        for idx in trough_indices
    ]

    return peaks, troughs


# ========================================
# Signal Detection Functions
# ========================================

def detect_overbought_oversold(analysis: FourierAnalysis,
                                overbought_threshold: float,
                                oversold_threshold: float) -> List[SignalPoint]:
    """
    Detect buy/sell signals based on overbought/oversold thresholds.

    Parameters:
    -----------
    analysis : FourierAnalysis
        Fourier analysis results
    overbought_threshold : float
        Detrended Fourier value above which is considered overbought (sell signal)
    oversold_threshold : float
        Detrended Fourier value below which is considered oversold (buy signal)

    Returns:
    --------
    List[SignalPoint]
        List of signal points in chronological order
    """
    signals = []
    detrended = analysis.detrended_fourier

    for i in range(1, len(detrended)):
        # Check for oversold crossing (buy signal)
        if detrended[i-1] > oversold_threshold and detrended[i] <= oversold_threshold:
            signals.append(SignalPoint(
                date=analysis.dates[i],
                index=i,
                price=analysis.prices[i],
                fourier_value=analysis.fourier_prices[i],
                detrended_value=detrended[i],
                signal_type='buy',
                reason='oversold'
            ))

        # Check for overbought crossing (sell signal)
        elif detrended[i-1] < overbought_threshold and detrended[i] >= overbought_threshold:
            signals.append(SignalPoint(
                date=analysis.dates[i],
                index=i,
                price=analysis.prices[i],
                fourier_value=analysis.fourier_prices[i],
                detrended_value=detrended[i],
                signal_type='sell',
                reason='overbought'
            ))

    return signals


def detect_peak_signals(analysis: FourierAnalysis,
                        prominence: float = 0.5) -> List[SignalPoint]:
    """
    Detect buy/sell signals based on Fourier peaks and troughs.

    Parameters:
    -----------
    analysis : FourierAnalysis
        Fourier analysis results
    prominence : float
        Minimum prominence for peak detection

    Returns:
    --------
    List[SignalPoint]
        List of signal points (troughs = buy, peaks = sell)
    """
    peaks, troughs = find_fourier_peaks(analysis, prominence)

    signals = []

    # Troughs are buy signals
    for trough in troughs:
        signals.append(SignalPoint(
            date=trough.date,
            index=trough.index,
            price=trough.price_at_point,
            fourier_value=analysis.fourier_prices[trough.index],
            detrended_value=trough.value,
            signal_type='buy',
            reason='trough'
        ))

    # Peaks are sell signals
    for peak in peaks:
        signals.append(SignalPoint(
            date=peak.date,
            index=peak.index,
            price=peak.price_at_point,
            fourier_value=analysis.fourier_prices[peak.index],
            detrended_value=peak.value,
            signal_type='sell',
            reason='peak'
        ))

    # Sort by date
    signals.sort(key=lambda x: x.date)

    return signals


# ========================================
# Backtesting Functions
# ========================================

def backtest_signals(signals: List[SignalPoint],
                     initial_capital: float = 10000.0,
                     shares_per_trade: Optional[int] = None) -> Dict:
    """
    Backtest a trading strategy based on signal points.

    Parameters:
    -----------
    signals : List[SignalPoint]
        List of buy/sell signals
    initial_capital : float
        Starting capital
    shares_per_trade : int, optional
        Fixed number of shares to trade. If None, uses all available capital.

    Returns:
    --------
    Dict
        Backtesting results including final portfolio value, trades, and metrics
    """
    capital = initial_capital
    shares = 0
    trades = []
    portfolio_values = []

    for signal in signals:
        if signal.signal_type == 'buy' and capital > 0:
            if shares_per_trade:
                shares_to_buy = min(shares_per_trade, int(capital / signal.price))
            else:
                shares_to_buy = int(capital / signal.price)

            cost = shares_to_buy * signal.price
            if shares_to_buy > 0:
                shares += shares_to_buy
                capital -= cost
                trades.append({
                    'date': signal.date,
                    'type': 'BUY',
                    'price': signal.price,
                    'shares': shares_to_buy,
                    'cost': cost,
                    'reason': signal.reason
                })

        elif signal.signal_type == 'sell' and shares > 0:
            if shares_per_trade:
                shares_to_sell = min(shares_per_trade, shares)
            else:
                shares_to_sell = shares

            revenue = shares_to_sell * signal.price
            shares -= shares_to_sell
            capital += revenue
            trades.append({
                'date': signal.date,
                'type': 'SELL',
                'price': signal.price,
                'shares': shares_to_sell,
                'revenue': revenue,
                'reason': signal.reason
            })

        # Track portfolio value
        portfolio_value = capital + (shares * signal.price)
        portfolio_values.append({
            'date': signal.date,
            'value': portfolio_value
        })

    # Calculate final portfolio value (at last signal price)
    final_price = signals[-1].price if signals else 0
    final_value = capital + (shares * final_price)

    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'final_capital': capital,
        'final_shares': shares,
        'total_return': ((final_value - initial_capital) / initial_capital) * 100,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'num_trades': len(trades)
    }


# ========================================
# Plotting Functions
# ========================================

def plot_stock(ticker: str, start_date: str, end_date: str, tick_size: str = '1d',
               fourier: bool = False, n_harmonics: Union[int, List[int]] = 10,
               smoothing_sigma: float = 2.0, use_candlestick: bool = True,
               overbought_threshold: Optional[float] = None,
               oversold_threshold: Optional[float] = None,
               show_signals: bool = False) -> go.Figure:
    """
    Create an interactive plot of stock data with optional Fourier curve fitting.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    tick_size : str, optional
        Data interval ('1d', '1h', '5m', etc.). Default is '1d'
    fourier : bool, optional
        If True, fits and displays Fourier curve(s). Default is False
    n_harmonics : int or list of int, optional
        Number of Fourier harmonics to use for fitting.
        Can be a single int (e.g., 10) or a list of ints (e.g., [5, 10, 15, 20]).
    smoothing_sigma : float, optional
        Smoothing parameter for Fourier curves. Higher = smoother. Default is 2.0
    use_candlestick : bool, optional
        If True, use candlestick chart. If False, use line chart. Default is True
    overbought_threshold : float, optional
        Threshold for overbought condition in detrended plot
    oversold_threshold : float, optional
        Threshold for oversold condition in detrended plot
    show_signals : bool, optional
        If True, show buy/sell signals on the chart. Default is False

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """

    # Download stock data (with caching)
    stock_data = download_cached(ticker, start=start_date, end=end_date, interval=tick_size, progress=False, auto_adjust=False)

    if stock_data.empty:
        raise ValueError(f"No data found for {ticker} in the specified date range")

    # Extract prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        close_prices = stock_data['Close'][ticker].values
        open_prices = stock_data['Open'][ticker].values
        high_prices = stock_data['High'][ticker].values
        low_prices = stock_data['Low'][ticker].values
    else:
        close_prices = stock_data['Close'].values
        open_prices = stock_data['Open'].values
        high_prices = stock_data['High'].values
        low_prices = stock_data['Low'].values

    dates = stock_data.index

    # Convert dates to string labels for x-axis to eliminate gaps
    date_labels = [d.strftime('%Y-%m-%d %H:%M' if tick_size in ['1m', '2m', '5m', '15m', '30m', '1h', '90m'] else '%Y-%m-%d')
                   for d in dates]

    # Create integer indices for plotting (removes gaps automatically)
    x_indices = list(range(len(dates)))

    # Determine if we need subplots based on fourier parameter
    if fourier:
        # Create figure with subplots: main plot + detrended Fourier
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.1,
            subplot_titles=(f'{ticker} Stock Price', 'Detrended Fourier Components'),
            shared_xaxes=True
        )
    else:
        fig = go.Figure()

    # Add stock price chart (candlestick or line)
    if use_candlestick:
        fig.add_trace(go.Candlestick(
            x=x_indices,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name=f'{ticker}',
            increasing_line_color='green',
            decreasing_line_color='red',
            customdata=date_labels,
            hovertemplate='<b>Date</b>: %{customdata}<br><b>Open</b>: $%{open:.2f}<br><b>High</b>: $%{high:.2f}<br><b>Low</b>: $%{low:.2f}<br><b>Close</b>: $%{close:.2f}<extra></extra>'
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=x_indices,
            y=close_prices,
            mode='lines',
            name=f'{ticker} Close Price',
            line=dict(color='blue', width=2),
            customdata=date_labels,
            hovertemplate='<b>Date</b>: %{customdata}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)

    # Add Fourier curve(s) if requested
    if fourier:
        # Convert n_harmonics to list if it's a single integer
        harmonics_list = n_harmonics if isinstance(n_harmonics, list) else [n_harmonics]

        # Color palette for different Fourier curves
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

        # Store analyses for signal detection
        analyses = []

        # Add a trace for each harmonic level
        for idx, n_harm in enumerate(harmonics_list):
            analysis = analyze_fourier(close_prices, dates, n_harm, smoothing_sigma)
            analyses.append(analysis)

            color = colors[idx % len(colors)]

            # Add Fourier curve to main plot
            fig.add_trace(go.Scatter(
                x=x_indices,
                y=analysis.fourier_prices,
                mode='lines',
                name=f'Fourier {n_harm}h',
                line=dict(color=color, width=2, dash='dash'),
                customdata=date_labels,
                hovertemplate=f'<b>Date</b>: %{{customdata}}<br><b>Fourier {n_harm}h</b>: $%{{y:.2f}}<extra></extra>',
                visible=True,
                legendgroup=f'fourier_{n_harm}'
            ), row=1, col=1)

            # Add detrended Fourier curve to bottom plot (centered around 0)
            fig.add_trace(go.Scatter(
                x=x_indices,
                y=analysis.detrended_fourier,
                mode='lines',
                name=f'Fourier {n_harm}h (detrended)',
                line=dict(color=color, width=2),
                customdata=date_labels,
                hovertemplate=f'<b>Date</b>: %{{customdata}}<br><b>Deviation</b>: $%{{y:.2f}}<extra></extra>',
                visible=True,
                legendgroup=f'fourier_{n_harm}',
                showlegend=False
            ), row=2, col=1)

            print(f"Fourier Fit ({n_harm} harmonics, σ={smoothing_sigma}) RMSE: ${analysis.rmse:.2f}")

        # Add horizontal line at y=0 for reference in the detrended plot
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)

        # Add overbought/oversold threshold lines
        if overbought_threshold is not None:
            fig.add_hline(y=overbought_threshold, line_dash="dash", line_color="red",
                         opacity=0.7, row=2, col=1,
                         annotation_text="Overbought", annotation_position="right")

        if oversold_threshold is not None:
            fig.add_hline(y=oversold_threshold, line_dash="dash", line_color="green",
                         opacity=0.7, row=2, col=1,
                         annotation_text="Oversold", annotation_position="right")

        # Add buy/sell signals if requested
        if show_signals and (overbought_threshold is not None or oversold_threshold is not None):
            # Use the first analysis for signal detection
            primary_analysis = analyses[0]

            if overbought_threshold is not None and oversold_threshold is not None:
                signals = detect_overbought_oversold(
                    primary_analysis, overbought_threshold, oversold_threshold
                )

                # Separate buy and sell signals
                buy_signals = [s for s in signals if s.signal_type == 'buy']
                sell_signals = [s for s in signals if s.signal_type == 'sell']

                # Add buy markers
                if buy_signals:
                    buy_indices = [s.index for s in buy_signals]
                    buy_dates_str = [date_labels[s.index] for s in buy_signals]
                    fig.add_trace(go.Scatter(
                        x=buy_indices,
                        y=[s.price for s in buy_signals],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='green', size=12, symbol='triangle-up'),
                        customdata=buy_dates_str,
                        hovertemplate='<b>BUY</b><br>Date: %{customdata}<br>Price: $%{y:.2f}<extra></extra>'
                    ), row=1, col=1)

                # Add sell markers
                if sell_signals:
                    sell_indices = [s.index for s in sell_signals]
                    sell_dates_str = [date_labels[s.index] for s in sell_signals]
                    fig.add_trace(go.Scatter(
                        x=sell_indices,
                        y=[s.price for s in sell_signals],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='red', size=12, symbol='triangle-down'),
                        customdata=sell_dates_str,
                        hovertemplate='<b>SELL</b><br>Date: %{customdata}<br>Price: $%{y:.2f}<extra></extra>'
                    ), row=1, col=1)

                print(f"\nDetected {len(buy_signals)} buy signals and {len(sell_signals)} sell signals")

    # Set up custom tick labels for x-axis using date strings
    # Show fewer ticks for better readability
    n_ticks = min(10, len(date_labels))
    tick_step = max(1, len(date_labels) // n_ticks)
    tickvals = list(range(0, len(date_labels), tick_step))
    ticktext = [date_labels[i] for i in tickvals]

    # Update layout based on whether we have subplots
    if fourier:
        fig.update_layout(
            title=f'{ticker} Stock Price ({start_date} to {end_date})',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            height=800,
            xaxis_rangeslider_visible=False
        )
        # Update y-axes labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Deviation ($)", row=2, col=1)
        # Update x-axes with custom tick labels
        fig.update_xaxes(
            title_text="Date",
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
    else:
        fig.update_layout(
            title=f'{ticker} Stock Price ({start_date} to {end_date})',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            height=600,
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=-45,
                rangeslider=dict(visible=True)
            )
        )

    return fig


# ========================================
# Utility Functions for Backtesting
# ========================================

def get_stock_data(ticker: str, start_date: str, end_date: str,
                   tick_size: str = '1d') -> pd.DataFrame:
    """Download and return stock data (with caching)."""
    data = download_cached(ticker, start=start_date, end=end_date, interval=tick_size, progress=False, auto_adjust=False)
    return data


def run_fourier_backtest(ticker: str, start_date: str, end_date: str,
                         n_harmonics: int = 10, smoothing_sigma: float = 2.0,
                         overbought_threshold: float = 5.0,
                         oversold_threshold: float = -5.0,
                         initial_capital: float = 10000.0,
                         tick_size: str = '1d') -> Dict:
    """
    Run a complete Fourier-based backtest.

    Parameters:
    -----------
    ticker : str
        Stock ticker
    start_date, end_date : str
        Date range
    n_harmonics : int
        Number of harmonics
    smoothing_sigma : float
        Smoothing parameter
    overbought_threshold, oversold_threshold : float
        Trading thresholds
    initial_capital : float
        Starting capital
    tick_size : str
        Data interval

    Returns:
    --------
    Dict
        Complete backtesting results including analysis and trades
    """
    # Download data
    stock_data = get_stock_data(ticker, start_date, end_date, tick_size)

    if stock_data.empty:
        raise ValueError(f"No data found for {ticker}")

    # Extract prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        prices = stock_data['Close'][ticker].values
    else:
        prices = stock_data['Close'].values

    dates = stock_data.index

    # Perform Fourier analysis
    analysis = analyze_fourier(prices, dates, n_harmonics, smoothing_sigma)

    # Detect signals
    signals = detect_overbought_oversold(analysis, overbought_threshold, oversold_threshold)

    # Run backtest
    backtest_results = backtest_signals(signals, initial_capital)

    # Combine results
    return {
        'ticker': ticker,
        'date_range': (start_date, end_date),
        'fourier_analysis': analysis,
        'signals': signals,
        'backtest': backtest_results,
        'parameters': {
            'n_harmonics': n_harmonics,
            'smoothing_sigma': smoothing_sigma,
            'overbought_threshold': overbought_threshold,
            'oversold_threshold': oversold_threshold
        }
    }


def print_backtest_summary(results: Dict):
    """Print a formatted summary of backtest results."""
    bt = results['backtest']
    params = results['parameters']

    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY: {results['ticker']}")
    print(f"{'='*60}")
    print(f"Date Range: {results['date_range'][0]} to {results['date_range'][1]}")
    print(f"\nParameters:")
    print(f"  Harmonics: {params['n_harmonics']}")
    print(f"  Smoothing σ: {params['smoothing_sigma']}")
    print(f"  Overbought: {params['overbought_threshold']}")
    print(f"  Oversold: {params['oversold_threshold']}")
    print(f"\nResults:")
    print(f"  Initial Capital: ${bt['initial_capital']:,.2f}")
    print(f"  Final Value: ${bt['final_value']:,.2f}")
    print(f"  Total Return: {bt['total_return']:.2f}%")
    print(f"  Number of Trades: {bt['num_trades']}")
    print(f"  Final Shares Held: {bt['final_shares']}")
    print(f"  Final Cash: ${bt['final_capital']:,.2f}")
    print(f"{'='*60}\n")

    if bt['trades']:
        print("Recent Trades:")
        for trade in bt['trades'][-5:]:
            # Convert numpy datetime64 to pandas Timestamp for formatting
            trade_date = pd.Timestamp(trade['date'])
            print(f"  {trade_date.strftime('%Y-%m-%d')} | {trade['type']:4s} | "
                  f"{trade['shares']:4d} shares @ ${trade['price']:7.2f} | {trade['reason']}")


# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    results = run_fourier_options_backtest(
        ticker='AAPL',
        start_date='2025-10-01',
        end_date='2025-12-01',
        n_harmonics=18,
        smoothing_sigma=0,
        overbought_threshold=9,
        oversold_threshold=-8,
        contracts_per_trade=1,
        max_positions=2,
        tick_size='1d',
        stoploss_percent=50,
        takeprofit_percent=50,
        days_to_expiry=30,
        otm_percent=2.0,
        initial_capital=10000,
        verbose = True
    )

    print(f"Total Return: {results.total_return:.2f}%")
    print(f"Win Rate: {results.win_rate:.1f}%")



    # # Example 1: Basic candlestick plot with Fourier
    # print("\n=== Example 1: Candlestick with Fourier ===")
    # fig1 = plot_stock(
    #     'AAPL',
    #     '2025-01-28',
    #     '2025-11-01',
    #     tick_size='1d',
    #     fourier=True,
    #     n_harmonics=[5, 10, 18, 20, 30],
    #     smoothing_sigma=0.0,
    #     use_candlestick=True,
    #     overbought_threshold=9,
    #     oversold_threshold=-8,
    #     show_signals=True
    # )
    # fig1.show()

    # Example 2: Run a complete backtest
    # print("\n=== Example 2: Fourier Backtesting ===")



    # for n in range(1,20):
    #     print(f"\n--- Running backtest with {n} harmonics ---")
    #     backtest_results = run_fourier_backtest(
    #         ticker='AAPL',
    #         start_date='2024-01-01',
    #         end_date='2024-12-20',
    #         n_harmonics=n,
    #         smoothing_sigma=1.0,
    #         overbought_threshold=8.0,
    #         oversold_threshold=-8.0,
    #         initial_capital=10000.0
    #     )
    #     print_backtest_summary(backtest_results)

    # Example 3: Analyze peaks
    # analysis = backtest_results['fourier_analysis']
    # peaks, troughs = find_fourier_peaks(analysis, prominence=1.0)
    # print(f"\nFound {len(peaks)} peaks and {len(troughs)} troughs")


    #TODO ITERATE OVER HARMONICS AND OTHER VARIABLES TO FIND IDEAL FOR STOCKS, THEN FIND IDEAL FOR OPTIONS


# ========================================
# Options Trading Data Classes
# ========================================

@dataclass
class OptionTrade:
    """Represents a single option trade"""
    entry_date: datetime
    entry_index: int
    option_type: str  # 'call' or 'put'
    strike_price: float
    expiration_date: str
    entry_price: float
    entry_stock_price: float
    signal_reason: str
    greeks_at_entry: Dict

    exit_date: Optional[datetime] = None
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None
    exit_stock_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stoploss', 'takeprofit', 'expiration', 'signal'
    pnl_percent: Optional[float] = None
    pnl_dollar: Optional[float] = None
    days_held: Optional[int] = None
    greeks_at_exit: Optional[Dict] = None


@dataclass
class OptionsBacktestResults:
    """Container for options backtesting results"""
    ticker: str
    date_range: Tuple[str, str]
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_pnl_percent: float
    average_days_held: float
    trades: List[OptionTrade]
    capital_history: List[Dict]
    parameters: Dict
    fourier_analysis: FourierAnalysis
    signals: List[SignalPoint]


# ========================================
# Options Backtesting Core Functions
# ========================================

def get_option_strike_price(stock_price: float, option_type: str,
                            otm_percent: float = 2.0) -> float:
    """
    Calculate an appropriate strike price for an option, rounded to match
    typical option strike spacing.

    Parameters:
    -----------
    stock_price : float
        Current stock price
    option_type : str
        'call' or 'put'
    otm_percent : float
        Percentage out-of-the-money (default: 2.0%)

    Returns:
    --------
    float
        Strike price rounded to typical option strike increments:
        - $0.50 increments for stocks under $25
        - $1.00 increments for stocks $25-$200
        - $2.50 increments for stocks $200-$500
        - $5.00 increments for stocks over $500
    """
    if option_type == 'call':
        # For calls, strike above current price
        strike = stock_price * (1 + otm_percent / 100)
    else:  # put
        # For puts, strike below current price
        strike = stock_price * (1 - otm_percent / 100)

    # Determine rounding increment based on stock price
    # These are typical market conventions for strike spacing
    if stock_price < 25:
        increment = 0.50
    elif stock_price < 200:
        increment = 1.00
    elif stock_price < 500:
        increment = 2.50
    else:
        increment = 5.00

    # Round to nearest increment
    rounded_strike = round(strike / increment) * increment

    return rounded_strike


def get_option_expiration(current_date: datetime, days_to_expiry: int = 30, ticker: str = None) -> str:
    """
    Get the option expiration date approximately days_to_expiry days from current_date.

    For historical backtesting/replay, this calculates the next Friday that's closest
    to the target expiration date. This approximates the standard weekly/monthly option
    expiration pattern without relying on current live expiration dates.

    Parameters:
    -----------
    current_date : datetime
        Current date (can be datetime or numpy.datetime64)
    days_to_expiry : int
        Target days until expiration (default: 30)
    ticker : str, optional
        Stock ticker symbol. Currently unused - kept for backward compatibility.
        In historical replay, we calculate expiration dates rather than querying live data.

    Returns:
    --------
    str
        Expiration date in 'YYYY-MM-DD' format
    """
    from datetime import timedelta, datetime as dt
    import pandas as pd

    # Convert numpy.datetime64 to Python datetime if needed
    if isinstance(current_date, pd.Timestamp) or hasattr(current_date, 'to_pydatetime'):
        current_date = current_date.to_pydatetime()
    elif not isinstance(current_date, dt):
        # Try converting with pandas
        current_date = pd.Timestamp(current_date).to_pydatetime()

    # Calculate target expiration date
    target_expiration = current_date + timedelta(days=days_to_expiry)

    # Find the next Friday (options typically expire on Fridays)
    # weekday(): Monday=0, Tuesday=1, ..., Friday=4, Saturday=5, Sunday=6
    days_until_friday = (4 - target_expiration.weekday()) % 7

    # If target is already a Friday, use it; otherwise find next Friday
    if days_until_friday == 0 and target_expiration.weekday() != 4:
        days_until_friday = 7

    expiration = target_expiration + timedelta(days=days_until_friday)

    return expiration.strftime('%Y-%m-%d')


def backtest_options_signals(ticker: str,
                             signals: List[SignalPoint],
                             analysis: FourierAnalysis,
                             initial_capital: float = 10000.0,
                             contracts_per_trade: int = 1,
                             stoploss_percent: float = 50.0,
                             takeprofit_percent: float = 50.0,
                             days_to_expiry: int = 30,
                             otm_percent: float = 2.0,
                             max_positions: int = 1,
                             verbose: bool = True,
                             use_cached_pricing: bool = True) -> OptionsBacktestResults:
    """
    Backtest an options trading strategy based on Fourier signals.

    This function simulates buying calls on bullish signals and puts on bearish signals,
    then monitors each position until it hits stop-loss, take-profit, expires, or
    receives an opposite signal.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    signals : List[SignalPoint]
        List of buy/sell signals from Fourier analysis
    analysis : FourierAnalysis
        Fourier analysis results
    initial_capital : float
        Starting capital (default: $10,000)
    contracts_per_trade : int
        Number of contracts per trade (default: 1)
        Each contract represents 100 shares
    stoploss_percent : float
        Stop-loss percentage (default: 50%)
    takeprofit_percent : float
        Take-profit percentage (default: 50%)
    days_to_expiry : int
        Days to option expiration (default: 30)
    otm_percent : float
        Out-of-the-money percentage for strike selection (default: 2%)
    max_positions : int
        Maximum number of open positions at once (default: 1)
    verbose : bool
        Print detailed trade information (default: True)

    Returns:
    --------
    OptionsBacktestResults
        Complete backtesting results
    """
    from functions import option_price_historical, greeks_historical

    # Use regular tqdm (auto mode) which handles both terminal and notebook
    from tqdm.auto import tqdm

    capital = initial_capital
    open_positions: List[OptionTrade] = []
    closed_trades: List[OptionTrade] = []
    capital_history = []

    # Cache for option prices to avoid redundant calculations
    # Key: (expiration_date, strike_price, option_type, current_date)
    option_price_cache = {}
    greeks_cache = {}

    if verbose:
        print(f"\n{'='*80}")
        print(f"OPTIONS BACKTESTING: {ticker}")
        print(f"{'='*80}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Contracts per Trade: {contracts_per_trade}")
        print(f"Stop-Loss: {stoploss_percent}% | Take-Profit: {takeprofit_percent}%")
        print(f"Days to Expiry: {days_to_expiry} | OTM: {otm_percent}%")
        print(f"Max Open Positions: {max_positions}")
        print(f"{'='*80}\n")

    # Create progress bar
    progress_bar = tqdm(total=len(signals), desc="Processing signals", unit="signal",
                       leave=True, position=0) if verbose else None

    # Process each signal
    for signal_idx, signal in enumerate(signals):
        if progress_bar:
            progress_bar.update(1)

        # Convert numpy.datetime64 to Python datetime if needed
        current_date = signal.date
        if hasattr(current_date, 'to_pydatetime'):
            current_date = current_date.to_pydatetime()
        elif not isinstance(current_date, datetime):
            import pandas as pd
            current_date = pd.Timestamp(current_date).to_pydatetime()

        current_price = signal.price

        # Check existing positions for exit conditions
        positions_to_close = []

        for pos in open_positions:
            # Get current option price (with caching)
            try:
                cache_key = (pos.expiration_date, pos.strike_price, pos.option_type, current_date.strftime('%Y-%m-%d'))
                if cache_key in option_price_cache:
                    current_option_price = option_price_cache[cache_key]
                else:
                    if progress_bar:
                        progress_bar.set_postfix_str(f"Pricing {pos.option_type} ${pos.strike_price}")
                    current_option_price = option_price_historical(
                        ticker, pos.expiration_date, pos.strike_price,
                        pos.option_type, current_date.strftime('%Y-%m-%d')
                    )
                    option_price_cache[cache_key] = current_option_price

                # Calculate P&L
                pnl_percent = ((current_option_price - pos.entry_price) / pos.entry_price) * 100

                # Check stop-loss
                if current_option_price <= pos.entry_price * (1 - stoploss_percent / 100):
                    if verbose:
                        print(f"[{current_date.strftime('%Y-%m-%d')}] STOP-LOSS: {pos.option_type.upper()} "
                              f"${pos.strike_price} exp {pos.expiration_date}")
                        print(f"  Entry: ${pos.entry_price:.2f} → Exit: ${current_option_price:.2f} "
                              f"({pnl_percent:.1f}%)")

                    pos.exit_date = current_date
                    pos.exit_index = signal.index
                    pos.exit_price = current_option_price
                    pos.exit_stock_price = current_price
                    pos.exit_reason = 'stoploss'
                    pos.pnl_percent = pnl_percent
                    pos.pnl_dollar = (current_option_price - pos.entry_price) * 100 * contracts_per_trade
                    pos.days_held = (current_date - pos.entry_date).days

                    # Get Greeks at exit (with caching)
                    greeks_key = (pos.expiration_date, pos.strike_price, pos.option_type, current_date.strftime('%Y-%m-%d'))
                    try:
                        if greeks_key in greeks_cache:
                            pos.greeks_at_exit = greeks_cache[greeks_key]
                        else:
                            pos.greeks_at_exit = greeks_historical(
                                ticker, pos.expiration_date, pos.strike_price,
                                pos.option_type, current_date.strftime('%Y-%m-%d')
                            )
                            greeks_cache[greeks_key] = pos.greeks_at_exit
                    except:
                        pos.greeks_at_exit = None

                    positions_to_close.append(pos)
                    capital += current_option_price * 100 * contracts_per_trade

                # Check take-profit
                elif current_option_price >= pos.entry_price * (1 + takeprofit_percent / 100):
                    if verbose:
                        print(f"[{current_date.strftime('%Y-%m-%d')}] TAKE-PROFIT: {pos.option_type.upper()} "
                              f"${pos.strike_price} exp {pos.expiration_date}")
                        print(f"  Entry: ${pos.entry_price:.2f} → Exit: ${current_option_price:.2f} "
                              f"({pnl_percent:.1f}%)")

                    pos.exit_date = current_date
                    pos.exit_index = signal.index
                    pos.exit_price = current_option_price
                    pos.exit_stock_price = current_price
                    pos.exit_reason = 'takeprofit'
                    pos.pnl_percent = pnl_percent
                    pos.pnl_dollar = (current_option_price - pos.entry_price) * 100 * contracts_per_trade
                    pos.days_held = (current_date - pos.entry_date).days

                    # Get Greeks at exit (with caching)
                    greeks_key = (pos.expiration_date, pos.strike_price, pos.option_type, current_date.strftime('%Y-%m-%d'))
                    try:
                        if greeks_key in greeks_cache:
                            pos.greeks_at_exit = greeks_cache[greeks_key]
                        else:
                            pos.greeks_at_exit = greeks_historical(
                                ticker, pos.expiration_date, pos.strike_price,
                                pos.option_type, current_date.strftime('%Y-%m-%d')
                            )
                            greeks_cache[greeks_key] = pos.greeks_at_exit
                    except:
                        pos.greeks_at_exit = None

                    positions_to_close.append(pos)
                    capital += current_option_price * 100 * contracts_per_trade

                # Check if signal reverses (buy signal closes puts, sell signal closes calls)
                elif (signal.signal_type == 'buy' and pos.option_type == 'put') or \
                     (signal.signal_type == 'sell' and pos.option_type == 'call'):
                    if verbose:
                        print(f"[{current_date.strftime('%Y-%m-%d')}] SIGNAL EXIT: {pos.option_type.upper()} "
                              f"${pos.strike_price} exp {pos.expiration_date}")
                        print(f"  Entry: ${pos.entry_price:.2f} → Exit: ${current_option_price:.2f} "
                              f"({pnl_percent:.1f}%)")

                    pos.exit_date = current_date
                    pos.exit_index = signal.index
                    pos.exit_price = current_option_price
                    pos.exit_stock_price = current_price
                    pos.exit_reason = 'signal'
                    pos.pnl_percent = pnl_percent
                    pos.pnl_dollar = (current_option_price - pos.entry_price) * 100 * contracts_per_trade
                    pos.days_held = (current_date - pos.entry_date).days

                    # Get Greeks at exit (with caching)
                    greeks_key = (pos.expiration_date, pos.strike_price, pos.option_type, current_date.strftime('%Y-%m-%d'))
                    try:
                        if greeks_key in greeks_cache:
                            pos.greeks_at_exit = greeks_cache[greeks_key]
                        else:
                            pos.greeks_at_exit = greeks_historical(
                                ticker, pos.expiration_date, pos.strike_price,
                                pos.option_type, current_date.strftime('%Y-%m-%d')
                            )
                            greeks_cache[greeks_key] = pos.greeks_at_exit
                    except:
                        pos.greeks_at_exit = None

                    positions_to_close.append(pos)
                    capital += current_option_price * 100 * contracts_per_trade

            except Exception as e:
                if progress_bar:
                    progress_bar.write(f"  Warning: Could not price option on {current_date.strftime('%Y-%m-%d')}: {str(e)[:100]}")
                continue

        # Move closed positions to closed_trades
        for pos in positions_to_close:
            open_positions.remove(pos)
            closed_trades.append(pos)

        # Open new position if signal warrants it and we have capacity
        if len(open_positions) < max_positions:
            # Determine option type based on signal
            if signal.signal_type == 'buy':
                option_type = 'call'
            elif signal.signal_type == 'sell':
                option_type = 'put'
            else:
                continue

            # Calculate strike price and expiration
            strike_price = get_option_strike_price(current_price, option_type, otm_percent)
            expiration_date = get_option_expiration(current_date, days_to_expiry, ticker)

            # Get option price (with caching)
            try:
                cache_key = (expiration_date, strike_price, option_type, current_date.strftime('%Y-%m-%d'))
                if cache_key in option_price_cache:
                    entry_option_price = option_price_cache[cache_key]
                else:
                    if progress_bar:
                        progress_bar.set_postfix_str(f"Opening {option_type} ${strike_price}")
                    entry_option_price = option_price_historical(
                        ticker, expiration_date, strike_price,
                        option_type, current_date.strftime('%Y-%m-%d')
                    )
                    option_price_cache[cache_key] = entry_option_price

                # Calculate cost
                cost = entry_option_price * 100 * contracts_per_trade

                # Check if we have enough capital
                if capital >= cost:
                    # Get Greeks at entry (with caching)
                    greeks_key = cache_key
                    try:
                        if greeks_key in greeks_cache:
                            entry_greeks = greeks_cache[greeks_key]
                        else:
                            entry_greeks = greeks_historical(
                                ticker, expiration_date, strike_price,
                                option_type, current_date.strftime('%Y-%m-%d')
                            )
                            greeks_cache[greeks_key] = entry_greeks
                    except:
                        entry_greeks = None

                    # Create new position
                    new_trade = OptionTrade(
                        entry_date=current_date,
                        entry_index=signal.index,
                        option_type=option_type,
                        strike_price=strike_price,
                        expiration_date=expiration_date,
                        entry_price=entry_option_price,
                        entry_stock_price=current_price,
                        signal_reason=signal.reason,
                        greeks_at_entry=entry_greeks
                    )

                    open_positions.append(new_trade)
                    capital -= cost

                    if verbose:
                        print(f"\n[{current_date.strftime('%Y-%m-%d')}] OPENED: {option_type.upper()} "
                              f"${strike_price} exp {expiration_date}")
                        print(f"  Stock: ${current_price:.2f} | Entry: ${entry_option_price:.2f} "
                              f"| Cost: ${cost:.2f}")
                        print(f"  Signal: {signal.reason} | Capital: ${capital:,.2f}")
                        if entry_greeks:
                            print(f"  Greeks - Delta: {entry_greeks['delta']:.3f}, "
                                  f"Gamma: {entry_greeks['gamma']:.4f}, "
                                  f"Theta: {entry_greeks['theta']:.3f}, "
                                  f"Vega: {entry_greeks['vega']:.3f}")
                else:
                    if verbose:
                        print(f"[{current_date.strftime('%Y-%m-%d')}] SKIPPED: Insufficient capital "
                              f"(need ${cost:.2f}, have ${capital:.2f})")

            except Exception as e:
                if progress_bar:
                    progress_bar.write(f"[{current_date.strftime('%Y-%m-%d')}] ERROR opening position: {str(e)[:100]}")
                continue

        # Record capital history
        capital_history.append({
            'date': current_date,
            'capital': capital,
            'open_positions': len(open_positions)
        })

    # Close progress bar
    if progress_bar:
        progress_bar.close()

        # Print cache statistics
        print(f"\nBacktest Cache Statistics:")
        print(f"  Option price cache: {len(option_price_cache)} unique calculations")
        print(f"  Greeks cache: {len(greeks_cache)} unique calculations")

        # Print function-level cache stats
        try:
            from functions import get_stock_price_historical
            from technical_retrievals import get_rates, get_historical_volatility

            stock_cache = get_stock_price_historical.cache_info()
            rate_cache = get_rates.cache_info()
            vol_cache = get_historical_volatility.cache_info()

            print(f"\nAPI Call Cache Statistics:")
            print(f"  Stock prices: {stock_cache.hits} hits, {stock_cache.misses} misses ({stock_cache.hits/(stock_cache.hits+stock_cache.misses)*100:.1f}% hit rate)")
            print(f"  Interest rates: {rate_cache.hits} hits, {rate_cache.misses} misses ({rate_cache.hits/(rate_cache.hits+rate_cache.misses)*100:.1f}% hit rate)")
            print(f"  Volatility: {vol_cache.hits} hits, {vol_cache.misses} misses ({vol_cache.hits/(vol_cache.hits+vol_cache.misses)*100:.1f}% hit rate)")
        except:
            pass

    # Close any remaining open positions at expiration
    for pos in open_positions:
        try:
            exit_option_price = option_price_historical(
                ticker, pos.expiration_date, pos.strike_price,
                pos.option_type, pos.expiration_date
            )

            pnl_percent = ((exit_option_price - pos.entry_price) / pos.entry_price) * 100

            pos.exit_date = datetime.strptime(pos.expiration_date, '%Y-%m-%d')
            pos.exit_price = exit_option_price
            pos.exit_reason = 'expiration'
            pos.pnl_percent = pnl_percent
            pos.pnl_dollar = (exit_option_price - pos.entry_price) * 100 * contracts_per_trade
            pos.days_held = (pos.exit_date - pos.entry_date).days

            try:
                pos.greeks_at_exit = greeks_historical(
                    ticker, pos.expiration_date, pos.strike_price,
                    pos.option_type, pos.expiration_date
                )
            except:
                pos.greeks_at_exit = None

            capital += exit_option_price * 100 * contracts_per_trade
            closed_trades.append(pos)

            if verbose:
                print(f"\n[{pos.expiration_date}] EXPIRED: {pos.option_type.upper()} "
                      f"${pos.strike_price}")
                print(f"  Entry: ${pos.entry_price:.2f} → Exit: ${exit_option_price:.2f} "
                      f"({pnl_percent:.1f}%)")

        except Exception as e:
            if verbose:
                print(f"Warning: Could not close position at expiration: {e}")
            closed_trades.append(pos)

    # Calculate statistics
    total_trades = len(closed_trades)
    winning_trades = len([t for t in closed_trades if t.pnl_dollar and t.pnl_dollar > 0])
    losing_trades = len([t for t in closed_trades if t.pnl_dollar and t.pnl_dollar <= 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Calculate averages (handle empty lists)
    pnl_values = [t.pnl_percent for t in closed_trades if t.pnl_percent is not None]
    days_values = [t.days_held for t in closed_trades if t.days_held is not None]

    avg_pnl_percent = np.mean(pnl_values) if pnl_values else 0.0
    avg_days_held = np.mean(days_values) if days_values else 0.0

    total_return = ((capital - initial_capital) / initial_capital) * 100

    # Create results dictionary
    # Convert dates to strings (handle numpy.datetime64)
    import pandas as pd
    start_date_str = pd.Timestamp(analysis.dates[0]).strftime('%Y-%m-%d')
    end_date_str = pd.Timestamp(analysis.dates[-1]).strftime('%Y-%m-%d')

    # Convert trades to dictionaries for easier analysis
    trades_list = []
    for trade in closed_trades:
        trades_list.append({
            'entry_date': trade.entry_date,
            'entry_index': trade.entry_index,
            'option_type': trade.option_type,
            'strike_price': trade.strike_price,
            'expiration_date': trade.expiration_date,
            'entry_price': trade.entry_price,
            'entry_stock_price': trade.entry_stock_price,
            'signal_reason': trade.signal_reason,
            'greeks_at_entry': trade.greeks_at_entry,
            'exit_date': trade.exit_date,
            'exit_index': trade.exit_index,
            'exit_price': trade.exit_price,
            'exit_stock_price': trade.exit_stock_price,
            'exit_reason': trade.exit_reason,
            'pnl_percent': trade.pnl_percent,
            'pnl_dollar': trade.pnl_dollar,
            'days_held': trade.days_held,
            'greeks_at_exit': trade.greeks_at_exit
        })

    results = {
        'ticker': ticker,
        'date_range': (start_date_str, end_date_str),
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'average_pnl_percent': avg_pnl_percent,
        'average_days_held': avg_days_held,
        'trades': trades_list,
        'capital_history': capital_history,
        'parameters': {
            'contracts_per_trade': contracts_per_trade,
            'stoploss_percent': stoploss_percent,
            'takeprofit_percent': takeprofit_percent,
            'days_to_expiry': days_to_expiry,
            'otm_percent': otm_percent,
            'max_positions': max_positions
        },
        'fourier_analysis': analysis,
        'signals': signals
    }

    return results


def print_options_backtest_summary(results: dict):
    """Print a formatted summary of options backtest results (from dictionary)."""
    print(f"BACKTEST SUMMARY: {results['ticker']}")
    print(f"{'='*80}")
    print(f"Date Range: {results['date_range'][0]} to {results['date_range'][1]}")
    print(f"\nCapital:")
    print(f"  Initial: ${results['initial_capital']:,.2f}")
    print(f"  Final: ${results['final_capital']:,.2f}")
    print(f"  Total Return: {results['total_return']:.2f}%")
    print(f"\nTrades:")
    print(f"  Total: {results['total_trades']}")
    print(f"  Winners: {results['winning_trades']} ({results['win_rate']:.1f}%)")
    print(f"  Losers: {results['losing_trades']}")
    print(f"  Avg P&L: {results['average_pnl_percent']:.2f}%")
    print(f"  Avg Days Held: {results['average_days_held']:.1f}")
    print(f"\nParameters:")
    for key, value in results['parameters'].items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")

    # Show sample trades
    if results['trades']:
        print("Sample Trades (last 5):")
        print(f"{'Date':<12} {'Type':<6} {'Strike':<8} {'Entry':<8} {'Exit':<8} {'P&L %':<10} {'Days':<6} {'Reason':<12}")
        print("-" * 80)
        for trade in results['trades'][-5:]:
            entry_date_str = trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date'])
            print(f"{entry_date_str:<12} "
                  f"{trade['option_type'].upper():<6} "
                  f"${trade['strike_price']:<7.0f} "
                  f"${trade['entry_price']:<7.2f} "
                  f"${trade['exit_price'] or 0:<7.2f} "
                  f"{trade['pnl_percent'] or 0:<9.1f}% "
                  f"{trade['days_held'] or 0:<6} "
                  f"{trade['exit_reason'] or 'open':<12}")
        print()


def run_fourier_options_backtest(ticker: str,
                                 start_date: str,
                                 end_date: str,
                                 n_harmonics: int = 10,
                                 smoothing_sigma: float = 2.0,
                                 overbought_threshold: float = 5.0,
                                 oversold_threshold: float = -5.0,
                                 initial_capital: float = 10000.0,
                                 contracts_per_trade: int = 1,
                                 stoploss_percent: float = 50.0,
                                 takeprofit_percent: float = 50.0,
                                 days_to_expiry: int = 30,
                                 otm_percent: float = 2.0,
                                 max_positions: int = 1,
                                 tick_size: str = '1d',
                                 verbose: bool = True) -> dict:
    """
    Run a complete Fourier-based options backtest.

    This is the main entry point for Fourier options backtesting.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date, end_date : str
        Date range in 'YYYY-MM-DD' format
    n_harmonics : int
        Number of Fourier harmonics (default: 10)
    smoothing_sigma : float
        Smoothing parameter (default: 2.0)
    overbought_threshold, oversold_threshold : float
        Trading thresholds for signal detection
    initial_capital : float
        Starting capital (default: $10,000)
    contracts_per_trade : int
        Number of option contracts per trade (default: 1)
    stoploss_percent : float
        Stop-loss percentage (default: 50%)
    takeprofit_percent : float
        Take-profit percentage (default: 50%)
    days_to_expiry : int
        Days to option expiration (default: 30)
    otm_percent : float
        Out-of-the-money percentage (default: 2%)
    max_positions : int
        Maximum open positions (default: 1)
    tick_size : str
        Data interval (default: '1d')
    verbose : bool
        Print progress and trade details during execution (default: True)
        Does NOT print final summary - use print_options_backtest_summary() for that

    Returns:
    --------
    dict
        Dictionary containing backtesting results with keys:
        - ticker: Stock ticker
        - date_range: Tuple of (start_date, end_date)
        - initial_capital, final_capital, total_return
        - total_trades, winning_trades, losing_trades, win_rate
        - average_pnl_percent, average_days_held
        - trades: List of trade dictionaries
        - capital_history: List of capital snapshots
        - parameters: Dict of backtest parameters
        - fourier_analysis: FourierAnalysis object
        - signals: List of SignalPoint objects

    Example:
    --------
    >>> results = run_fourier_options_backtest(
    ...     ticker='AAPL',
    ...     start_date='2024-01-01',
    ...     end_date='2024-12-20',
    ...     n_harmonics=10,
    ...     stoploss_percent=50,
    ...     takeprofit_percent=50,
    ...     verbose=False  # Suppress progress output
    ... )
    >>> print(f"Total Return: {results['total_return']:.2f}%")
    >>> print(f"Win Rate: {results['win_rate']:.1f}%")
    >>> # Print formatted summary if desired
    >>> print_options_backtest_summary(results)
    """
    # Download data
    stock_data = get_stock_data(ticker, start_date, end_date, tick_size)

    if stock_data.empty:
        raise ValueError(f"No data found for {ticker}")

    # Extract prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        prices = stock_data['Close'][ticker].values
    else:
        prices = stock_data['Close'].values

    dates = stock_data.index

    # Perform Fourier analysis
    analysis = analyze_fourier(prices, dates, n_harmonics, smoothing_sigma)

    # Detect signals
    signals = detect_overbought_oversold(analysis, overbought_threshold, oversold_threshold)

    if verbose:
        print(f"\nDetected {len(signals)} signals")
        print(f"Buy signals: {len([s for s in signals if s.signal_type == 'buy'])}")
        print(f"Sell signals: {len([s for s in signals if s.signal_type == 'sell'])}")

        if len(signals) == 0:
            print("\n⚠️  WARNING: No signals detected! Check your thresholds.")
            print(f"   Overbought threshold: {overbought_threshold}")
            print(f"   Oversold threshold: {oversold_threshold}")
            print(f"   Try adjusting these values if you expect signals.")

    # Run options backtest
    results = backtest_options_signals(
        ticker=ticker,
        signals=signals,
        analysis=analysis,
        initial_capital=initial_capital,
        contracts_per_trade=contracts_per_trade,
        stoploss_percent=stoploss_percent,
        takeprofit_percent=takeprofit_percent,
        days_to_expiry=days_to_expiry,
        otm_percent=otm_percent,
        max_positions=max_positions,
        verbose=verbose
    )

    return results


# ========================================
# ALPACA LIVE TRADING INTEGRATION GUIDE
# ========================================

"""
GUIDE: Taking Fourier Options Trading Live with Alpaca

This guide explains how to integrate this backtesting system with Alpaca for live options trading.

PREREQUISITES:
--------------
1. Alpaca Account with options trading enabled
2. API keys (Paper or Live trading)
3. Install required packages:
   pip install alpaca-trade-api

4. Ensure you have approved options trading level (typically Level 2+ for long calls/puts)


STEP 1: SETUP ALPACA CONNECTION
--------------------------------

from alpaca_trade_api import REST
import os

# Store your keys in environment variables (NEVER commit these to git!)
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading first!
# For live trading: 'https://api.alpaca.markets'

# Initialize Alpaca API
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Verify connection
account = api.get_account()
print(f"Account Status: {account.status}")
print(f"Buying Power: ${float(account.buying_power):,.2f}")


STEP 2: CONVERT BACKTEST SIGNALS TO ALPACA ORDERS
--------------------------------------------------

def get_alpaca_option_symbol(ticker, expiration_date, strike_price, option_type):
    '''
    Alpaca uses OCC (Options Clearing Corporation) format for option symbols:
    Format: TICKER + YYMMDD + C/P + STRIKE (8 digits, padded)

    Example: AAPL call, $150 strike, exp 2025-01-17
    Symbol: AAPL250117C00150000

    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        expiration_date: Date string 'YYYY-MM-DD'
        strike_price: Strike price (e.g., 150.0)
        option_type: 'call' or 'put'

    Returns:
        OCC formatted option symbol
    '''
    from datetime import datetime

    # Parse expiration date
    exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
    exp_str = exp_dt.strftime('%y%m%d')  # YYMMDD format

    # Option type (C or P)
    opt_char = 'C' if option_type == 'call' else 'P'

    # Format strike price (8 digits, multiply by 1000 for decimals)
    strike_int = int(strike_price * 1000)
    strike_str = f"{strike_int:08d}"

    # Build OCC symbol
    occ_symbol = f"{ticker.upper()}{exp_str}{opt_char}{strike_str}"

    return occ_symbol


def place_option_order(api, ticker, expiration_date, strike_price, option_type,
                       qty, side='buy', order_type='limit', limit_price=None):
    '''
    Place an option order with Alpaca.

    Args:
        api: Alpaca REST API object
        ticker: Stock ticker
        expiration_date: Option expiration date 'YYYY-MM-DD'
        strike_price: Strike price
        option_type: 'call' or 'put'
        qty: Number of contracts (1 contract = 100 shares)
        side: 'buy' or 'sell' (default: 'buy' for opening long positions)
        order_type: 'market' or 'limit' (default: 'limit')
        limit_price: Limit price if order_type='limit' (price per share, NOT per contract)

    Returns:
        Order object from Alpaca
    '''
    # Get OCC symbol
    symbol = get_alpaca_option_symbol(ticker, expiration_date, strike_price, option_type)

    # Build order parameters
    order_params = {
        'symbol': symbol,
        'qty': qty,
        'side': side,
        'type': order_type,
        'time_in_force': 'day',  # Day order (expires at market close)
    }

    # Add limit price if specified
    if order_type == 'limit' and limit_price is not None:
        order_params['limit_price'] = limit_price

    # Submit order
    try:
        order = api.submit_order(**order_params)
        print(f"Order submitted: {order.id}")
        print(f"  Symbol: {symbol}")
        print(f"  Side: {side}")
        print(f"  Qty: {qty}")
        print(f"  Type: {order_type}")
        if limit_price:
            print(f"  Limit Price: ${limit_price:.2f}")
        return order
    except Exception as e:
        print(f"Error submitting order: {e}")
        return None


STEP 3: IMPLEMENT LIVE SIGNAL MONITORING
-----------------------------------------

def monitor_fourier_signals_live(api, ticker, n_harmonics=10, smoothing_sigma=2.0,
                                 overbought_threshold=5.0, oversold_threshold=-5.0,
                                 days_to_expiry=30, otm_percent=2.0,
                                 contracts_per_trade=1, check_interval_minutes=60):
    '''
    Live monitoring loop that checks for Fourier signals and places orders.

    This is a SIMPLIFIED example. Production code should include:
    - Error handling and recovery
    - Position tracking database
    - Stop-loss/take-profit monitoring
    - Market hours checking
    - Logging and alerting
    - Risk management controls

    Args:
        api: Alpaca REST API object
        ticker: Stock ticker to monitor
        n_harmonics: Fourier harmonics parameter
        smoothing_sigma: Smoothing parameter
        overbought_threshold: Sell signal threshold
        oversold_threshold: Buy signal threshold
        days_to_expiry: Days to option expiration
        otm_percent: Out-of-the-money percentage
        contracts_per_trade: Number of contracts per trade
        check_interval_minutes: Minutes between signal checks
    '''
    import time
    from datetime import datetime, timedelta

    print(f"Starting live monitoring for {ticker}...")
    print(f"Check interval: {check_interval_minutes} minutes")

    # Track open positions (in production, use database)
    open_positions = {}

    while True:
        try:
            # Check if market is open
            clock = api.get_clock()
            if not clock.is_open:
                print(f"Market closed. Next open: {clock.next_open}")
                time.sleep(60 * 5)  # Check every 5 minutes
                continue

            # Get recent stock data (use last 90 days for Fourier analysis)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

            stock_data = get_stock_data(ticker, start_date, end_date, '1d')

            if isinstance(stock_data.columns, pd.MultiIndex):
                prices = stock_data['Close'][ticker].values
            else:
                prices = stock_data['Close'].values

            dates = stock_data.index

            # Perform Fourier analysis
            analysis = analyze_fourier(prices, dates, n_harmonics, smoothing_sigma)

            # Detect signals (focus on most recent signal)
            signals = detect_overbought_oversold(analysis, overbought_threshold, oversold_threshold)

            if signals:
                latest_signal = signals[-1]  # Most recent signal

                print(f"\n[{datetime.now()}] Latest Signal: {latest_signal.signal_type.upper()}")
                print(f"  Stock Price: ${latest_signal.price:.2f}")
                print(f"  Reason: {latest_signal.reason}")

                # Check if we should act on this signal
                # (Only trade if signal is from today)
                if latest_signal.date.date() == datetime.now().date():

                    # Determine option type
                    if latest_signal.signal_type == 'buy':
                        option_type = 'call'
                    elif latest_signal.signal_type == 'sell':
                        option_type = 'put'
                    else:
                        continue

                    # Calculate strike and expiration
                    current_price = latest_signal.price
                    strike_price = get_option_strike_price(current_price, option_type, otm_percent)
                    expiration_date = get_option_expiration(datetime.now(), days_to_expiry, ticker)

                    # Get current option quote
                    symbol = get_alpaca_option_symbol(ticker, expiration_date, strike_price, option_type)

                    try:
                        quote = api.get_latest_trade(symbol)
                        current_option_price = quote.price

                        print(f"  Option: {option_type.upper()} ${strike_price} exp {expiration_date}")
                        print(f"  Option Price: ${current_option_price:.2f}")

                        # Check buying power
                        account = api.get_account()
                        buying_power = float(account.buying_power)
                        required_capital = current_option_price * 100 * contracts_per_trade

                        if buying_power >= required_capital:
                            # Place order with limit slightly above current price
                            limit_price = current_option_price * 1.02  # 2% slippage allowance

                            order = place_option_order(
                                api, ticker, expiration_date, strike_price, option_type,
                                qty=contracts_per_trade, side='buy', order_type='limit',
                                limit_price=limit_price
                            )

                            if order:
                                # Track position
                                open_positions[order.id] = {
                                    'symbol': symbol,
                                    'entry_price': current_option_price,
                                    'entry_date': datetime.now(),
                                    'strike': strike_price,
                                    'expiration': expiration_date,
                                    'type': option_type,
                                    'qty': contracts_per_trade
                                }
                        else:
                            print(f"  Insufficient buying power (need ${required_capital:.2f}, have ${buying_power:.2f})")

                    except Exception as e:
                        print(f"  Error getting option quote: {e}")

            # Monitor existing positions for stop-loss/take-profit
            # (Production code would check each position and close if needed)
            for order_id, pos in list(open_positions.items()):
                try:
                    # Get current option price
                    quote = api.get_latest_trade(pos['symbol'])
                    current_price = quote.price

                    # Calculate P&L
                    pnl_percent = ((current_price - pos['entry_price']) / pos['entry_price']) * 100

                    # Check stop-loss (50% default)
                    if pnl_percent <= -50:
                        print(f"\nSTOP-LOSS: Closing {pos['symbol']} at {pnl_percent:.1f}% loss")
                        close_order = place_option_order(
                            api, ticker, pos['expiration'], pos['strike'], pos['type'],
                            qty=pos['qty'], side='sell', order_type='market'
                        )
                        if close_order:
                            del open_positions[order_id]

                    # Check take-profit (50% default)
                    elif pnl_percent >= 50:
                        print(f"\nTAKE-PROFIT: Closing {pos['symbol']} at {pnl_percent:.1f}% gain")
                        close_order = place_option_order(
                            api, ticker, pos['expiration'], pos['strike'], pos['type'],
                            qty=pos['qty'], side='sell', order_type='market'
                        )
                        if close_order:
                            del open_positions[order_id]

                except Exception as e:
                    print(f"Error monitoring position {pos['symbol']}: {e}")

            # Wait before next check
            print(f"\nNext check in {check_interval_minutes} minutes...")
            time.sleep(check_interval_minutes * 60)

        except KeyboardInterrupt:
            print("\nStopping monitor...")
            break
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying


STEP 4: PRODUCTION DEPLOYMENT CHECKLIST
----------------------------------------

Before going live with real money:

1. PAPER TRADING
   - Test with Alpaca paper trading for at least 2 weeks
   - Verify all order placements work correctly
   - Monitor for unexpected behavior

2. RISK MANAGEMENT
   - Set maximum position size limits
   - Implement portfolio-level stop-loss
   - Set maximum daily loss limits
   - Limit total capital allocation (e.g., max 20% of portfolio)

3. MONITORING & ALERTS
   - Set up logging (use Python logging module)
   - Implement email/SMS alerts for trades
   - Monitor account health (margin, buying power)
   - Track all errors and exceptions

4. ERROR HANDLING
   - Handle API rate limits (Alpaca has 200 requests/minute limit)
   - Retry logic for failed orders
   - Fallback to market orders if limit orders don't fill
   - Handle split/dividend adjustments

5. DATA QUALITY
   - Validate stock data before trading
   - Check for data gaps or anomalies
   - Use multiple data sources if possible

6. COMPLIANCE
   - Pattern Day Trader (PDT) rule: Need $25k+ for >3 day trades in 5 days
   - Options level requirements
   - Tax implications of options trading
   - Keep detailed trade logs for tax reporting

7. INFRASTRUCTURE
   - Run on reliable server (not personal laptop)
   - Use cloud services (AWS, GCP, Azure) for uptime
   - Implement graceful shutdown handling
   - Database for position tracking (SQLite, PostgreSQL)

8. BACKTESTING VALIDATION
   - Run extensive backtests across multiple tickers
   - Test different market conditions (bull, bear, sideways)
   - Account for slippage and commissions
   - Validate against walk-forward periods

EXAMPLE PRODUCTION ENTRY POINT:
--------------------------------

if __name__ == "__main__":
    # Initialize Alpaca
    api = REST(
        os.environ['ALPACA_API_KEY'],
        os.environ['ALPACA_SECRET_KEY'],
        'https://paper-api.alpaca.markets'  # Start with paper!
    )

    # Start monitoring
    monitor_fourier_signals_live(
        api=api,
        ticker='AAPL',
        n_harmonics=10,
        smoothing_sigma=2.0,
        overbought_threshold=5.0,
        oversold_threshold=-5.0,
        days_to_expiry=30,
        otm_percent=2.0,
        contracts_per_trade=1,
        check_interval_minutes=60
    )

IMPORTANT DISCLAIMERS:
----------------------
1. Options trading involves significant risk and is not suitable for all investors
2. Past performance does not guarantee future results
3. This code is for educational purposes - test thoroughly before live trading
4. Start with paper trading and small positions
5. Consult a financial advisor before trading with real money
6. Monitor your positions regularly - automated systems can fail
7. Be aware of Alpaca's commission structure for options
8. Understand the Greeks and how they affect option prices

RESOURCES:
----------
- Alpaca Options Docs: https://alpaca.markets/docs/trading/options/
- Alpaca API Reference: https://alpaca.markets/docs/api-references/
- Options Education: https://www.optionseducation.org/
- Risk Management: https://www.investopedia.com/options-trading-risk-management-4689492
"""
