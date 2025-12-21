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
            x=dates,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name=f'{ticker}',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=dates,
            y=close_prices,
            mode='lines',
            name=f'{ticker} Close Price',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
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
                x=dates,
                y=analysis.fourier_prices,
                mode='lines',
                name=f'Fourier {n_harm}h',
                line=dict(color=color, width=2, dash='dash'),
                hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Fourier {n_harm}h</b>: $%{{y:.2f}}<extra></extra>',
                visible=True,
                legendgroup=f'fourier_{n_harm}'
            ), row=1, col=1)

            # Add detrended Fourier curve to bottom plot (centered around 0)
            fig.add_trace(go.Scatter(
                x=dates,
                y=analysis.detrended_fourier,
                mode='lines',
                name=f'Fourier {n_harm}h (detrended)',
                line=dict(color=color, width=2),
                hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Deviation</b>: $%{{y:.2f}}<extra></extra>',
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
                    fig.add_trace(go.Scatter(
                        x=[s.date for s in buy_signals],
                        y=[s.price for s in buy_signals],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='green', size=12, symbol='triangle-up'),
                        hovertemplate='<b>BUY</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ), row=1, col=1)

                # Add sell markers
                if sell_signals:
                    fig.add_trace(go.Scatter(
                        x=[s.date for s in sell_signals],
                        y=[s.price for s in sell_signals],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='red', size=12, symbol='triangle-down'),
                        hovertemplate='<b>SELL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ), row=1, col=1)

                print(f"\nDetected {len(buy_signals)} buy signals and {len(sell_signals)} sell signals")

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
        # Update x-axes to remove market closed gaps
        fig.update_xaxes(title_text="Date", row=2, col=1, rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ])
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
            ],
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
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # Hide weekends
                ],
                type="date"
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
    # Example 1: Basic candlestick plot with Fourier
    print("\n=== Example 1: Candlestick with Fourier ===")
    fig1 = plot_stock(
        'SPY',
        '2025-05-28',
        '2025-12-20',
        tick_size='1d',
        fourier=True,
        n_harmonics=[5, 10, 18, 20, 30],
        smoothing_sigma=0.0,
        use_candlestick=True,
        overbought_threshold=9,
        oversold_threshold=-8,
        show_signals=True
    )
    fig1.show()

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
