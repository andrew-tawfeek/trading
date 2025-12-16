import yfinance as yf
import numpy as np
import pandas as pd

def countdown(date):
    # days until date
    from datetime import datetime
    target_date = datetime.strptime(date, "%Y-%m-%d")
    today = datetime.now()
    count = target_date - today
    return count.days

def get_rates(expiration_date, historical_date=None):
    """
    Determine appropriate risk-free rate based on time to expiration.

    Args:
        expiration_date (str): Expiration date in 'YYYY-MM-DD' format
        historical_date (str): Optional historical date in 'YYYY-MM-DD' format.
                              If None, uses current date. If provided, fetches
                              historical Treasury rate for that date.

    Returns:
        float: Risk-free rate as decimal (e.g., 0.043 for 4.3%)
    """
    import yfinance as yf
    from datetime import datetime, timedelta

    # Calculate days to expiry from the reference date
    if historical_date:
        reference_date = datetime.strptime(historical_date, "%Y-%m-%d")
        expiry_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        days_to_expiry = (expiry_date - reference_date).days
    else:
        days_to_expiry = countdown(expiration_date)
    
    # Determine which Treasury rate to use based on time to expiration
    if days_to_expiry <= 90:  # 0-3 months
        ticker = "^IRX"  # 13-week T-bill
    elif days_to_expiry <= 180:  # 3-6 months
        ticker = "^IRX"  # 13-week T-bill (closest proxy)
    elif days_to_expiry <= 365:  # 6-12 months
        ticker = "^FVX"  # 5-year Treasury (scaled down)
    else:  # 1+ years
        ticker = "^TNX"  # 10-year Treasury
    
    try:
        # Fetch rate (current or historical)
        rate_ticker = yf.Ticker(ticker)

        if historical_date:
            # Fetch historical rate for specific date
            # Add a day buffer to ensure we get data
            hist_date = datetime.strptime(historical_date, "%Y-%m-%d")
            start_date = (hist_date - timedelta(days=5)).strftime("%Y-%m-%d")
            end_date = (hist_date + timedelta(days=1)).strftime("%Y-%m-%d")

            hist = rate_ticker.history(start=start_date, end=end_date)
            if len(hist) == 0:
                raise ValueError("No historical rate data available")

            # Get the closest date to our target
            rate = hist['Close'].iloc[-1] / 100
        else:
            # Fetch current rate
            rate = rate_ticker.history(period='1d')['Close'].iloc[-1] / 100

        # Sanity check: rates should be between 0% and 20%
        if rate <= 0 or rate > 0.20:
            raise ValueError("Rate outside reasonable bounds")

        return rate
        
    except Exception as e:
        # Fallback to reasonable estimates if data fetch fails
        print(f"Warning: Could not fetch rate for {ticker}. Using estimate. Error: {e}")
        
        if days_to_expiry <= 90:
            return 0.043  # ~4.3%
        elif days_to_expiry <= 180:
            return 0.042  # ~4.2%
        elif days_to_expiry <= 365:
            return 0.041  # ~4.1%
        else:
            return 0.040  # ~4.0%

def get_historical_volatility(ticker_symbol, date, lookback_days=30):
    """
    Calculate historical volatility using past returns.

    Args:
        ticker_symbol (str): Stock ticker symbol
        date (str): Reference date in 'YYYY-MM-DD' format
        lookback_days (int): Number of days to look back for volatility calculation

    Returns:
        float: Annualized historical volatility as decimal (e.g., 0.25 for 25%)
    """
    from datetime import datetime, timedelta

    # Parse the reference date
    end_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=lookback_days + 10)  # Extra days for weekends/holidays

    try:
        # Fetch historical stock prices
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(start=start_date.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"))

        if len(hist) < lookback_days * 0.7:  # Need at least 70% of requested days
            raise ValueError(f"Insufficient data: only {len(hist)} days available")

        # Calculate log returns
        prices = hist['Close']
        returns = np.log(prices / prices.shift(1))

        # Calculate annualized volatility (assuming 252 trading days per year)
        volatility = returns.std() * np.sqrt(252)

        return float(volatility)

    except Exception as e:
        print(f"Warning: Could not calculate historical volatility for {ticker_symbol}. Error: {e}")
        # Fallback to reasonable estimate based on typical stock volatility
        return 0.30  # 30% annual volatility as default


def get_historical_volatility_intraday(ticker_symbol, date, time, lookback_days=30):
    """
    Calculate historical volatility using intraday data for recent dates.

    Args:
        ticker_symbol (str): Stock ticker symbol
        date (str): Reference date in 'YYYY-MM-DD' format
        time (str): Reference time in 'HH:MM' format (24-hour, Eastern Time)
        lookback_days (int): Number of days to look back for volatility calculation

    Returns:
        float: Annualized historical volatility as decimal (e.g., 0.25 for 25%)

    Note: Only works for last 60 days. Falls back to daily volatility if intraday fails.
    """
    from datetime import datetime, timedelta

    # Parse the reference datetime
    end_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")

    # Check if within 60-day window
    days_ago = (datetime.now() - end_datetime).days
    if days_ago > 60:
        # Fall back to daily volatility for older dates
        return get_historical_volatility(ticker_symbol, date, lookback_days)

    try:
        # Fetch intraday data (1-hour intervals for better coverage)
        start_date = (end_datetime - timedelta(days=lookback_days + 5)).strftime("%Y-%m-%d")
        end_date = (end_datetime + timedelta(days=1)).strftime("%Y-%m-%d")

        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(start=start_date, end=end_date, interval='1h')

        if len(hist) < lookback_days * 3:  # Need at least ~3 hours per day
            raise ValueError(f"Insufficient intraday data: only {len(hist)} hours available")

        # Calculate log returns
        prices = hist['Close']
        returns = np.log(prices / prices.shift(1))

        # Calculate volatility
        # Annualize based on trading hours: ~6.5 hours/day * 252 days = 1638 trading hours/year
        volatility = returns.std() * np.sqrt(1638)

        return float(volatility)

    except Exception as e:
        print(f"Warning: Could not calculate intraday volatility for {ticker_symbol}. Using daily volatility. Error: {e}")
        # Fall back to daily volatility calculation
        return get_historical_volatility(ticker_symbol, date, lookback_days)

