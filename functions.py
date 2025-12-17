import yfinance as yf
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
from py_vollib.black_scholes import black_scholes
import numpy as np
from datetime import datetime, timedelta
from technical_retrievals import *
import pandas as pd
import os

def greeks(ticker_symbol, strike_date, strike_price, option_type, status = False):
    assert option_type in ['put', 'call'], "option_type must be 'put' or 'call'"

    days = countdown(strike_date)
    if days < 2:
        return "Error: Option too close to expiration for accurate Greeks calculation"
    
    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(strike_date)

    if option_type == 'put':
        option = chain.puts[chain.puts['strike'] == strike_price].iloc[0]
    elif option_type == 'call':
        option = chain.calls[chain.calls['strike'] == strike_price].iloc[0]

    S = ticker.history(period='1d')['Close'].iloc[-1]

    if option_type == 'put':
        K = option['strike']
        IV = option['impliedVolatility']
    elif option_type == 'call':
        K = option['strike']
        IV = option['impliedVolatility']

    t = countdown(strike_date)/365  # Calculate from expiration

    # Calculate Greeks
    d = delta(option_type[0], S, K, t, get_rates(strike_date), IV)
    g = gamma(option_type[0], S, K, t, get_rates(strike_date), IV)
    v = vega(option_type[0], S, K, t, get_rates(strike_date), IV)
    th = theta(option_type[0], S, K, t, get_rates(strike_date), IV)
    r = rho(option_type[0], S, K, t, get_rates(strike_date), IV)
    
    return {'delta': float(d), 'gamma': float(g), 'vega': float(v), 'theta': float(th), 'rho': float(r)}


def option_price(ticker_symbol, strike_date, strike_price, option_type):
    """
    Get the current price of an option.
    
    Args:
        ticker_symbol: Stock ticker (e.g., 'AAPL')
        strike_date: Expiration date in 'YYYY-MM-DD' format
        strike_price: Strike price (e.g., 150.0)
        option_type: 'call' or 'put'
    
    Returns:
        dict with bid, ask, lastPrice, and volume
    """
    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(strike_date)
    
    if option_type == 'put':
        option = chain.puts[chain.puts['strike'] == strike_price].iloc[0]
    elif option_type == 'call':
        option = chain.calls[chain.calls['strike'] == strike_price].iloc[0]
    
    return {
        'lastPrice': float(option['lastPrice']),
        'bid': float(option['bid']),
        'ask': float(option['ask']),
        'volume': int(option['volume'])
    }




def continuous_monitor(ticker):
    ticker = yf.Ticker(ticker) # this class has all data
    price = ticker.history(period='1d')['Close'].iloc[-1]

    print(f"Starting price: {price}")

    try:
        while True:
            new_price = ticker.history(period='1d')['Close'].iloc[-1]
            if price != new_price:
                price = new_price
                print(f"New price: {price}")
            #time.sleep(5)  # Wait 5 seconds between checks
    except KeyboardInterrupt:
        print("Stopped monitoring")


def get_stock_price_historical(ticker_symbol, date):
    """
    Get historical stock price for a specific date.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        date (str): Date in 'YYYY-MM-DD' format

    Returns:
        float: Close price for that date
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        target_date = datetime.strptime(date, "%Y-%m-%d")

        # Fetch data for a range around the target date to handle weekends/holidays
        start_date = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

        hist = ticker.history(start=start_date, end=end_date)

        if len(hist) == 0:
            raise ValueError(f"No historical data available for {ticker_symbol} around {date}")

        # Get the closest available date (should be on or before target date)
        price = hist['Close'].iloc[-1]

        return float(price)

    except Exception as e:
        raise ValueError(f"Error fetching historical stock price for {ticker_symbol} on {date}: {e}")


def get_stock_price_intraday(ticker_symbol, date, time):
    """
    Get intraday stock price for a specific date and time.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        date (str): Date in 'YYYY-MM-DD' format
        time (str): Time in 'HH:MM' format (24-hour, Eastern Time)
                   Must be during market hours (09:30-16:00 ET)

    Returns:
        float: Stock price at that timestamp

    Note: Only works for last 60 days due to yfinance limitations.
          Uses 1-hour interval data for best availability.
    """
    from datetime import datetime, timedelta
    import pytz

    # Validate market hours
    hour, minute = map(int, time.split(':'))
    if not ((hour == 9 and minute >= 30) or (10 <= hour < 16) or (hour == 16 and minute == 0)):
        raise ValueError("Time must be during market hours (09:30-16:00 ET)")

    try:
        ticker = yf.Ticker(ticker_symbol)
        target_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")

        # Check if date is within last 60 days
        days_ago = (datetime.now() - target_datetime).days
        if days_ago > 60:
            raise ValueError("Intraday data only available for last 60 days. Use get_stock_price_historical() for older dates.")

        # Fetch 1-hour interval data for the specific day
        # Add buffer days to ensure we get the data
        start_date = (target_datetime - timedelta(days=2)).strftime("%Y-%m-%d")
        end_date = (target_datetime + timedelta(days=1)).strftime("%Y-%m-%d")

        # Use 1h interval (most reliable for 60-day window)
        hist = ticker.history(start=start_date, end=end_date, interval='1h')

        if len(hist) == 0:
            raise ValueError(f"No intraday data available for {ticker_symbol} around {date} {time}")

        # Convert index to Eastern Time for comparison
        et_tz = pytz.timezone('US/Eastern')
        hist.index = hist.index.tz_convert(et_tz)

        # Create target datetime in ET
        target_et = et_tz.localize(target_datetime)

        # Find the closest timestamp (within 1 hour)
        time_diffs = abs(hist.index - target_et)
        closest_idx = time_diffs.argmin()

        # Verify the closest time is within 1 hour
        if time_diffs[closest_idx] > timedelta(hours=1):
            raise ValueError(f"No data found within 1 hour of {date} {time}")

        price = hist['Close'].iloc[closest_idx]
        return float(price)

    except Exception as e:
        raise ValueError(f"Error fetching intraday stock price for {ticker_symbol} at {date} {time}: {e}")


def option_price_historical(ticker_symbol, strike_date, strike_price, option_type,
                           historical_date, iv=None):
    """
    Calculate theoretical option price at a historical date using Black-Scholes-Merton model.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        strike_date (str): Option expiration date in 'YYYY-MM-DD' format
        strike_price (float): Strike price (e.g., 150.0)
        option_type (str): 'call' or 'put'
        historical_date (str): Date to price option at in 'YYYY-MM-DD' format
        iv (float, optional): Implied volatility as decimal (e.g., 0.25 for 25%).
                             If None, historical volatility will be calculated.

    Returns:
        float: Theoretical option price using BSM model
    """
    assert option_type in ['put', 'call'], "option_type must be 'put' or 'call'"

    # Get historical stock price
    S = get_stock_price_historical(ticker_symbol, historical_date)

    # Strike price
    K = strike_price

    # Calculate time to expiration from historical date
    hist_date = datetime.strptime(historical_date, "%Y-%m-%d")
    exp_date = datetime.strptime(strike_date, "%Y-%m-%d")
    days_to_expiry = (exp_date - hist_date).days

    if days_to_expiry <= 0:
        raise ValueError("Historical date must be before expiration date")

    t = days_to_expiry / 365  # Time to expiration in years

    # Get risk-free rate for historical date
    r = get_rates(strike_date, historical_date)

    # Get or calculate implied volatility
    if iv is None:
        sigma = get_historical_volatility(ticker_symbol, historical_date)
    else:
        sigma = iv

    # Calculate theoretical option price using Black-Scholes
    price = black_scholes(option_type[0], S, K, t, r, sigma)

    return float(price)


def option_price_intraday(ticker_symbol, strike_date, strike_price, option_type,
                         historical_date, time, iv=None):
    """
    Calculate theoretical option price at a specific intraday time using Black-Scholes-Merton model.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        strike_date (str): Option expiration date in 'YYYY-MM-DD' format
        strike_price (float): Strike price (e.g., 150.0)
        option_type (str): 'call' or 'put'
        historical_date (str): Date in 'YYYY-MM-DD' format (within last 60 days)
        time (str): Time in 'HH:MM' format (24-hour, Eastern Time, market hours only)
        iv (float, optional): Implied volatility as decimal (e.g., 0.25 for 25%).
                             If None, historical volatility will be calculated.

    Returns:
        float: Theoretical option price using BSM model

    Note: Only works for last 60 days due to yfinance intraday data limitations.
    """
    assert option_type in ['put', 'call'], "option_type must be 'put' or 'call'"

    # Get intraday stock price
    S = get_stock_price_intraday(ticker_symbol, historical_date, time)

    # Strike price
    K = strike_price

    # Calculate time to expiration from historical datetime
    from datetime import datetime
    hist_datetime = datetime.strptime(f"{historical_date} {time}", "%Y-%m-%d %H:%M")
    exp_datetime = datetime.strptime(f"{strike_date} 16:00", "%Y-%m-%d %H:%M")  # Options expire at 4 PM ET
    time_diff = exp_datetime - hist_datetime

    if time_diff.total_seconds() <= 0:
        raise ValueError("Historical datetime must be before expiration")

    # Convert to years (including fractional days)
    t = time_diff.total_seconds() / (365.25 * 24 * 3600)

    # Get risk-free rate for historical date
    r = get_rates(strike_date, historical_date)

    # Get or calculate implied volatility (uses daily volatility calculation)
    if iv is None:
        sigma = get_historical_volatility(ticker_symbol, historical_date)
    else:
        sigma = iv

    # Calculate theoretical option price using Black-Scholes
    price = black_scholes(option_type[0], S, K, t, r, sigma)

    return float(price)


def greeks_historical(ticker_symbol, strike_date, strike_price, option_type,
                     historical_date, iv=None):
    """
    Calculate option Greeks at a historical date using Black-Scholes-Merton model.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        strike_date (str): Option expiration date in 'YYYY-MM-DD' format
        strike_price (float): Strike price (e.g., 150.0)
        option_type (str): 'call' or 'put'
        historical_date (str): Date to calculate Greeks at in 'YYYY-MM-DD' format
        iv (float, optional): Implied volatility as decimal (e.g., 0.25 for 25%).
                             If None, historical volatility will be calculated.

    Returns:
        dict: Dictionary with delta, gamma, vega, theta, and rho values
    """
    assert option_type in ['put', 'call'], "option_type must be 'put' or 'call'"

    # Get historical stock price
    S = get_stock_price_historical(ticker_symbol, historical_date)

    # Strike price
    K = strike_price

    # Calculate time to expiration from historical date
    hist_date = datetime.strptime(historical_date, "%Y-%m-%d")
    exp_date = datetime.strptime(strike_date, "%Y-%m-%d")
    days_to_expiry = (exp_date - hist_date).days

    if days_to_expiry <= 0:
        raise ValueError("Historical date must be before expiration date")

    if days_to_expiry < 2:
        return "Error: Option too close to expiration for accurate Greeks calculation"

    t = days_to_expiry / 365  # Time to expiration in years

    # Get risk-free rate for historical date
    r = get_rates(strike_date, historical_date)

    # Get or calculate implied volatility
    if iv is None:
        sigma = get_historical_volatility(ticker_symbol, historical_date)
    else:
        sigma = iv

    # Calculate Greeks using Black-Scholes
    d = delta(option_type[0], S, K, t, r, sigma)
    g = gamma(option_type[0], S, K, t, r, sigma)
    v = vega(option_type[0], S, K, t, r, sigma)
    th = theta(option_type[0], S, K, t, r, sigma)
    rho_val = rho(option_type[0], S, K, t, r, sigma)

    return {
        'delta': float(d),
        'gamma': float(g),
        'vega': float(v),
        'theta': float(th),
        'rho': float(rho_val)
    }


def greeks_intraday(ticker_symbol, strike_date, strike_price, option_type,
                   historical_date, time, iv=None):
    """
    Calculate option Greeks at a specific intraday time using Black-Scholes-Merton model.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        strike_date (str): Option expiration date in 'YYYY-MM-DD' format
        strike_price (float): Strike price (e.g., 150.0)
        option_type (str): 'call' or 'put'
        historical_date (str): Date in 'YYYY-MM-DD' format (within last 60 days)
        time (str): Time in 'HH:MM' format (24-hour, Eastern Time, market hours only)
        iv (float, optional): Implied volatility as decimal (e.g., 0.25 for 25%).
                             If None, historical volatility will be calculated.

    Returns:
        dict: Dictionary with delta, gamma, vega, theta, and rho values

    Note: Only works for last 60 days due to yfinance intraday data limitations.
    """
    assert option_type in ['put', 'call'], "option_type must be 'put' or 'call'"

    # Get intraday stock price
    S = get_stock_price_intraday(ticker_symbol, historical_date, time)

    # Strike price
    K = strike_price

    # Calculate time to expiration from historical datetime
    from datetime import datetime
    hist_datetime = datetime.strptime(f"{historical_date} {time}", "%Y-%m-%d %H:%M")
    exp_datetime = datetime.strptime(f"{strike_date} 16:00", "%Y-%m-%d %H:%M")  # Options expire at 4 PM ET
    time_diff = exp_datetime - hist_datetime

    if time_diff.total_seconds() <= 0:
        raise ValueError("Historical datetime must be before expiration")

    # Convert to years (including fractional days)
    t = time_diff.total_seconds() / (365.25 * 24 * 3600)

    if t < (2 / 365.25):  # Less than 2 days
        return "Error: Option too close to expiration for accurate Greeks calculation"

    # Get risk-free rate for historical date
    r = get_rates(strike_date, historical_date)

    # Get or calculate implied volatility (uses daily volatility calculation)
    if iv is None:
        sigma = get_historical_volatility(ticker_symbol, historical_date)
    else:
        sigma = iv

    # Calculate Greeks using Black-Scholes
    d = delta(option_type[0], S, K, t, r, sigma)
    g = gamma(option_type[0], S, K, t, r, sigma)
    v = vega(option_type[0], S, K, t, r, sigma)
    th = theta(option_type[0], S, K, t, r, sigma)
    rho_val = rho(option_type[0], S, K, t, r, sigma)

    return {
        'delta': float(d),
        'gamma': float(g),
        'vega': float(v),
        'theta': float(th),
        'rho': float(rho_val)
    }



def options_purchase(ticker_symbol, strike_date, strike_price, date, time,
                    option_type, stoploss=20, takeprofit=50, iv=None):
    """
    Simulate buying an option and monitoring it until stop-loss, take-profit, or expiration.

    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        strike_date (str): Option expiration date in 'YYYY-MM-DD' format
        strike_price (float): Strike price (e.g., 150.0)
        date (str): Purchase date in 'YYYY-MM-DD' format (within last 60 days for intraday)
        time (str): Purchase time in 'HH:MM' format (24-hour, Eastern Time, market hours)
        option_type (str): 'call' or 'put'
        stoploss (float): Stop-loss percentage (default: 20%)
        takeprofit (float): Take-profit percentage (default: 50%)
        iv (float, optional): Implied volatility as decimal. If None, calculated from history.

    Returns:
        dict: Contains:
            - entry_price: Initial option price
            - exit_price: Final option price when limit triggered
            - exit_time: Time when position was closed
            - exit_reason: 'stoploss', 'takeprofit', 'expiration', or 'position_open'
            - pnl_percent: Profit/loss percentage
            - pnl_dollar: Profit/loss in dollars (per contract)
            - days_held: Number of days position was held

    Note: Monitors across multiple days until stop-loss, take-profit, expiration, or current date.
          Uses intraday (hourly) monitoring for dates within last 60 days,
          otherwise uses end-of-day prices. If monitoring reaches current date with position
          still open, returns with exit_reason='position_open'.
    """
    from datetime import datetime, timedelta
    import pytz

    assert option_type in ['put', 'call'], "option_type must be 'put' or 'call'"

    # Get entry price
    entry_price = option_price_intraday(ticker_symbol, strike_date, strike_price,
                                       option_type, date, time, iv)

    # Calculate stop-loss and take-profit thresholds
    stoploss_price = entry_price * (1 - stoploss / 100)
    takeprofit_price = entry_price * (1 + takeprofit / 100)

    print(f"Entry: ${entry_price:.2f}")
    print(f"Stop-loss: ${stoploss_price:.2f} (-{stoploss}%)")
    print(f"Take-profit: ${takeprofit_price:.2f} (+{takeprofit}%)")

    # Parse dates
    entry_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    expiration_date = datetime.strptime(strike_date, "%Y-%m-%d")

    # Get current date (don't fetch data beyond this)
    today = datetime.now().date()

    # Check if we can use intraday data (within 60 days)
    days_ago = (datetime.now() - entry_datetime).days
    use_intraday = days_ago <= 60

    # Start monitoring from entry date
    current_date = datetime.strptime(date, "%Y-%m-%d")

    print(f"\nMonitoring position from {date} until expiration ({strike_date})...")
    print("-" * 60)

    # Monitor each day until expiration or current date
    while current_date <= expiration_date and current_date.date() <= today:
        current_date_str = current_date.strftime("%Y-%m-%d")

        # Skip weekends (yfinance won't have data)
        if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            current_date += timedelta(days=1)
            continue

        print(f"\n{current_date_str}:")

        # Determine if this is entry day or subsequent day
        is_entry_day = (current_date_str == date)

        # Check if current date is within intraday monitoring window
        check_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        days_from_now = (datetime.now() - check_date).days
        can_use_intraday = days_from_now <= 60

        if can_use_intraday and use_intraday:
            # Intraday monitoring (hourly)
            if is_entry_day:
                # Start from entry time + 1 hour
                start_hour = entry_datetime.hour + 1
            else:
                # Start from market open (9:30 AM, but use 10:00 for hourly data)
                start_hour = 10

            # Monitor hourly until market close (4 PM)
            for hour in range(start_hour, 17):  # 10 AM to 4 PM
                check_time = f"{hour:02d}:00"

                try:
                    current_price = option_price_intraday(ticker_symbol, strike_date, strike_price,
                                                         option_type, current_date_str, check_time, iv)

                    print(f"  {check_time}: ${current_price:.2f}", end="")

                    # Check stop-loss
                    if current_price <= stoploss_price:
                        print(f" - STOP-LOSS TRIGGERED!")
                        days_held = (check_date - datetime.strptime(date, "%Y-%m-%d")).days
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                        pnl_dollar = (current_price - entry_price) * 100

                        return {
                            'entry_price': round(entry_price, 2),
                            'exit_price': round(current_price, 2),
                            'exit_time': f"{current_date_str} {check_time}",
                            'exit_reason': 'stoploss',
                            'pnl_percent': round(pnl_percent, 2),
                            'pnl_dollar': round(pnl_dollar, 2),
                            'days_held': days_held
                        }

                    # Check take-profit
                    if current_price >= takeprofit_price:
                        print(f" - TAKE-PROFIT TRIGGERED!")
                        days_held = (check_date - datetime.strptime(date, "%Y-%m-%d")).days
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                        pnl_dollar = (current_price - entry_price) * 100

                        return {
                            'entry_price': round(entry_price, 2),
                            'exit_price': round(current_price, 2),
                            'exit_time': f"{current_date_str} {check_time}",
                            'exit_reason': 'takeprofit',
                            'pnl_percent': round(pnl_percent, 2),
                            'pnl_dollar': round(pnl_dollar, 2),
                            'days_held': days_held
                        }

                    print()  # New line

                except Exception as e:
                    print(f" - Error: {e}")
                    continue  # Skip this hour if data unavailable
        else:
            # End-of-day monitoring only
            try:
                current_price = option_price_historical(ticker_symbol, strike_date, strike_price,
                                                       option_type, current_date_str, iv)

                print(f"  EOD: ${current_price:.2f}", end="")

                # Check stop-loss
                if current_price <= stoploss_price:
                    print(f" - STOP-LOSS TRIGGERED!")
                    days_held = (check_date - datetime.strptime(date, "%Y-%m-%d")).days
                    pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    pnl_dollar = (current_price - entry_price) * 100

                    return {
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(current_price, 2),
                        'exit_time': f"{current_date_str} 16:00",
                        'exit_reason': 'stoploss',
                        'pnl_percent': round(pnl_percent, 2),
                        'pnl_dollar': round(pnl_dollar, 2),
                        'days_held': days_held
                    }

                # Check take-profit
                if current_price >= takeprofit_price:
                    print(f" - TAKE-PROFIT TRIGGERED!")
                    days_held = (check_date - datetime.strptime(date, "%Y-%m-%d")).days
                    pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    pnl_dollar = (current_price - entry_price) * 100

                    return {
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(current_price, 2),
                        'exit_time': f"{current_date_str} 16:00",
                        'exit_reason': 'takeprofit',
                        'pnl_percent': round(pnl_percent, 2),
                        'pnl_dollar': round(pnl_dollar, 2),
                        'days_held': days_held
                    }

                print()  # New line

            except Exception as e:
                print(f"  Error: {e}")
                # Continue to next day even if EOD data unavailable

        # Move to next day
        current_date += timedelta(days=1)

    # Exited loop - determine why
    if current_date.date() > today:
        # Reached current date boundary - position still open
        print(f"\nReached current date ({today}). Position still open.")

        # Try to get the most recent price
        try:
            last_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
            # Skip weekends for last price check
            last_check = current_date - timedelta(days=1)
            while last_check.weekday() >= 5:
                last_check -= timedelta(days=1)
            last_date_str = last_check.strftime("%Y-%m-%d")

            # Try intraday first if available, otherwise EOD
            days_from_now = (datetime.now() - last_check).days
            if days_from_now <= 60:
                last_price = option_price_intraday(ticker_symbol, strike_date, strike_price,
                                                   option_type, last_date_str, "16:00", iv)
            else:
                last_price = option_price_historical(ticker_symbol, strike_date, strike_price,
                                                     option_type, last_date_str, iv)
        except Exception as e:
            print(f"Warning: Could not fetch last price. Using entry price. Error: {e}")
            last_price = entry_price
            last_date_str = date

        days_held = (last_check - datetime.strptime(date, "%Y-%m-%d")).days
        pnl_percent = ((last_price - entry_price) / entry_price) * 100
        pnl_dollar = (last_price - entry_price) * 100

        return {
            'entry_price': round(entry_price, 2),
            'exit_price': round(last_price, 2),
            'exit_time': f"{last_date_str} (current)",
            'exit_reason': 'position_open',
            'pnl_percent': round(pnl_percent, 2),
            'pnl_dollar': round(pnl_dollar, 2),
            'days_held': days_held
        }
    else:
        # Reached expiration without hitting limits
        print(f"\nOption expired at {strike_date}")
        exit_price = option_price_historical(ticker_symbol, strike_date, strike_price,
                                            option_type, strike_date, iv)

        days_held = (expiration_date - datetime.strptime(date, "%Y-%m-%d")).days
        pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        pnl_dollar = (exit_price - entry_price) * 100

        return {
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'exit_time': f"{strike_date} 16:00",
            'exit_reason': 'expiration',
            'pnl_percent': round(pnl_percent, 2),
            'pnl_dollar': round(pnl_dollar, 2),
            'days_held': days_held
        }


def import_tickers_from_csv(csv_file='nasdaq_tickers.csv'):
    """
    Import ticker data from CSV file.

    Args:
        csv_file (str): Path to the CSV file (default: 'nasdaq_tickers.csv')

    Returns:
        pandas.DataFrame: DataFrame containing all ticker information

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found. Please run fetch_nasdaq_tickers.py first.")

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} tickers from {csv_file}")
    print(f"\nColumns: {', '.join(df.columns.tolist())}")
    print(f"\nSector breakdown:")
    print(df['Sector'].value_counts())

    return df


def update(csv_file='nasdaq_tickers.csv', save_backup=True):
    """
    Update all ticker data in the CSV file with latest market prices and information.

    Args:
        csv_file (str): Path to the CSV file to update (default: 'nasdaq_tickers.csv')
        save_backup (bool): Whether to save a backup before updating (default: True)

    Returns:
        pandas.DataFrame: Updated DataFrame with current market data

    This function:
    - Reads the existing CSV file
    - Fetches latest data for each ticker
    - Updates: Last Price, Volume, 52 Week High/Low, Market Cap, P/E, etc.
    - Saves the updated data back to CSV
    """
    import time

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found. Please run fetch_nasdaq_tickers.py first.")

    # Load existing data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} tickers from {csv_file}")

    # Save backup if requested
    if save_backup:
        backup_file = csv_file.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(backup_file, index=False)
        print(f"Backup saved to {backup_file}")

    print(f"\nUpdating ticker data...")
    print("-" * 60)

    updated_count = 0
    failed_count = 0
    failed_tickers = []

    for index, row in df.iterrows():
        ticker_symbol = row['Ticker']

        try:
            print(f"[{index+1}/{len(df)}] Updating {ticker_symbol}...", end=' ')

            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            hist = ticker.history(period='5d')

            if len(hist) == 0:
                print("No price data - skipped")
                failed_count += 1
                failed_tickers.append(ticker_symbol)
                continue

            last_price = hist['Close'].iloc[-1]

            # Update the row with latest data
            df.at[index, 'Last Price'] = round(last_price, 2)
            df.at[index, 'Volume'] = int(hist['Volume'].iloc[-1])
            df.at[index, 'Market Cap'] = info.get('marketCap', df.at[index, 'Market Cap'])
            df.at[index, '52 Week High'] = info.get('fiftyTwoWeekHigh', df.at[index, '52 Week High'])
            df.at[index, '52 Week Low'] = info.get('fiftyTwoWeekLow', df.at[index, '52 Week Low'])
            df.at[index, 'Average Volume'] = info.get('averageVolume', df.at[index, 'Average Volume'])
            df.at[index, 'P/E Ratio'] = info.get('trailingPE', df.at[index, 'P/E Ratio'])
            df.at[index, 'Dividend Yield'] = info.get('dividendYield', df.at[index, 'Dividend Yield'])
            df.at[index, 'Beta'] = info.get('beta', df.at[index, 'Beta'])
            df.at[index, 'Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print(f"${last_price:.2f} ✓")
            updated_count += 1

            # Rate limiting
            if (index + 1) % 10 == 0:
                time.sleep(1)

        except Exception as e:
            print(f"Error: {str(e)[:50]}")
            failed_count += 1
            failed_tickers.append(ticker_symbol)
            continue

    # Save updated data
    df.to_csv(csv_file, index=False)

    print("-" * 60)
    print(f"\nUpdate complete!")
    print(f"Successfully updated: {updated_count} tickers")
    print(f"Failed to update: {failed_count} tickers")

    if failed_tickers:
        print(f"\nFailed tickers: {', '.join(failed_tickers[:20])}")
        if len(failed_tickers) > 20:
            print(f"... and {len(failed_tickers) - 20} more")

    print(f"\nUpdated data saved to {csv_file}")

    return df


def get_ticker_info(ticker_symbol, csv_file='nasdaq_tickers.csv'):
    """
    Get information for a specific ticker from the CSV file.

    Args:
        ticker_symbol (str): The ticker symbol to look up (e.g., 'AAPL')
        csv_file (str): Path to the CSV file (default: 'nasdaq_tickers.csv')

    Returns:
        dict: Dictionary containing all information for the ticker
        None: If ticker not found
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

    df = pd.read_csv(csv_file)
    ticker_data = df[df['Ticker'] == ticker_symbol.upper()]

    if len(ticker_data) == 0:
        print(f"Ticker '{ticker_symbol}' not found in {csv_file}")
        return None

    return ticker_data.iloc[0].to_dict()


def filter_tickers_by_sector(sector, csv_file='nasdaq_tickers.csv'):
    """
    Filter tickers by sector.

    Args:
        sector (str): Sector name (e.g., 'Technology', 'Healthcare')
        csv_file (str): Path to the CSV file (default: 'nasdaq_tickers.csv')

    Returns:
        pandas.DataFrame: DataFrame containing tickers from the specified sector
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

    df = pd.read_csv(csv_file)
    filtered = df[df['Sector'].str.contains(sector, case=False, na=False)]

    print(f"Found {len(filtered)} tickers in sector '{sector}'")
    return filtered
