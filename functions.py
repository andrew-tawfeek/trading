import yfinance as yf
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
from functions import *
import time

def greeks(ticker_symbol, strike_date, strike_price):
    days = countdown(strike_date)
    if days < 2:
        return "Error: Option too close to expiration for accurate Greeks calculation"
    
    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(strike_date)
    put = chain.puts[chain.puts['strike'] == strike_price].iloc[0]


    S = ticker.history(period='1d')['Close'].iloc[-1]
    K = put['strike']
    IV = put['impliedVolatility']
    t = countdown(strike_date)/365  # Calculate from expiration

    # Calculate Greeks
    d = delta('p', S, K, t, get_rates(strike_date), IV)
    g = gamma('p', S, K, t, get_rates(strike_date), IV)
    v = vega('p', S, K, t, get_rates(strike_date), IV)
    th = theta('p', S, K, t, get_rates(strike_date), IV)
    r = rho('p', S, K, t, get_rates(strike_date), IV)
    
    return {'delta': float(d), 'gamma': float(g), 'vega': float(v), 'theta': float(th), 'rho': float(r)}

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



def countdown(date):
    # days until date
    from datetime import datetime
    target_date = datetime.strptime(date, "%Y-%m-%d")
    today = datetime.now()
    count = target_date - today
    return count.days

def get_rates(expiration_date):
    """
    Determine appropriate risk-free rate based on time to expiration.
    
    Args:
        expiration_date (str): Expiration date in 'YYYY-MM-DD' format
        
    Returns:
        float: Risk-free rate as decimal (e.g., 0.043 for 4.3%)
    """
    import yfinance as yf
    
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
        # Fetch current rate
        rate_ticker = yf.Ticker(ticker)
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

