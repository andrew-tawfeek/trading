import yfinance as yf

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

