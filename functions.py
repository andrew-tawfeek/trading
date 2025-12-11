import yfinance as yf
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
from technical_retrievals import *

def greeks(ticker_symbol, strike_date, strike_price, option_type):
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
