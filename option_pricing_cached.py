"""
Cached option pricing functions to avoid redundant API calls.

This module wraps the functions from functions.py with caching to dramatically
speed up backtesting when the same rates/volatility are needed repeatedly.
"""

from functools import lru_cache
from functions import option_price_historical, greeks_historical
from technical_retrievals import get_rates, get_historical_volatility
from datetime import datetime


# ========================================
# Cached Rate and Volatility Functions
# ========================================

# Cache rates - only depends on expiration and historical date
@lru_cache(maxsize=1000)
def get_rates_cached(expiration_date: str, historical_date: str = None):
    """Cached version of get_rates to avoid redundant API calls."""
    return get_rates(expiration_date, historical_date)


# Cache volatility - only depends on ticker and date
@lru_cache(maxsize=1000)
def get_historical_volatility_cached(ticker_symbol: str, date: str, lookback_days: int = 30):
    """Cached version of get_historical_volatility to avoid redundant API calls."""
    return get_historical_volatility(ticker_symbol, date, lookback_days)


# ========================================
# Cached Option Pricing Functions
# ========================================

def option_price_historical_cached(ticker_symbol, strike_date, strike_price,
                                   option_type, historical_date, iv=None,
                                   rate_cache=None, vol_cache=None):
    """
    Calculate historical option price with caching for rates and volatility.

    This is a drop-in replacement for option_price_historical() that uses
    cached rate and volatility lookups to avoid redundant API calls.

    Args:
        ticker_symbol: Stock ticker
        strike_date: Expiration date 'YYYY-MM-DD'
        strike_price: Strike price
        option_type: 'call' or 'put'
        historical_date: Pricing date 'YYYY-MM-DD'
        iv: Optional implied volatility (if None, uses historical vol)
        rate_cache: Optional dict to cache rates
        vol_cache: Optional dict to cache volatility

    Returns:
        float: Option price
    """
    # Use the regular function - it will call get_rates and get_historical_volatility
    # which are now cached via @lru_cache
    return option_price_historical(ticker_symbol, strike_date, strike_price,
                                   option_type, historical_date, iv)


def greeks_historical_cached(ticker_symbol, strike_date, strike_price,
                             option_type, historical_date, iv=None,
                             rate_cache=None, vol_cache=None):
    """
    Calculate historical Greeks with caching for rates and volatility.

    This is a drop-in replacement for greeks_historical() that uses
    cached rate and volatility lookups to avoid redundant API calls.

    Args:
        ticker_symbol: Stock ticker
        strike_date: Expiration date 'YYYY-MM-DD'
        strike_price: Strike price
        option_type: 'call' or 'put'
        historical_date: Pricing date 'YYYY-MM-DD'
        iv: Optional implied volatility (if None, uses historical vol)
        rate_cache: Optional dict to cache rates
        vol_cache: Optional dict to cache volatility

    Returns:
        dict: Greeks dictionary
    """
    # Use the regular function - it will call get_rates and get_historical_volatility
    # which are now cached via @lru_cache
    return greeks_historical(ticker_symbol, strike_date, strike_price,
                            option_type, historical_date, iv)


# ========================================
# Cache Management
# ========================================

def clear_pricing_cache():
    """Clear all cached pricing data."""
    get_rates_cached.cache_clear()
    get_historical_volatility_cached.cache_clear()
    print("Cleared pricing cache")


def get_cache_info():
    """Get information about cache usage."""
    return {
        'rates_cache': get_rates_cached.cache_info(),
        'volatility_cache': get_historical_volatility_cached.cache_info()
    }
