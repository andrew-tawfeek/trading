"""
YFinance Data Caching Module

This module provides a caching layer for YFinance downloads to prevent rate limiting
and improve performance when repeatedly accessing the same data.
"""

import yfinance as yf
import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import hashlib


class YFinanceCache:
    """
    A simple file-based cache for YFinance data.

    Caches downloaded stock data to disk to avoid repeated API calls.
    Cache entries expire after a configurable time period.
    """

    def __init__(self, cache_dir: str = ".yfinance_cache", cache_expiry_hours: int = 24):
        """
        Initialize the YFinance cache.

        Parameters:
        -----------
        cache_dir : str
            Directory to store cache files. Default is '.yfinance_cache' in current directory.
        cache_expiry_hours : int
            Number of hours before cache entries expire. Default is 24 hours.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_expiry = timedelta(hours=cache_expiry_hours)

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, ticker: str, start_date: str, end_date: str,
                       interval: str, auto_adjust: bool) -> str:
        """
        Generate a unique cache key based on download parameters.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date
        end_date : str
            End date
        interval : str
            Data interval
        auto_adjust : bool
            Auto adjust parameter

        Returns:
        --------
        str
            Unique cache key (filename)
        """
        # Create a string with all parameters
        params = f"{ticker}_{start_date}_{end_date}_{interval}_{auto_adjust}"

        # Hash it to create a filename-safe key
        hash_obj = hashlib.md5(params.encode())
        return f"{ticker}_{hash_obj.hexdigest()}.pkl"

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """
        Check if a cache file is still valid (not expired).

        Parameters:
        -----------
        cache_file : Path
            Path to the cache file

        Returns:
        --------
        bool
            True if cache is valid, False otherwise
        """
        if not cache_file.exists():
            return False

        # Check file modification time
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - mod_time

        return age < self.cache_expiry

    def get(self, ticker: str, start_date: str, end_date: str,
            interval: str = '1d', auto_adjust: bool = False) -> Optional[pd.DataFrame]:
        """
        Get data from cache if available and valid.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date
        end_date : str
            End date
        interval : str
            Data interval. Default is '1d'
        auto_adjust : bool
            Auto adjust parameter. Default is False

        Returns:
        --------
        pd.DataFrame or None
            Cached data if available and valid, None otherwise
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date, interval, auto_adjust)
        cache_file = self.cache_dir / cache_key

        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                # print(f"[CACHE HIT] Loaded {ticker} data from cache")
                return data
            except Exception as e:
                print(f"[CACHE ERROR] Failed to load cache: {e}")
                # If cache is corrupted, delete it
                cache_file.unlink(missing_ok=True)
                return None

        return None

    def set(self, ticker: str, start_date: str, end_date: str,
            interval: str, auto_adjust: bool, data: pd.DataFrame):
        """
        Store data in cache.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date
        end_date : str
            End date
        interval : str
            Data interval
        auto_adjust : bool
            Auto adjust parameter
        data : pd.DataFrame
            Data to cache
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date, interval, auto_adjust)
        cache_file = self.cache_dir / cache_key

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"[CACHE SAVED] Cached {ticker} data for future use")
        except Exception as e:
            print(f"[CACHE ERROR] Failed to save cache: {e}")

    def clear(self):
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print(f"[CACHE CLEARED] Removed all cached data")

    def download(self, ticker: str, start: str, end: str,
                interval: str = '1d', progress: bool = False,
                auto_adjust: bool = False) -> pd.DataFrame:
        """
        Download stock data with caching.

        This method checks the cache first. If data is not cached or expired,
        it downloads from YFinance and caches the result.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start : str
            Start date in format 'YYYY-MM-DD'
        end : str
            End date in format 'YYYY-MM-DD'
        interval : str
            Data interval ('1d', '1h', '5m', etc.). Default is '1d'
        progress : bool
            Show download progress bar. Default is False
        auto_adjust : bool
            Auto adjust all OHLC values. Default is False

        Returns:
        --------
        pd.DataFrame
            Stock data
        """
        # Try to get from cache first
        cached_data = self.get(ticker, start, end, interval, auto_adjust)

        if cached_data is not None:
            return cached_data

        # Cache miss - download from YFinance
        print(f"[CACHE MISS] Downloading {ticker} data from {start} to {end}...")
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=progress,
            auto_adjust=auto_adjust
        )

        # Cache the downloaded data
        if not data.empty:
            self.set(ticker, start, end, interval, auto_adjust, data)

        return data


# Global cache instance
_global_cache = None


def get_cache(cache_dir: str = ".yfinance_cache",
              cache_expiry_hours: int = 24) -> YFinanceCache:
    """
    Get or create the global cache instance.

    Parameters:
    -----------
    cache_dir : str
        Directory to store cache files
    cache_expiry_hours : int
        Number of hours before cache entries expire

    Returns:
    --------
    YFinanceCache
        Global cache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = YFinanceCache(cache_dir, cache_expiry_hours)

    return _global_cache


def download_cached(ticker: str, start: str, end: str,
                    interval: str = '1d', progress: bool = False,
                    auto_adjust: bool = False) -> pd.DataFrame:
    """
    Convenience function to download stock data with caching.

    Uses the global cache instance.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start : str
        Start date in format 'YYYY-MM-DD'
    end : str
        End date in format 'YYYY-MM-DD'
    interval : str
        Data interval ('1d', '1h', '5m', etc.). Default is '1d'
    progress : bool
        Show download progress bar. Default is False
    auto_adjust : bool
        Auto adjust all OHLC values. Default is False

    Returns:
    --------
    pd.DataFrame
        Stock data
    """
    cache = get_cache()
    return cache.download(ticker, start, end, interval, progress, auto_adjust)


def clear_cache():
    """Clear all cached data."""
    cache = get_cache()
    cache.clear()


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    print("Example 1: Basic caching")
    data1 = download_cached('AAPL', '2024-01-01', '2024-12-20')
    print(f"Downloaded {len(data1)} rows\n")

    # Second call should hit cache
    print("Example 2: Cache hit")
    data2 = download_cached('AAPL', '2024-01-01', '2024-12-20')
    print(f"Downloaded {len(data2)} rows\n")

    # Different parameters = different cache entry
    print("Example 3: Different parameters")
    data3 = download_cached('AAPL', '2024-01-01', '2024-12-20', interval='1h')
    print(f"Downloaded {len(data3)} rows\n")

    # Clear cache
    print("Example 4: Clear cache")
    clear_cache()
