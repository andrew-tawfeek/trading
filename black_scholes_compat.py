"""
Python 3.13-compatible Black-Scholes implementation.

This module provides a drop-in replacement for py_vollib that works with Python 3.13.
py_vollib has compatibility issues with Python 3.13 due to numba dependencies.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict


def black_scholes(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        flag: 'c' for call, 'p' for put
        S: Current stock price
        K: Strike price
        t: Time to expiration (in years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)

    Returns:
        float: Option price
    """
    if t <= 0:
        # At or past expiration
        if flag == 'c':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if flag == 'c':
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)


def delta(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Calculate option delta."""
    if t <= 0:
        if flag == 'c':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    if flag == 'c':
        return float(norm.cdf(d1))
    else:  # put
        return float(norm.cdf(d1) - 1)


def gamma(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Calculate option gamma."""
    if t <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(t)))


def vega(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Calculate option vega (sensitivity to volatility)."""
    if t <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return float(S * norm.pdf(d1) * np.sqrt(t) / 100)  # Divided by 100 for 1% change


def theta(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Calculate option theta (time decay)."""
    if t <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if flag == 'c':
        theta_val = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(t))
                    - r * K * np.exp(-r * t) * norm.cdf(d2))
    else:  # put
        theta_val = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(t))
                    + r * K * np.exp(-r * t) * norm.cdf(-d2))

    return float(theta_val / 365)  # Per day


def rho(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Calculate option rho (sensitivity to interest rate)."""
    if t <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if flag == 'c':
        rho_val = K * t * np.exp(-r * t) * norm.cdf(d2)
    else:  # put
        rho_val = -K * t * np.exp(-r * t) * norm.cdf(-d2)

    return float(rho_val / 100)  # Per 1% change


def greeks(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> Dict[str, float]:
    """
    Calculate all Greeks at once.

    Returns:
        dict: Dictionary with keys 'delta', 'gamma', 'vega', 'theta', 'rho'
    """
    return {
        'delta': delta(flag, S, K, t, r, sigma),
        'gamma': gamma(flag, S, K, t, r, sigma),
        'vega': vega(flag, S, K, t, r, sigma),
        'theta': theta(flag, S, K, t, r, sigma),
        'rho': rho(flag, S, K, t, r, sigma)
    }


# Test if this module works
if __name__ == "__main__":
    # Test with typical values
    S = 100  # Stock price
    K = 100  # Strike
    t = 0.25  # 3 months
    r = 0.05  # 5% risk-free rate
    sigma = 0.2  # 20% volatility

    print("Testing Python 3.13-compatible Black-Scholes:")
    print(f"Stock: ${S}, Strike: ${K}, Time: {t}y, Rate: {r}, Vol: {sigma}")
    print(f"\nCall price: ${black_scholes('c', S, K, t, r, sigma):.2f}")
    print(f"Put price: ${black_scholes('p', S, K, t, r, sigma):.2f}")

    print(f"\nCall Greeks:")
    call_greeks = greeks('c', S, K, t, r, sigma)
    for name, value in call_greeks.items():
        print(f"  {name}: {value:.4f}")

    print(f"\nPut Greeks:")
    put_greeks = greeks('p', S, K, t, r, sigma)
    for name, value in put_greeks.items():
        print(f"  {name}: {value:.4f}")
