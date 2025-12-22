"""
Example: Options Greeks Analysis

This script demonstrates how to use the Greeks analysis features.
"""

from functions import greeks, option_price, options_purchase
import yfinance as yf

# Example 1: Basic Greeks Calculation
# ====================================
print("="*80)
print("EXAMPLE 1: Basic Greeks Calculation")
print("="*80)

result = greeks(
    ticker_symbol='AAPL',
    strike_date='2026-01-16',
    strike_price=150.0,
    option_type='call',
    status=False,  # No detailed analysis
    silent=True     # No output
)

print(f"\nGreeks for AAPL $150 Call (exp 2026-01-16):")
print(f"  Delta:  {result['delta']:>8.4f}")
print(f"  Gamma:  {result['gamma']:>8.4f}")
print(f"  Vega:   {result['vega']:>8.4f}")
print(f"  Theta:  {result['theta']:>8.4f}")
print(f"  Rho:    {result['rho']:>8.4f}")

# Example 2: Detailed Analysis with Recommendations
# ==================================================
print("\n" + "="*80)
print("EXAMPLE 2: Detailed Analysis with Buy Recommendation")
print("="*80)

result = greeks(
    ticker_symbol='AAPL',
    strike_date='2026-01-16',
    strike_price=150.0,
    option_type='call',
    status=True,    # Include buy recommendation
    silent=False    # Print detailed analysis
)

print(f"\nBuy Score: {result['buy_score']}/100")
print(f"Recommendation: {result['recommendation']}")

# Example 3: Comparing Multiple Strikes
# ======================================
print("\n" + "="*80)
print("EXAMPLE 3: Comparing Multiple Strikes")
print("="*80)

ticker_obj = yf.Ticker('AAPL')
current_price = ticker_obj.history(period='1d')['Close'].iloc[-1]

strikes = [
    current_price * 0.95,  # 5% OTM put
    current_price,          # ATM
    current_price * 1.05   # 5% OTM call
]

print(f"\nCurrent AAPL Price: ${current_price:.2f}\n")
print(f"{'Strike':<10} {'Type':<8} {'Delta':<10} {'Theta':<10} {'Buy Score':<12} {'Recommendation':<20}")
print("-" * 80)

for strike in strikes:
    strike_rounded = round(strike)

    # Call option
    call_result = greeks(
        'AAPL', '2026-01-16', strike_rounded, 'call',
        status=True, silent=True
    )
    print(f"${strike_rounded:<9.0f} {'Call':<8} {call_result['delta']:<10.3f} "
          f"{call_result['theta']:<10.3f} {call_result['buy_score']:<12} "
          f"{call_result['recommendation']:<20}")

    # Put option
    put_result = greeks(
        'AAPL', '2026-01-16', strike_rounded, 'put',
        status=True, silent=True
    )
    print(f"${strike_rounded:<9.0f} {'Put':<8} {put_result['delta']:<10.3f} "
          f"{put_result['theta']:<10.3f} {put_result['buy_score']:<12} "
          f"{put_result['recommendation']:<20}")

# Example 4: Simulate Option Purchase
# ====================================
print("\n" + "="*80)
print("EXAMPLE 4: Simulate Option Purchase with Risk Management")
print("="*80)

print("\nSimulating purchase of AAPL $150 call...")
print("Entry: 2025-12-10 at 10:30 AM")
print("Risk Management: 20% stop-loss, 50% take-profit\n")

result = options_purchase(
    ticker_symbol='AAPL',
    strike_date='2026-01-16',
    strike_price=150.0,
    date='2025-12-10',
    time='10:30',
    option_type='call',
    stoploss=20,
    takeprofit=50
)

print(f"\n{'='*80}")
print("TRADE RESULTS")
print(f"{'='*80}")
print(f"Entry Price:    ${result['entry_price']:.2f}")
print(f"Exit Price:     ${result['exit_price']:.2f}")
print(f"Exit Time:      {result['exit_time']}")
print(f"Exit Reason:    {result['exit_reason']}")
print(f"P&L:            {result['pnl_percent']:+.2f}% (${result['pnl_dollar']:+.2f} per contract)")
print(f"Days Held:      {result['days_held']}")

# Example 5: Screening Multiple Tickers
# ======================================
print("\n" + "="*80)
print("EXAMPLE 5: Screening Multiple Tickers for Buy Signals")
print("="*80)

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
buy_candidates = []

print(f"\nScanning {len(tickers)} tickers for buy opportunities...\n")
print(f"{'Ticker':<10} {'Strike':<10} {'Type':<8} {'Buy Score':<12} {'Recommendation':<20}")
print("-" * 70)

for ticker in tickers:
    try:
        ticker_obj = yf.Ticker(ticker)
        price = ticker_obj.history(period='1d')['Close'].iloc[-1]

        # Check ATM call and put
        strike = round(price)

        call_result = greeks(
            ticker, '2026-01-16', strike, 'call',
            status=True, silent=True
        )

        if call_result['buy_score'] >= 70:
            buy_candidates.append((ticker, strike, 'call', call_result['buy_score']))
            print(f"{ticker:<10} ${strike:<9.0f} {'Call':<8} {call_result['buy_score']:<12} "
                  f"{call_result['recommendation']:<20}")

        put_result = greeks(
            ticker, '2026-01-16', strike, 'put',
            status=True, silent=True
        )

        if put_result['buy_score'] >= 70:
            buy_candidates.append((ticker, strike, 'put', put_result['buy_score']))
            print(f"{ticker:<10} ${strike:<9.0f} {'Put':<8} {put_result['buy_score']:<12} "
                  f"{put_result['recommendation']:<20}")

    except Exception as e:
        print(f"{ticker:<10} Error: {str(e)[:50]}")

if buy_candidates:
    print(f"\n{'='*80}")
    print(f"FOUND {len(buy_candidates)} BUY CANDIDATE(S)")
    print(f"{'='*80}")
    for ticker, strike, opt_type, score in sorted(buy_candidates, key=lambda x: x[3], reverse=True):
        print(f"{ticker} ${strike} {opt_type.upper()} - Buy Score: {score}/100")
else:
    print(f"\nNo strong buy candidates found (buy score >= 70)")

print("\n" + "="*80)
print("Examples completed!")
print("="*80)
