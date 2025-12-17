import yfinance as yf
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
from py_vollib.black_scholes import black_scholes
import numpy as np
from datetime import datetime, timedelta
from technical_retrievals import *
import pandas as pd
import os


def greeks(ticker_symbol, strike_date, strike_price, option_type, status = False, silent = False):
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
    
    greeks_dict = {'delta': float(d), 'gamma': float(g), 'vega': float(v), 'theta': float(th), 'rho': float(r)}
    
    if status:
        if not silent:
            print("\n" + "="*70)
            print(f"GREEKS ANALYSIS: {ticker_symbol} {option_type.upper()} ${K} exp {strike_date}")
            print("="*70)
        
        # Basic Information
        if not silent:
            print(f"\nCURRENT MARKET DATA")
            print(f"   Stock Price: ${S:.2f}")
            print(f"   Strike Price: ${K:.2f}")
            print(f"   Days to Expiration: {days}")
            print(f"   Implied Volatility: {IV*100:.2f}%")
        
        # Moneyness
        moneyness_pct = ((S - K) / K) * 100
        if option_type == 'call':
            if S > K:
                moneyness = f"ITM (In-The-Money) by ${S-K:.2f} ({moneyness_pct:.2f}%)"
            elif abs(S - K) < 0.02 * K:
                moneyness = f"ATM (At-The-Money) within ${abs(S-K):.2f}"
            else:
                moneyness = f"OTM (Out-of-The-Money) by ${K-S:.2f} ({-moneyness_pct:.2f}%)"
        else:  # put
            if S < K:
                moneyness = f"ITM (In-The-Money) by ${K-S:.2f} ({-moneyness_pct:.2f}%)"
            elif abs(S - K) < 0.02 * K:
                moneyness = f"ATM (At-The-Money) within ${abs(S-K):.2f}"
            else:
                moneyness = f"OTM (Out-of-The-Money) by ${S-K:.2f} ({moneyness_pct:.2f}%)"
        
        if not silent:
            print(f"   Moneyness: {moneyness}")
        
        # Raw Greeks
        if not silent:
            print(f"\nRAW GREEKS VALUES")
            print(f"   Delta (Δ):   {d:>8.4f}")
            print(f"   Gamma (Γ):   {g:>8.4f}")
            print(f"   Vega (ν):    {v:>8.4f}")
            print(f"   Theta (Θ):   {th:>8.4f}")
            print(f"   Rho (ρ):     {r:>8.4f}")
        
        # DELTA ANALYSIS
        delta_prob = abs(d) * 100
        if not silent:
            print(f"\nDELTA ANALYSIS")
            print(f"   Probability of Expiring ITM: ~{delta_prob:.1f}%")
            print(f"   Equivalent Share Position: {d*100:.2f} shares")
        
        if abs(d) < 0.25:
            delta_class = "Deep OTM - Low directional exposure"
        elif abs(d) < 0.45:
            delta_class = "OTM - Moderate directional exposure"
        elif abs(d) < 0.55:
            delta_class = "ATM - High directional exposure"
        elif abs(d) < 0.75:
            delta_class = "ITM - Strong directional exposure"
        else:
            delta_class = "Deep ITM - Very strong directional exposure"
        if not silent:
            print(f"   Delta Classification: {delta_class}")
        
        if option_type == 'call':
            delta_direction = "Bullish" if d > 0 else "Neutral/Bearish"
        else:
            delta_direction = "Bearish" if d < 0 else "Neutral/Bullish"
        if not silent:
            print(f"   Direction: {delta_direction}")
        
        # GAMMA ANALYSIS
        if not silent:
            print(f"\nGAMMA ANALYSIS")
            print(f"   Delta Change per $1 Move: {g:.4f}")
        
        gamma_normalized = g * S  # Normalize by stock price
        if gamma_normalized < 0.01:
            gamma_risk = "Low - Stable delta"
        elif gamma_normalized < 0.03:
            gamma_risk = "Medium - Moderate delta acceleration"
        elif gamma_normalized < 0.05:
            gamma_risk = "High - Significant delta changes expected"
        else:
            gamma_risk = "Very High - Extreme delta sensitivity"
        if not silent:
            print(f"   Gamma Risk Level: {gamma_risk}")
        
        # Distance from maximum gamma (ATM)
        distance_from_atm = abs(S - K) / S * 100
        if not silent:
            print(f"   Distance from Max Gamma (ATM): {distance_from_atm:.2f}%")
        
        if days < 10 and distance_from_atm < 5 and not silent:
            print(f"   WARNING: High gamma risk near expiration!")
        
        # VEGA ANALYSIS
        if not silent:
            print(f"\nVEGA ANALYSIS")
            print(f"   P&L per 1% IV Change: ${v:.2f}")
        
        vega_exposure = "Long Volatility" if v > 0 else "Short Volatility"
        if not silent:
            print(f"   Volatility Exposure: {vega_exposure}")
        
        # Vega as percentage of option value (approximate)
        if abs(v) < 0.05:
            vega_class = "Low IV Sensitivity"
        elif abs(v) < 0.15:
            vega_class = "Medium IV Sensitivity"
        elif abs(v) < 0.30:
            vega_class = "High IV Sensitivity"
        else:
            vega_class = "Very High IV Sensitivity"
        if not silent:
            print(f"   Vega Classification: {vega_class}")
        
        # THETA ANALYSIS
        if not silent:
            print(f"\nTHETA ANALYSIS")
            print(f"   Daily Time Decay: ${th:.2f}")
            print(f"   Weekly Time Decay: ${th*7:.2f}")
            print(f"   Monthly Time Decay (30d): ${th*30:.2f}")
        
        # Theta acceleration warning
        if not silent:
            if days < 30:
                print(f"   WARNING: Entering theta acceleration zone (<30 DTE)")
            if days < 14:
                print(f"   WARNING: HIGH theta decay period (<14 DTE)")
            if days < 7:
                print(f"   WARNING: EXTREME theta decay period (<7 DTE)")
        
        # Theta magnitude classification
        abs_theta = abs(th)
        if abs_theta < 0.02:
            theta_class = "Low - Minimal time decay"
        elif abs_theta < 0.05:
            theta_class = "Medium - Moderate time decay"
        elif abs_theta < 0.10:
            theta_class = "High - Significant time decay"
        else:
            theta_class = "Very High - Severe time decay"
        if not silent:
            print(f"   Theta Classification: {theta_class}")
        
        # Break-even move to offset theta
        if abs(d) > 0.01:
            breakeven_move = abs(th / d)
            if not silent:
                print(f"   Stock Move to Offset Daily Theta: ${breakeven_move:.2f}")
        
        # RHO ANALYSIS
        if not silent:
            print(f"\nRHO ANALYSIS")
            print(f"   P&L per 1% Rate Change: ${r:.2f}")
        
        # Rho is generally negligible for short-dated options
        if days < 90:
            rho_relevance = "Negligible (short-dated option)"
        elif days < 180:
            rho_relevance = "Low (medium-dated option)"
        else:
            rho_relevance = "Moderate (long-dated option)"
        if not silent:
            print(f"   Interest Rate Sensitivity: {rho_relevance}")
        
        # RISK METRICS
        if not silent:
            print(f"\nRISK METRICS")
        
        # Delta/Theta ratio (reward per day of time decay)
        if th != 0:
            delta_theta_ratio = abs(d / th)
            if not silent:
                print(f"   Delta/Theta Ratio: {delta_theta_ratio:.2f} (directional gain needed per $1 theta)")
        
        # Gamma/Vega ratio
        if v != 0:
            gamma_vega_ratio = abs(g / v)
            if not silent:
                print(f"   Gamma/Vega Ratio: {gamma_vega_ratio:.4f} (convexity vs volatility)")
        
        # Total Greek Risk Score (weighted combination)
        risk_score = (abs(d) * 0.3 + abs(g) * S * 0.3 + abs(v) * 0.2 + 
                      abs(th) * 10 * 0.15 + abs(r) * 0.05)
        if not silent:
            print(f"   Composite Risk Score: {risk_score:.2f}")
        
        # SENSITIVITY SCENARIOS
        if not silent:
            print(f"\nSENSITIVITY SCENARIOS")
        
        # Stock price moves
        move_1pct = S * 0.01
        move_5pct = S * 0.05
        move_10pct = S * 0.10
        
        pl_up_1 = d * move_1pct + 0.5 * g * move_1pct**2
        pl_down_1 = d * (-move_1pct) + 0.5 * g * (-move_1pct)**2
        pl_up_5 = d * move_5pct + 0.5 * g * move_5pct**2
        pl_down_5 = d * (-move_5pct) + 0.5 * g * (-move_5pct)**2
        pl_up_10 = d * move_10pct + 0.5 * g * move_10pct**2
        pl_down_10 = d * (-move_10pct) + 0.5 * g * (-move_10pct)**2
        
        if not silent:
            print(f"   Stock Move Scenarios:")
            print(f"      +1% (${S + move_1pct:.2f}): P&L ≈ ${pl_up_1:.2f}")
            print(f"      -1% (${S - move_1pct:.2f}): P&L ≈ ${pl_down_1:.2f}")
            print(f"      +5% (${S + move_5pct:.2f}): P&L ≈ ${pl_up_5:.2f}")
            print(f"      -5% (${S - move_5pct:.2f}): P&L ≈ ${pl_down_5:.2f}")
            print(f"     +10% (${S + move_10pct:.2f}): P&L ≈ ${pl_up_10:.2f}")
            print(f"     -10% (${S - move_10pct:.2f}): P&L ≈ ${pl_down_10:.2f}")
        
        # IV scenarios
        if not silent:
            print(f"   IV Change Scenarios:")
            print(f"      +10% IV: P&L ≈ ${v * 10:.2f}")
            print(f"      -10% IV: P&L ≈ ${-v * 10:.2f}")
            print(f"      +25% IV: P&L ≈ ${v * 25:.2f}")
            print(f"      -25% IV: P&L ≈ ${-v * 25:.2f}")
            print(f"      +50% IV: P&L ≈ ${v * 50:.2f}")
            print(f"      -50% IV: P&L ≈ ${-v * 50:.2f}")
        
        # Time decay scenarios
        if not silent:
            print(f"   Time Decay Scenarios:")
            print(f"      After 1 day: P&L ≈ ${th:.2f}")
            print(f"      After 1 week: P&L ≈ ${th * 7:.2f}")
            print(f"      After 2 weeks: P&L ≈ ${th * 14:.2f}")
        
        # STOP-LOSS AND TAKE-PROFIT RECOMMENDATIONS
        if not silent:
            print(f"\nSTOP-LOSS & TAKE-PROFIT RECOMMENDATIONS")
        
        # Get current option price (mid price as estimate)
        option_price = option.get('lastPrice', 0)
        if option_price == 0:
            # Try to estimate from bid/ask
            bid = option.get('bid', 0)
            ask = option.get('ask', 0)
            if bid > 0 and ask > 0:
                option_price = (bid + ask) / 2
        
        if option_price > 0:
            if not silent:
                print(f"   Current Option Price: ${option_price:.2f}")
            
            # Stop-loss recommendations based on risk profile
            # Conservative: 20-30% loss
            # Moderate: 30-50% loss
            # Aggressive: 50-70% loss
            
            # Adjust based on Greeks characteristics
            if days < 14:
                # Short-dated options: tighter stops due to theta
                sl_conservative = option_price * 0.75  # 25% loss
                sl_moderate = option_price * 0.60      # 40% loss
                sl_aggressive = option_price * 0.45    # 55% loss
                stop_note = "(Tighter stops for short DTE)"
            elif abs(th) > 0.10:
                # High theta decay: tighter stops
                sl_conservative = option_price * 0.75
                sl_moderate = option_price * 0.60
                sl_aggressive = option_price * 0.45
                stop_note = "(Tighter stops due to high theta)"
            elif abs(d) < 0.30:
                # Far OTM: tighter stops (lower probability)
                sl_conservative = option_price * 0.75
                sl_moderate = option_price * 0.55
                sl_aggressive = option_price * 0.40
                stop_note = "(Tighter stops for OTM option)"
            else:
                # Standard stops
                sl_conservative = option_price * 0.70  # 30% loss
                sl_moderate = option_price * 0.50      # 50% loss
                sl_aggressive = option_price * 0.30    # 70% loss
                stop_note = "(Standard risk profile)"
            
            if not silent:
                print(f"\n   Stop-Loss Levels {stop_note}:")
                print(f"      Conservative (25-30% loss): ${sl_conservative:.2f}")
                print(f"      Moderate (40-50% loss):     ${sl_moderate:.2f}")
                print(f"      Aggressive (55-70% loss):   ${sl_aggressive:.2f}")
            
            # Take-profit recommendations
            # Based on risk-reward ratios and option characteristics
            
            if abs(d) > 0.70:
                # Deep ITM: smaller percentage gains expected
                tp_conservative = option_price * 1.25  # 25% gain
                tp_moderate = option_price * 1.50      # 50% gain
                tp_aggressive = option_price * 1.75    # 75% gain
                tp_note = "(Lower % targets for ITM)"
            elif abs(d) < 0.30:
                # Far OTM: higher percentage gains possible
                tp_conservative = option_price * 1.50  # 50% gain
                tp_moderate = option_price * 2.00      # 100% gain
                tp_aggressive = option_price * 3.00    # 200% gain
                tp_note = "(Higher % targets for OTM)"
            else:
                # ATM: balanced targets
                tp_conservative = option_price * 1.40  # 40% gain
                tp_moderate = option_price * 1.75      # 75% gain
                tp_aggressive = option_price * 2.25    # 125% gain
                tp_note = "(Balanced targets for ATM)"
            
            if not silent:
                print(f"\n   Take-Profit Levels {tp_note}:")
                print(f"      Conservative (25-50% gain):  ${tp_conservative:.2f}")
                print(f"      Moderate (50-100% gain):     ${tp_moderate:.2f}")
                print(f"      Aggressive (100-200% gain):  ${tp_aggressive:.2f}")
            
            # Risk-reward ratios
            rr_conservative = (tp_conservative - option_price) / (option_price - sl_conservative)
            rr_moderate = (tp_moderate - option_price) / (option_price - sl_moderate)
            rr_aggressive = (tp_aggressive - option_price) / (option_price - sl_aggressive)
            
            if not silent:
                print(f"\n   Risk-Reward Ratios:")
                print(f"      Conservative: {rr_conservative:.2f}:1")
                print(f"      Moderate:     {rr_moderate:.2f}:1")
                print(f"      Aggressive:   {rr_aggressive:.2f}:1")
            
            # Underlying stock price levels for stop/target
            # Calculate what stock price would cause option to hit these levels
            if not silent:
                print(f"\n   Approximate Stock Price Levels:")
            
            # For stop-loss - estimate stock move needed
            # This is approximate: (stop_price - current_price) / delta
            if abs(d) > 0.05:
                stock_sl_conservative = S + (sl_conservative - option_price) / d
                stock_sl_moderate = S + (sl_moderate - option_price) / d
                stock_sl_aggressive = S + (sl_aggressive - option_price) / d
                
                stock_tp_conservative = S + (tp_conservative - option_price) / d
                stock_tp_moderate = S + (tp_moderate - option_price) / d
                stock_tp_aggressive = S + (tp_aggressive - option_price) / d
                
                if not silent:
                    print(f"      Stop-Loss Stock Prices:")
                    print(f"         Conservative: ${stock_sl_conservative:.2f} ({((stock_sl_conservative/S - 1)*100):+.1f}%)")
                    print(f"         Moderate:     ${stock_sl_moderate:.2f} ({((stock_sl_moderate/S - 1)*100):+.1f}%)")
                    print(f"         Aggressive:   ${stock_sl_aggressive:.2f} ({((stock_sl_aggressive/S - 1)*100):+.1f}%)")
                    
                    print(f"      Take-Profit Stock Prices:")
                    print(f"         Conservative: ${stock_tp_conservative:.2f} ({((stock_tp_conservative/S - 1)*100):+.1f}%)")
                    print(f"         Moderate:     ${stock_tp_moderate:.2f} ({((stock_tp_moderate/S - 1)*100):+.1f}%)")
                    print(f"         Aggressive:   ${stock_tp_aggressive:.2f} ({((stock_tp_aggressive/S - 1)*100):+.1f}%)")
            else:
                if not silent:
                    print(f"      (Stock price estimates unavailable - delta too low)")
            
            # Add trailing stop recommendation
            if days > 30:
                trailing_pct = 25
            elif days > 14:
                trailing_pct = 20
            else:
                trailing_pct = 15
            
            if not silent:
                print(f"\n   Trailing Stop Recommendation:")
                print(f"      Trailing stop: {trailing_pct}% from peak")
                print(f"      (Adjust tighter as expiration approaches)")
            
        else:
            if not silent:
                print(f"   (Option price data unavailable for stop-loss/take-profit calculation)")

        
        # PURCHASE RECOMMENDATION SCORE
        if not silent:
            print(f"\nPURCHASE RECOMMENDATION ANALYSIS")
        
        # Initialize score (0-100 scale)
        buy_score = 50  # Start neutral
        
        # Delta scoring (directional alignment)
        if option_type == 'call':
            # For calls, positive delta is good
            if d > 0.7:
                buy_score += 15
                if not silent:
                    print(f"   [+] Strong call delta ({d:.3f}): +15")
            elif d > 0.5:
                buy_score += 10
                if not silent:
                    print(f"   [+] Good call delta ({d:.3f}): +10")
            elif d > 0.3:
                buy_score += 5
                if not silent:
                    print(f"   [~] Moderate call delta ({d:.3f}): +5")
            else:
                buy_score -= 5
                if not silent:
                    print(f"   [-] Weak call delta ({d:.3f}): -5")
        else:  # put
            # For puts, negative delta is good (in absolute terms)
            if d < -0.7:
                buy_score += 15
                if not silent:
                    print(f"   [+] Strong put delta ({d:.3f}): +15")
            elif d < -0.5:
                buy_score += 10
                if not silent:
                    print(f"   [+] Good put delta ({d:.3f}): +10")
            elif d < -0.3:
                buy_score += 5
                if not silent:
                    print(f"   [~] Moderate put delta ({d:.3f}): +5")
            else:
                buy_score -= 5
                if not silent:
                    print(f"   [-] Weak put delta ({d:.3f}): -5")
        
        # Gamma scoring (positive gamma is good for long options)
        gamma_normalized = g * S
        if gamma_normalized > 0.05:
            buy_score += 10
            if not silent:
                print(f"   [+] High gamma sensitivity ({g:.4f}): +10")
        elif gamma_normalized > 0.03:
            buy_score += 5
            if not silent:
                print(f"   [~] Moderate gamma ({g:.4f}): +5")
        elif gamma_normalized < 0.01:
            buy_score -= 5
            if not silent:
                print(f"   [-] Low gamma ({g:.4f}): -5")
        
        # Theta scoring (theta decay is bad for long options)
        abs_theta = abs(th)
        if abs_theta > 0.10:
            buy_score -= 15
            if not silent:
                print(f"   [-] Very high theta decay (${th:.3f}/day): -15")
        elif abs_theta > 0.05:
            buy_score -= 10
            if not silent:
                print(f"   [-] High theta decay (${th:.3f}/day): -10")
        elif abs_theta > 0.02:
            buy_score -= 5
            if not silent:
                print(f"   [~] Moderate theta decay (${th:.3f}/day): -5")
        else:
            buy_score += 5
            if not silent:
                print(f"   [+] Low theta decay (${th:.3f}/day): +5")
        
        # Vega scoring (high vega can be good if expecting vol increase)
        abs_vega = abs(v)
        if abs_vega > 0.30:
            buy_score += 8
            if not silent:
                print(f"   [+] High vega - good IV leverage (${v:.3f}): +8")
        elif abs_vega > 0.15:
            buy_score += 5
            if not silent:
                print(f"   [~] Moderate vega (${v:.3f}): +5")
        else:
            buy_score -= 3
            if not silent:
                print(f"   [~] Low vega - limited IV sensitivity (${v:.3f}): -3")
        
        # Time to expiration scoring
        if days < 7:
            buy_score -= 20
            if not silent:
                print(f"   [-] Very short time to expiration ({days} days): -20")
        elif days < 14:
            buy_score -= 10
            if not silent:
                print(f"   [-] Short time to expiration ({days} days): -10")
        elif days < 30:
            buy_score -= 5
            if not silent:
                print(f"   [~] Limited time to expiration ({days} days): -5")
        elif days < 60:
            buy_score += 5
            if not silent:
                print(f"   [+] Good time window ({days} days): +5")
        else:
            buy_score += 10
            if not silent:
                print(f"   [+] Ample time to expiration ({days} days): +10")
        
        # Moneyness scoring
        if abs(moneyness_pct) < 2:
            buy_score += 10
            if not silent:
                print(f"   [+] Near ATM - optimal gamma/theta balance: +10")
        elif abs(moneyness_pct) < 5:
            buy_score += 5
            if not silent:
                print(f"   [~] Close to ATM: +5")
        elif abs(moneyness_pct) > 15:
            buy_score -= 10
            if not silent:
                print(f"   [-] Far from ATM - low probability: -10")
        
        # Delta/Theta efficiency
        if th != 0:
            dt_ratio = abs(d / th)
            if dt_ratio > 50:
                buy_score += 8
                if not silent:
                    print(f"   [+] Excellent delta/theta efficiency ({dt_ratio:.1f}): +8")
            elif dt_ratio > 30:
                buy_score += 5
                if not silent:
                    print(f"   [+] Good delta/theta efficiency ({dt_ratio:.1f}): +5")
            elif dt_ratio < 15:
                buy_score -= 8
                if not silent:
                    print(f"   [-] Poor delta/theta efficiency ({dt_ratio:.1f}): -8")
        
        # Normalize score to 0-100
        buy_score = max(0, min(100, buy_score))
        
        if not silent:
            print(f"\n{'='*70}")
            print(f"   FINAL PURCHASE RECOMMENDATION SCORE: {buy_score:.1f}/100")
        
        if not silent:
            # Qualitative assessment
            if buy_score >= 75:
                recommendation = "STRONG BUY"
                marker = "[+++]"
            elif buy_score >= 60:
                recommendation = "BUY"
                marker = "[++]"
            elif buy_score >= 50:
                recommendation = "NEUTRAL/SLIGHT BUY"
                marker = "[+]"
            elif buy_score >= 40:
                recommendation = "NEUTRAL/SLIGHT AVOID"
                marker = "[-]"
            elif buy_score >= 25:
                recommendation = "AVOID"
                marker = "[--]"
            else:
                recommendation = "STRONG AVOID"
                marker = "[---]"
            
            print(f"   {marker} RECOMMENDATION: {recommendation}")
            print(f"{'='*70}\n")
        else:
            # Silent mode - still determine recommendation but don't print
            if buy_score >= 75:
                recommendation = "STRONG BUY"
            elif buy_score >= 60:
                recommendation = "BUY"
            elif buy_score >= 50:
                recommendation = "NEUTRAL/SLIGHT BUY"
            elif buy_score >= 40:
                recommendation = "NEUTRAL/SLIGHT AVOID"
            elif buy_score >= 25:
                recommendation = "AVOID"
            else:
                recommendation = "STRONG AVOID"
        greeks_dict['buy_score'] = buy_score
        greeks_dict['recommendation'] = recommendation
    
    return greeks_dict


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
