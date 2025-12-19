import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load environment variables from .env file
load_dotenv()

# Initialize the Alpaca Trading Client for Paper Trading
trading_client = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
    paper=True  # Set to True for paper trading
)


def get_account():
    """Get account information"""
    account = trading_client.get_account()
    return account


def get_positions():
    """Get all open positions"""
    positions = trading_client.get_all_positions()
    return positions


def submit_market_order(symbol: str, qty: float, side: str):
    """
    Submit a market order
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        qty: Number of shares
        side: 'buy' or 'sell'
    """
    order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
    
    market_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY
    )
    
    order = trading_client.submit_order(order_data=market_order_data)
    return order


def submit_limit_order(symbol: str, qty: float, side: str, limit_price: float):
    """
    Submit a limit order
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        qty: Number of shares
        side: 'buy' or 'sell'
        limit_price: The limit price for the order
    """
    order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
    
    limit_order_data = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price
    )
    
    order = trading_client.submit_order(order_data=limit_order_data)
    return order


def get_orders():
    """Get all open orders"""
    orders = trading_client.get_orders()
    return orders


def cancel_all_orders():
    """Cancel all open orders"""
    trading_client.cancel_orders()


if __name__ == "__main__":
    # Test the connection
    account = get_account()
    print(f"Account Status: {account.status}")
    print(f"Buying Power: ${account.buying_power}")
    print(f"Portfolio Value: ${account.portfolio_value}")
