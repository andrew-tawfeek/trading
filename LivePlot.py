# Raw Package
import numpy as np
import pandas as pd
import yfinance as yf

# Graphing/Visualization
import datetime as dt
import plotly.graph_objs as go

# Create input field for our desired stock
stock = input("Enter a stock ticker symbol: ")

# Retrieve stock data frame (df) from yfinance API at an interval of 1m
df = yf.download(tickers=stock, period='1d', interval='1m')

# Optional: Debug DataFrame structure
# print("DataFrame columns:", df.columns)
# print("DataFrame head:", df.head())

# Declare plotly figure (go)
fig = go.Figure()

# Create candlestick chart with correct column access
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open', stock],    # Multi-index access
    high=df['High', stock],    # Including stock symbol
    low=df['Low', stock],      # for proper DataFrame
    close=df['Close', stock],  # column access
    name='market data'
))

# Update layout with title and axis labels
fig.update_layout(
    title=str(stock) + ' Live Share Price:',
    yaxis_title='Stock Price (USD per Shares)'
)

# Configure x-axis with range slider and selectors
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# Display the interactive plot
fig.show()

