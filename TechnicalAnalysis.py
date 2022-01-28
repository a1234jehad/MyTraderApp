import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

cf.go_offline()
import warnings

warnings.simplefilter("ignore")
import yfinance as yf
from plotly.subplots import make_subplots
from ta.trend import MACD
from ta.momentum import StochasticOscillator

class TechnicalAnalysis:
    def __init__(self, Ticker, period, intervals):
        # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        self.stock_df = yf.download(tickers=Ticker, period=period, interval=intervals)
        self.ticker = Ticker

    def plot_candlestick(self):
        close = self.stock_df['Adj Close']
        high = self.stock_df['High']
        low = self.stock_df['Low']
        open = self.stock_df['Open']

        self.stock_df['MA5'] = self.stock_df.Close.rolling(5).mean()
        self.stock_df['MA20'] = self.stock_df.Close.rolling(20).mean()
        # Calculates 5 and 20 day moving average

        candles = go.Candlestick(x=self.stock_df.index, open=open, high=high,
                                 low=low, close=close, name="Candles")

        ma5 = go.Scatter(x=self.stock_df.index, y=self.stock_df.MA5,
                         line=dict(color='orange', width=1), name="MA5")
        ma20 = go.Scatter(x=self.stock_df.index, y=self.stock_df.MA20,
                          line=dict(color='green', width=1), name="MA20")
        # Create 5 and 20 day moving average for uptrend etc..

        vol = go.Bar(x=self.stock_df.index, y=self.stock_df['Volume'], name="Volume")  # Create volume bar chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add plots
        fig.add_trace(trace=candles, secondary_y=True)
        fig.add_trace(trace=ma5, secondary_y=True)
        fig.add_trace(trace=ma20, secondary_y=True)
        fig.add_trace(trace=vol, secondary_y=False)
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=15,
                             label="15M",
                             step="minute",
                             stepmode="backward"),
                        dict(count=45,
                             label="45M",
                             step="minute",
                             stepmode="backward"),
                        dict(count=1,
                             label="1HR",
                             step="hour",
                             stepmode="todate"),
                        dict(count=1,
                             label="1D",
                             step="day",
                             stepmode="todate"),
                        dict(count=3,
                             label="3D",
                             step="day",
                             stepmode="todate"),
                        dict(count=7,
                             label="1W",
                             step="day",
                             stepmode="backward"),
                        dict(count=14,
                             label="2W",
                             step="day",
                             stepmode="backward"),
                        dict(count=1,
                             label="1M",
                             step="month",
                             stepmode="backward"),
                        dict(label="All", step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        fig.update_layout(xaxis_title="Dates", yaxis_title="Stock Price",
                          title="Microsoft Candlestick Chart",
                          width=1000, height=800)

        if (self.ticker[-2:] == "SR"):
            fig.update_xaxes(
                rangeslider_visible=True,
                rangebreaks=[
                    dict(bounds=["fri", "sun"]),
                    dict(bounds=[15, 10], pattern="hour"),
                ]
            )
        else:
            fig.update_xaxes(
                rangeslider_visible=True,
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16, 9.5], pattern="hour"),
                    dict(values=["2020-12-25", "2021-01-01", "2021-07-04"])
                ]
            )
        fig.show()

    def plot_MAs(self):
        self.stock_df['MA20'] = self.stock_df["Adj Close"].rolling(20).mean()
        self.stock_df['MA100'] = self.stock_df["Adj Close"].rolling(100).mean()
        ma20 = go.Scatter(x=self.stock_df['MA20'].index, y=self.stock_df['MA20'],
                          line=dict(color='orange', width=1), name="MA20")
        ma100 = go.Scatter(x=self.stock_df['MA100'].index, y=self.stock_df['MA100'],
                           line=dict(color='green', width=1), name="MA100")
        price = go.Scatter(x=self.stock_df.index, y=self.stock_df['Adj Close'],
                           line=dict(color='blue', width=1), name="Price")

        fig = go.Figure()
        fig.add_trace(ma20)
        fig.add_trace(ma100)
        fig.add_trace(price)

        fig.update_xaxes(
            rangeslider_visible=True, title='Zoom on Dates Using Slider')
        fig.update_yaxes(title="Stock Price")
        fig.show()

    def plot_EMA(self):
        # A EMA can be used to reduce the lag by putting more emphasis on recent price data
        self.stock_df['MA20'] = self.stock_df["Adj Close"].rolling(20).mean()
        self.stock_df['EMA20'] = self.stock_df["Adj Close"].ewm(span=20, adjust=False).mean()

        ema20 = go.Scatter(x=self.stock_df['EMA20'].index, y=self.stock_df['EMA20'],
                           line=dict(color='green', width=1), name="EMA20")
        ma20 = go.Scatter(x=self.stock_df['MA20'].index, y=self.stock_df['MA20'],
                          line=dict(color='orange', width=1), name="MA20")
        price = go.Scatter(x=self.stock_df.index, y=self.stock_df['Adj Close'],
                           line=dict(color='blue', width=1), name="Price")
        fig = go.Figure()
        fig.add_trace(ma20)
        fig.add_trace(ema20)
        fig.add_trace(price)
        fig.update_xaxes(
            rangeslider_visible=True, title='Zoom on Dates Using Slider')
        fig.update_yaxes(title="Stock Price (USD)")
        fig.show()

    def plot_death_Golden_crosses_US(self):
        '''
          When a Death Cross occurs, that is a sign that a major sell off will occur. A Death Cross is said to occur typically
          when the 50 day moving average falls below a 200 day. A Golden Cross accures when the short term average crosses
          the long term again moving higher.
          '''
        gspc_df = yf.download(tickers='^gspc', period='max', interval='1d')
        gspc_ma50 = gspc_df['Adj Close'].rolling(window=50).mean()
        gspc_ma200 = gspc_df['Adj Close'].rolling(window=200).mean()
        ma50 = go.Scatter(x=gspc_ma50.index, y=gspc_ma50,
                          line=dict(color='orange', width=1), name="MA50")
        ma200 = go.Scatter(x=gspc_ma200.index, y=gspc_ma200,
                           line=dict(color='green', width=1), name="MA200")
        gspc_prc = go.Scatter(x=gspc_df.index, y=gspc_df['Adj Close'],
                              line=dict(color='blue', width=1), name="Price")

        fig = go.Figure()
        fig.add_trace(ma50)
        fig.add_trace(ma200)
        fig.add_trace(gspc_prc)

        fig.update_xaxes(
            rangeslider_visible=True, title='Zoom on Dates Using Slider')
        fig.update_yaxes(title="Stock Price")
        fig.show()


TA = TechnicalAnalysis("MSFT", "10y", "1d")
print(TA.stock_df)
# TA.plot_death_Golden_crosses_US()
