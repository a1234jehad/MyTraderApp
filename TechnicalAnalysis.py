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

        if(self.ticker[-2:] == "SR"):
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


TA = TechnicalAnalysis("7010.SR","1mo","5m")
print(TA.stock_df)
TA.plot_candlestick()