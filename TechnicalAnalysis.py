import datetime
from time import strftime

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
from ta.momentum import StochasticOscillator, rsi
import talib as ta


class TechnicalAnalysis:
    def __init__(self, Ticker, period, intervals):
        # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        self.stock_df = yf.download(tickers=Ticker, period=period, interval=intervals)
        self.stock_df = self.stock_df[self.stock_df["Volume"]>0]
        #self.stock_df = self.stock_df.drop("2022-01-16")
        self.ticker = Ticker
        self.intv = intervals

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
                          title=self.ticker,
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
        fig.update_layout(title=f"MA {self.ticker}")
        fig.show()

    def plot_EMA(self):
        # A EMA can be used to reduce the lag by putting more emphasis on recent price data
        close = self.stock_df['Adj Close']
        high = self.stock_df['High']
        low = self.stock_df['Low']
        open = self.stock_df['Open']

        self.stock_df['MA20'] = self.stock_df["Adj Close"].rolling(20).mean()
        self.stock_df['EMA20'] = self.stock_df["Adj Close"].ewm(span=20, adjust=False).mean()
        self.stock_df['EMA50'] = self.stock_df["Adj Close"].ewm(span=50, adjust=False).mean()
        ema20 = go.Scatter(x=self.stock_df['EMA20'].index, y=self.stock_df['EMA20'],
                           line=dict(color='green', width=1), name="EMA20")
        ema50 = go.Scatter(x=self.stock_df['EMA50'].index, y=self.stock_df['EMA50'],
                           line=dict(color='red', width=1), name="EMA50")
        ma20 = go.Scatter(x=self.stock_df['MA20'].index, y=self.stock_df['MA20'],
                          line=dict(color='orange', width=1), name="MA20")
        # price = go.Scatter(x=self.stock_df.index, y=self.stock_df['Adj Close'],
        #                    line=dict(color='blue', width=1), name="Price")
        candles = go.Candlestick(x=self.stock_df.index, open=open, high=high,
                                 low=low, close=close, name="Candles")
        fig = go.Figure()
        fig.add_trace(ma20)
        fig.add_trace(ema20)
        fig.add_trace(ema50)
        fig.add_trace(candles)

        fig.update_yaxes(title="Stock Price")
        if (self.ticker[-2:] == "SR"):
            fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"])])
        else:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_layout(title=f"EMA {self.ticker}")

        fig.show()

    def plot_death_Golden_crosses(self):
        '''
          When a Death Cross occurs, that is a sign that a major sell off will occur. A Death Cross is said to occur typically
          when the 50 day moving average falls below a 200 day. A Golden Cross accures when the short term average crosses
          the long term again moving higher.
          '''
        gspc_df = self.stock_df
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
        fig.update_layout(title=f"Death & Golden Crosses {self.ticker}")
        if (self.ticker[-2:] == "SR"):
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"]), dict(bounds=[15, 10], pattern="hour")])

        else:
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
        fig.show()

    def plot_macd(self):
        '''
        Moving Average Convergence & Divergence
        The MACD helps us to see buy & sell signals. It shows the difference between 2 moving averages.

        While these signals are derived from moving averages they occur much more quickly then with moving averages.
        It is important to know that since the signals occur earlier that they are also more risky.

        When the signal line crosses the MACD line moving upwards this is bullish and vice versa. The slope of the angle tells
        you how strong the trend is.
        '''
        candle = go.Candlestick(x=self.stock_df.index, open=self.stock_df['Open'],
                                high=self.stock_df['High'], low=self.stock_df['Low'],
                                close=self.stock_df['Close'])
        self.stock_df['MA12'] = self.stock_df['Adj Close'].ewm(span=12, adjust=False).mean()
        self.stock_df['MA26'] = self.stock_df['Adj Close'].ewm(span=26, adjust=False).mean()
        macd = MACD(close=self.stock_df['Close'],
                    window_slow=26,
                    window_fast=12,
                    window_sign=9)  # Calculate the MACD

        # A signal line uses a period of 9 and when it crosses the other moving
        # average it is a sign to buy or sell
        # A Stochastic (stuh ka stuhk) Oscillator is also plotted because it
        # gives us a signal of an upcoming trend reversal
        # Values range from 0 to 100 and values over 80 are considered to be
        # overbought while those under 20 are considered oversold
        # We calculate normally over a 14 day period
        # We are smoothing price data
        sto_os = StochasticOscillator(high=self.stock_df['High'],
                                      close=self.stock_df['Close'],
                                      low=self.stock_df['Low'],
                                      window=14,
                                      smooth_window=3)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        fig.add_trace(candle, row=1, col=1)
        fig.add_trace(go.Scatter(x=self.stock_df.index,
                                 y=macd.macd(),
                                 line=dict(color='blue', width=2)
                                 ), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.stock_df.index,
                                 y=macd.macd_signal(),
                                 line=dict(color='orange', width=2)
                                 ), row=2, col=1)

        # The histogram shows the difference between the MACD and signal line
        # When the MACD line is below the signal that is a negative value and vice versa
        fig.add_trace(go.Bar(x=self.stock_df.index,
                             y=macd.macd_diff()
                             ), row=2, col=1)

        # The MacD line is a calculation made by subtracting the 26 period
        # EMA from the 12 period EMA
        fig.add_trace(go.Scatter(x=self.stock_df.index,
                                 y=macd.macd(),
                                 line=dict(color='blue', width=2)
                                 ), row=2, col=1)

        # The signal is calculated by taking the average of the last 9 values
        # of the MACD line (The signal line is a slower more smoothed out version of
        # the MACD line)
        fig.add_trace(go.Scatter(x=self.stock_df.index,
                                 y=macd.macd_signal(),
                                 line=dict(color='orange', width=2)
                                 ), row=2, col=1)

        # Plot Stochastics
        # This is the faster of the 2 lines called the K line
        fig.add_trace(go.Scatter(x=self.stock_df.index,
                                 y=sto_os.stoch(),
                                 line=dict(color='blue', width=2)
                                 ), row=3, col=1)
        # This line is slower and is known as the D line and it is an average of the K line
        fig.add_trace(go.Scatter(x=self.stock_df.index,
                                 y=sto_os.stoch_signal(),
                                 line=dict(color='orange', width=2)
                                 ), row=3, col=1)

        # Draw 20 and 80 lines
        fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="green", row=3, col=1)

        # Add volume
        fig.add_trace(go.Bar(x=self.stock_df.index,
                             y=self.stock_df['Volume']
                             ), row=4, col=1)

        # Update titles
        fig.update_layout(title=f"MACD {self.ticker}")
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="StoOs", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)

        fig.update_layout(height=900, width=1200,
                          showlegend=False,
                          xaxis_rangeslider_visible=False)
        if (self.ticker[-2:] == "SR"):
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"]), dict(bounds=[15, 10], pattern="hour")])

        else:
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
        # Don't buy based on this chart during downtrends and don't sell during uptrends!

        fig.show()

    def plot_RSI(self):
        """
        The RSI is used to determine if a security is overbought or oversold. With them you can take advantage of
         potential changes in trend. The 2 most commonly used oscillators are the RSI and Stochastic RSI.
        """
        self.stock_df['RSI'] = rsi(self.stock_df['Close'], window=14)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.01,
                            row_heights=[0.7, 0.3])

        candle = go.Candlestick(x=self.stock_df.index, open=self.stock_df['Open'],
                                high=self.stock_df['High'], low=self.stock_df['Low'],
                                close=self.stock_df['Close'], name='Candlestick')
        rsi_ = go.Scatter(x=self.stock_df.index, y=self.stock_df['RSI'],
                          line=dict(color='blue', width=2))
        fig.add_trace(candle, row=1, col=1)
        fig.add_trace(rsi_, row=2, col=1)
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(title=f"RSI {self.ticker}")
        fig.update_layout(height=900, width=1200,
                          showlegend=False,
                          xaxis_rangeslider_visible=False)
        if (self.ticker[-2:] == "SR"):
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"]), dict(bounds=[15, 10], pattern="hour")])

        else:
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])

        fig.show()

    def plot_Bollinger_bands(self):
        """
        Bollinger Bands plot 2 lines using a moving average and the standard deviation defines how far apart the
        lines are. They also are used to define if prices are to high or low. When bands tighten it is believed
        a sharp price move in some direction. Prices tend to bounce off of the bands which provides potential
        market actions.

        A strong trend should be noted if the price moves outside the band. If prices go over the resistance line it is
        in overbought territory and if it breaks through support it is a sign of an oversold position.
        """
        self.stock_df['Mean'] = self.stock_df['Close'].rolling(window=20).mean()
        self.stock_df['SD'] = self.stock_df['Close'].rolling(window=20).std()
        self.stock_df['BB_Hi'] = self.stock_df['Mean'] + (2 * self.stock_df['SD'])
        self.stock_df['BB_Low'] = self.stock_df['Mean'] - (2 * self.stock_df['SD'])

        fig = go.Figure()
        candle = go.Candlestick(x=self.stock_df.index, open=self.stock_df['Open'],
                                high=self.stock_df['High'], low=self.stock_df['Low'],
                                close=self.stock_df['Close'], name='Candlestick')
        bb_hi = go.Scatter(x=self.stock_df.index, y=self.stock_df['BB_Hi'],
                           line=dict(color='green', width=1), name="BB Hi")

        bb_low = go.Scatter(x=self.stock_df.index, y=self.stock_df['BB_Low'],
                            line=dict(color='orange', width=1), name="BB Low")

        # Add plots to the figure
        fig.add_trace(candle)
        fig.add_trace(bb_hi)
        fig.add_trace(bb_low)

        # Add title
        fig.update_layout(title=f"Bollinger_bands {self.ticker}")

        if (self.ticker[-2:] == "SR"):
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"]), dict(bounds=[15, 10], pattern="hour")])

        else:
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])

        fig.show()

    def plot_Ichimoku(self):
        """
        The Ichimoku (One Look) is considered an all in one indicator. It provides information on momentum, support and
        resistance. It is made up of 5 lines. If you are a short term trader you create 1 minute or 6 hour. Long term
        traders focus on day or weekly data.

        Conversion Line (Tenkan-sen) : Represents support, resistance and reversals. Used to measure short term trends.
        Baseline (Kijun-sen) : Represents support, resistance and confirms trend changes. Allows you to evaluate the
        strength of medium term trends. Called the baseline because it lags the price.
        Leading Span A (Senkou A) : Used to identify future areas of support and resistance
        Leading Span B (Senkou B) : Other line used to identify suture support and resistance
        Lagging Span (Chikou) : Shows possible support and resistance. It is used to confirm signals obtained from other
        lines.
        Cloud (Kumo) : Space between Span A and B. Represents the divergence in price evolution.

         What the Lines Mean??

         Lagging Span : When above the price it is bullish and when below bearish. It is used
         with other indicators because it is mainly a filter.

         Baseline : When below price this is considered support.
         When above price this is considered resistance. We are in an uptrend when the slope increases and vice
         versa. The slope of the curve tells us the strength of the trend.

         Conversion : We focus on its position
         versus the Baseline. When the Conversion crosses above the Baseline we are in an upward trend and vice
         versa. This is considered a strong indicator when above the Cloud and weak when below.

         Cloud : The thicker
         the Cloud, the stronger the trend and vice versa. When the Spans cross many times we are in a range. When
         they cross this is a sign of a reversal of trend.
        """

        def get_fill_color(label):
            if label >= 1:
                return 'rgba(0,250,0,0.4)'
            else:
                return 'rgba(250,0,0,0.4)'

        df = self.stock_df

        # Conversion
        hi_val = df['High'].rolling(window=9).max()
        low_val = df['Low'].rolling(window=9).min()
        df['Conversion'] = (hi_val + low_val) / 2

        # Baseline
        hi_val2 = df['High'].rolling(window=26).max()
        low_val2 = df['Low'].rolling(window=26).min()
        df['Baseline'] = (hi_val2 + low_val2) / 2

        # Spans
        df['SpanA'] = ((df['Conversion'] + df['Baseline']) / 2).shift(26)
        hi_val3 = df['High'].rolling(window=52).max()
        low_val3 = df['Low'].rolling(window=52).min()
        df['SpanB'] = ((hi_val3 + low_val3) / 2).shift(26)
        df['Lagging'] = df['Close'].shift(-26)

        candle = go.Candlestick(x=df.index, open=df['Open'],
                                high=df['High'], low=df['Low'],
                                close=df['Close'], name='Candlestick')

        df1 = df.copy()
        fig = go.Figure()
        df['label'] = np.where(df['SpanA'] > df['SpanB'], 1, 0)

        df['group'] = df['label'].ne(df['label'].shift()).cumsum()

        df = df.groupby('group')

        dfs = []
        for name, data in df:
            dfs.append(data)

        for df in dfs:
            fig.add_traces(go.Scatter(x=df.index, y=df.SpanA,
                                      line=dict(color='rgba(0,0,0,0)')))

            fig.add_traces(go.Scatter(x=df.index, y=df.SpanB,
                                      line=dict(color='rgba(0,0,0,0)'),
                                      fill='tonexty',
                                      fillcolor=get_fill_color(df['label'].iloc[0])))

        baseline = go.Scatter(x=df1.index, y=df1['Baseline'],
                              line=dict(color='pink', width=2), name="Baseline")

        conversion = go.Scatter(x=df1.index, y=df1['Conversion'],
                                line=dict(color='black', width=1), name="Conversion")

        lagging = go.Scatter(x=df1.index, y=df1['Lagging'],
                             line=dict(color='purple', width=2, dash='dot'), name="Lagging")

        span_a = go.Scatter(x=df1.index, y=df1['SpanA'],
                            line=dict(color='green', width=2, dash='dot'), name="Span A")

        span_b = go.Scatter(x=df1.index, y=df1['SpanB'],
                            line=dict(color='red', width=1, dash='dot'), name="Span B")

        # Add plots to the figure
        fig.add_trace(candle)
        fig.add_trace(baseline)
        fig.add_trace(conversion)
        fig.add_trace(lagging)
        fig.add_trace(span_a)
        fig.add_trace(span_b)

        fig.update_layout(height=900, width=1000,
                          showlegend=True, title=f"Ichimoku {self.ticker}")
        if (self.ticker[-2:] == "SR"):
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"]), dict(bounds=[15, 10], pattern="hour")])

        else:
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])

        fig.show()

    def red_blue_white_strategy(self):
        # back testing included
        df = self.stock_df
        emas_used = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]
        fig = go.Figure()
        price = go.Scatter(x=self.stock_df.index, y=self.stock_df['Adj Close'],
                           line=dict(color='green', width=1), name="Price")
        fig.add_trace(price)
        for ema in emas_used:
            df[f"Ema_{ema}"] = round(df.iloc[:, 4].ewm(span=ema, adjust=False).mean(), 2)
            if ema < 30:
                fig.add_trace(go.Scatter(x=self.stock_df[f"Ema_{ema}"].index, y=self.stock_df[f"Ema_{ema}"],
                                         line=dict(color='red', width=1), name=f"EMA{ema}"))
            else:
                fig.add_trace(go.Scatter(x=self.stock_df[f"Ema_{ema}"].index, y=self.stock_df[f"Ema_{ema}"],
                                         line=dict(color='blue', width=1), name=f"EMA{ema}"))
        fig.update_yaxes(title="Stock Price")
        if (self.ticker[-2:] == "SR"):
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"]), dict(bounds=[15, 10], pattern="hour")])

        else:
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
        fig.update_layout(title=f"red_blue_white_strategy {self.ticker}")

        fig.show()

        df = df.iloc[60:]
        pos = 0
        num = 0
        percentchange = []
        for i in df.index:
            cmin = min(df["Ema_3"][i], df["Ema_5"][i], df["Ema_8"][i], df["Ema_10"][i], df["Ema_12"][i],
                       df["Ema_15"][i], )
            cmax = max(df["Ema_30"][i], df["Ema_35"][i], df["Ema_40"][i], df["Ema_45"][i], df["Ema_50"][i],
                       df["Ema_60"][i], )
            close = df["Adj Close"][i]

            if (cmin > cmax):
                #print("Red White Blue")
                if (pos == 0):
                    bp = close
                    pos = 1
                    #print("Buying now at " + str(bp))


            elif (cmin < cmax):
                #print("Blue White Red")
                if (pos == 1):
                    pos = 0
                    sp = close
                    #print("Selling now at " + str(sp))
                    pc = (sp / bp - 1) * 100
                    percentchange.append(pc)
            if (num == df["Adj Close"].count() - 1 and pos == 1):
                pos = 0
                sp = close
                #print("Selling now at " + str(sp))
                pc = (sp / bp - 1) * 100
                percentchange.append(pc)

            num += 1

        #print(percentchange)

        gains = 0
        ng = 0
        losses = 0
        nl = 0
        totalR = 1

        for i in percentchange:
            if (i > 0):
                gains += i
                ng += 1
            else:
                losses += i
                nl += 1
            totalR = totalR * ((i / 100) + 1)

        totalR = round((totalR - 1) * 100, 2)

        if (ng > 0):
            avgGain = gains / ng
            maxR = str(max(percentchange))
        else:
            avgGain = 0
            maxR = "undefined"

        if (nl > 0):
            avgLoss = losses / nl
            maxL = str(min(percentchange))
            ratio = str(-avgGain / avgLoss)
        else:
            avgLoss = 0
            maxL = "undefined"
            ratio = "inf"

        if (ng > 0 or nl > 0):
            battingAvg = ng / (ng + nl)
        else:
            battingAvg = 0

        print()
        print("Results for " + self.ticker + " going back to " + str(df.index[0]) + ", Sample size: " + str(
            ng + nl) + " trades")
        print("EMAs used: " + str(emas_used))
        print("Batting Avg: " + str(battingAvg))
        print("Gain/loss ratio: " + ratio)
        print("Average Gain: " + str(avgGain))
        print("Average Loss: " + str(avgLoss))
        print("Max Return: " + maxR)
        print("Max Loss: " + maxL)
        print("Total return over " + str(ng + nl) + " trades: " + str(totalR) + "%")

    def Green_Line_Breakout_strategy(self):
        # GLV = green line value
        # ATH then rest for 3 months
        df = self.stock_df
        df.drop(df[df["Volume"] < 1000].index, inplace=True)
        dfmonth = df.groupby(pd.Grouper(freq="M"))["High"].max()
        now = datetime.datetime.now()
        glDate = 0
        lastGLV = 0
        currentDate = ""
        curentGLV = 0
        glvs = []
        for index, value in dfmonth.items():
            if value > curentGLV:
                curentGLV = value
                currentDate = index
                counter = 0
            if value < curentGLV:
                counter = counter + 1

                if counter == 3 and ((index.month != now.month) or (index.year != now.year)):
                    if curentGLV != lastGLV:
                        print(f"{curentGLV} on {glDate}")
                        if glDate != 0:
                            glvs.append(curentGLV)
                    glDate = currentDate
                    lastGLV = curentGLV
                    counter = 0

        if lastGLV == 0:
            message = self.ticker + " has not formed a green line yet"
        else:
            message = ("Last Green Line: " + str(lastGLV) + " on " + str(glDate))
        close = self.stock_df['Adj Close']
        high = self.stock_df['High']
        low = self.stock_df['Low']
        open = self.stock_df['Open']
        candles = go.Candlestick(x=self.stock_df.index, open=open, high=high,
                                 low=low, close=close, name="Candles")
        fig = go.Figure()
        fig.add_trace(candles)
        fig.update_yaxes(title="Stock Price")
        if (self.ticker[-2:] == "SR"):
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"]), dict(bounds=[15, 10], pattern="hour")])

        else:
            if self.intv == '1d':
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            else:
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
        fig.update_layout(title=f"GLV {self.ticker}")
        for x in glvs:
            fig.add_hline(y=x)
        fig.show()

    def plot_resistance_pivot(self):
        df = self.stock_df
        df["High"].plot(label='High')

        pivots = []
        dates = []
        counter = 0
        lastPivot = 0

        Range = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dateRange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in df.index:
            current_max = max(Range, default=0)
            value = round(df['High'][i], 2)
            Range = Range[1:9]
            Range.append(value)
            dateRange = dateRange[1:9]
            dateRange.append(i)
            if current_max == max(Range, default=0):
                counter += 1
            else:
                counter = 0
            if counter == 5:
                lastPivot = current_max
                dateloc = Range.index(lastPivot)
                lastDate = dateRange[dateloc]

                pivots.append(lastPivot)
                dates.append(lastDate)
        timeD = datetime.timedelta(days=30)

        for i in range(len(pivots)):
            plt.plot_date([dates[i], dates[i] + timeD], [pivots[i], pivots[i]], linestyle="-", linewidth=2, marker=",")
        plt.show()

    def rm_claculate(self, AvgGain, AvgLoss):
        # AvgGain = 15
        # AvgLoss = 5
        smaUsed = [50, 200]
        emaUsed = [21]
        df = self.stock_df
        close = self.stock_df['Adj Close'][-1]
        maxStop = close * ((100 - AvgLoss) / 100)
        Target1R = round(close * ((100 + AvgGain) / 100), 2)
        Target2R = round(close * (((100 + (2 * AvgGain)) / 100)), 2)
        Target3R = round(close * (((100 + (3 * AvgGain)) / 100)), 2)
        for x in smaUsed:
            sma = x
            df["SMA_" + str(sma)] = round(df.iloc[:, 4].rolling(window=sma).mean(), 2)
        for x in emaUsed:
            ema = x
            df['EMA_' + str(ema)] = round(df.iloc[:, 4].ewm(span=ema, adjust=False).mean(), 2)
        sma50 = round(df["SMA_50"][-1], 2)
        sma200 = round(df["SMA_200"][-1], 2)
        ema21 = round(df["EMA_21"][-1], 2)
        low5 = round(min(df["Low"].tail(5)), 2)
        pf50 = round(((close / sma50) - 1) * 100, 2)
        check50 = df["SMA_50"][-1] > maxStop
        pf200 = round(((close / sma200) - 1) * 100, 2)
        check200 = ((close / df["SMA_200"][-1]) - 1) * 100 > 100
        pf21 = round(((close / ema21) - 1) * 100, 2)
        check21 = df["EMA_21"][-1] > maxStop
        pfl = round(((close / low5) - 1) * 100, 2)
        checkl = low5 > maxStop
        print()
        print("Current Stock: " + self.ticker + " Price: " + str(round(close, 2)))
        print("21 EMA: " + str(ema21) + " | 50 SMA: " + str(sma50) + " | 200 SMA: " + str(
            sma200) + " | 5 day Low: " + str(
            low5))
        print("-------------------------------------------------")
        print("Max Stop: " + str(round(maxStop, 2)))
        print("Price Targets:")
        print("1R: " + str(Target1R))
        print("2R: " + str(Target2R))
        print("3R: " + str(Target3R))
        print("From 5 Day Low " + str(pfl) + "% -Within Max Stop: " + str(checkl))
        print("From 21 day EMA " + str(pf21) + "% -Within Max Stop: " + str(check21))
        print("From 50 day SMA " + str(pf50) + "% -Within Max Stop: " + str(check50))
        print("From 200 Day SMA " + str(pf200) + "% -In Danger Zone (Over 100% from 200 SMA): " + str(check200))
        print()

    def look_for_volume_stocks(self):
        df_US = pd.read_csv("US.csv")
        df_SA = pd.read_csv("Tasi.csv")
        df = pd.DataFrame()
        df['Tickers'] = pd.DataFrame(df_US['Ticker'].tolist() + df_SA['Ticker'].tolist())
        vol_stocks = []
        for stock in df['Tickers']:
            try:
                stock_info = yf.Ticker(stock)
                history = stock_info.history(period="5d")
                pre_avg_vol = history['Volume'].iloc[1:4:1].mean()
                vol = history['Volume'][-1]
                if vol > pre_avg_vol * 2:
                    vol_stocks.append(stock)
            except:
                pass
        print(vol_stocks)

    def pattern_detector(self):
        data = self.stock_df
        close = self.stock_df['Adj Close']
        high = self.stock_df['High']
        low = self.stock_df['Low']
        open = self.stock_df['Open']
        patterns = {
            "CDL2CROWS": "Two Crows",
            "CDL3BLACKCROWS": "Three Black Crows",
            "CDL3INSIDE": "Three Inside Up/Down",
            "CDL3LINESTRIKE": "Three-Line Strike",
            "CDL3OUTSIDE": "Three Outside Up/Down",
            "CDL3STARSINSOUTH": "Three Stars In The South",
            "CDL3WHITESOLDIERS": "Three Advancing White Soldiers",
            "CDLABANDONEDBABY": "Abandoned Baby",
            "CDLADVANCEBLOCK": "Advance Block",
            "CDLBELTHOLD": "Belt-hold",
            "CDLBREAKAWAY": "Breakaway",
            "CDLCLOSINGMARUBOZU": "Closing Marubozu",
            "CDLCONCEALBABYSWALL": "Concealing Baby Swallow",
            "CDLCOUNTERATTACK": "Counterattack",
            "CDLDARKCLOUDCOVER": "Dark Cloud Cover",
            "CDLDOJI": "Doji",
            "CDLDOJISTAR": "Doji Star",
            "CDLDRAGONFLYDOJI": "Dragonfly Doji",
            "CDLENGULFING": "Engulfing Pattern",
            "CDLEVENINGDOJISTAR": "Evening Doji Star",
            "CDLEVENINGSTAR": "Evening Star",
            "CDLGAPSIDESIDEWHITE": "Up/Down-gap side-by-side white lines",
            "CDLGRAVESTONEDOJI": "Gravestone Doji",
            "CDLHAMMER": "Hammer",
            "CDLHANGINGMAN": "Hanging Man",
            "CDLHARAMI": "Harami Pattern",
            "CDLHARAMICROSS": "Harami Cross Pattern",
            "CDLHIGHWAVE": "High-Wave Candle",
            "CDLHIKKAKE": "Hikkake Pattern",
            "CDLHIKKAKEMOD": "Modified Hikkake Pattern",
            "CDLHOMINGPIGEON": "Homing Pigeon",
            "CDLIDENTICAL3CROWS": "Identical Three Crows",
            "CDLINNECK": "In-Neck Pattern",
            "CDLINVERTEDHAMMER": "Inverted Hammer",
            "CDLKICKING": "Kicking",
            "CDLKICKINGBYLENGTH": "Kicking (bull/bear) determined by the longer marubozu",
            "CDLLADDERBOTTOM": "Ladder Bottom",
            "CDLLONGLEGGEDDOJI": "Long Legged Doji",
            "CDLLONGLINE": "Long Line Candle",
            "CDLMARUBOZU": "Marubozu",
            "CDLMATCHINGLOW": "Matching Low",
            "CDLMATHOLD": "Mat Hold",
            "CDLMORNINGDOJISTAR": "Morning Doji Star",
            "CDLMORNINGSTAR": "Morning Star",
            "CDLONNECK": "On-Neck Pattern",
            "CDLPIERCING": "Piercing Pattern",
            "CDLRICKSHAWMAN": "Rickshaw Man",
            "CDLRISEFALL3METHODS": "Rising/Falling Three Methods",
            "CDLSEPARATINGLINES": "Separating Lines",
            "CDLSHOOTINGSTAR": "Shooting Star",
            "CDLSHORTLINE": "Short Line Candle",
            "CDLSPINNINGTOP": "Spinning Top",
            "CDLSTALLEDPATTERN": "Stalled Pattern",
            "CDLSTICKSANDWICH": "Stick Sandwich",
            "CDLTAKURI": "Takuri (Dragonfly) Doji with very long lower shadow)",
            "CDLTASUKIGAP": "Tasuki Gap",
            "CDLTHRUSTING": "Thrusting Pattern",
            "CDLTRISTAR": "Tristar Pattern",
            "CDLUNIQUE3RIVER": "Unique 3 River",
            "CDLUPSIDEGAP2CROWS": "Upside Gap Two Crows",
            "CDLXSIDEGAP3METHODS": "Upside/Downside Gap Three Methods"
        }
        patterns_detect = {}
        for key, value in patterns.items():
            result = getattr(ta, f'{key}')(open, high, low, close)
            data[value] = result
            days = data[data[f'{value}'] !=0 ]
            if days.size > 0:
                last_day = days.index[-1]

                patterns_detect[f'{value}'] = last_day

        # patterns_description = {
        #
        # }
        pt = sorted([(value,key) for (key,value) in patterns_detect.items()],reverse=True)
        for key,value in pt:

            key = key.to_pydatetime()
            key_date = key.strftime('%Y-%m-%d')
            key_time = key.strftime('%H:%M:%S')

            # st_key =  datetime.strptime(str(key),'%Y-%m-%d %H:%M:%S')
            # msg += f'{st_key[0:10]} {value}\n'
            if key_time != "00:00:00":
                print(key_date,key_time,value)
            else:
                print(key_date,value)


TA = TechnicalAnalysis("3002.SR", "5y", "1d")
print(TA.stock_df.tail(50))

# TA.plot_MAs()
# TA.plot_EMA()
TA.plot_death_Golden_crosses()
TA.plot_macd()
TA.plot_RSI()
TA.plot_Bollinger_bands()
TA.plot_Ichimoku()
TA.red_blue_white_strategy()
TA.Green_Line_Breakout_strategy()
TA.plot_resistance_pivot()
TA.rm_claculate(15,5)
# TA.look_for_volume_stocks()
TA.pattern_detector()
