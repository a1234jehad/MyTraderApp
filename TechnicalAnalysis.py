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
from ta.momentum import StochasticOscillator,rsi


class TechnicalAnalysis:
    def __init__(self, Ticker, period, intervals):
        # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        self.stock_df = yf.download(tickers=Ticker, period=period, interval=intervals)
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
        fig.show()

    def plot_EMA(self):
        # A EMA can be used to reduce the lag by putting more emphasis on recent price data
        close = self.stock_df['Adj Close']
        high = self.stock_df['High']
        low = self.stock_df['Low']
        open = self.stock_df['Open']

        self.stock_df['MA20'] = self.stock_df["Adj Close"].rolling(20).mean()
        self.stock_df['EMA20'] = self.stock_df["Adj Close"].ewm(span=20, adjust=False).mean()
        self.stock_df['EMA50'] = self.stock_df["Adj Close"].ewm(span=20, adjust=False).mean()
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

        fig.update_yaxes(title="Stock Price (USD)")
        if (self.ticker[-2:] == "SR"):
            fig.update_xaxes(rangebreaks=[dict(bounds=["fri", "sun"])])
        else:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

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
        fig.update_layout(title=self.ticker)
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
        self.stock_df['RSI'] = rsi(self.stock_df['Close'],window=14)
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
        fig.update_layout(title=self.ticker)
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
        self.stock_df['Mean'] =  self.stock_df['Close'].rolling(window=20).mean()
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
        fig.update_layout(title=self.ticker)

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
                          showlegend=True)
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

TA = TechnicalAnalysis("AMD", "1y", "1d")
# print(TA.stock_df)
# TA.plot_macd()
# print(TA.stock_df)
# TA.plot_death_Golden_crosses_US()
#TA.plot_RSI()
TA.plot_Bollinger_bands()
TA.plot_Ichimoku()