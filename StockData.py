import matplotlib
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import mplfinance as mpf
import yfinance as yf


class StockData:
    def __init__(self, ticker, starting_year, starting_month, starting_day, ending_year, ending_month,
                 ending_day):
        self.ticker = ticker
        self.starting_year = starting_year
        self.starting_month = starting_month
        self.starting_day = starting_day
        self.ending_year = ending_year
        self.ending_month = ending_month
        self.ending_day = ending_day
        self.stock = yf.Ticker(self.ticker)
        self.starting_date = f"{self.starting_year}-{self.starting_month}-{self.starting_day}"
        self.ending_date = f"{self.ending_year}-{self.ending_month}-{self.ending_day}"
        self.stock
    def get_ticker(self):
        return self.ticker

    def get_start_date(self):
        return self.starting_date

    def get_end_date(self):
        return self.ending_date

    def save_to_csv_from_yahoo(self):
        """
        :param
        :return saves the stock data to a csv file
         """
        start_date = dt.datetime(self.starting_year, self.starting_month, self.starting_day)
        ending_date = dt.datetime(self.ending_year, self.ending_month, self.ending_day)

        df = web.DataReader(self.ticker, 'yahoo', start_date, ending_date)
        df.to_csv(f"{self.ticker}.csv")  # NEED TO BE CHANGED !!!!!

    def get_dataframe_from_csv(self):
        """
        :precondition .csv file of the ticker exists
        :param
        :return dataframe with the data for this ticker
        :exception .csv file not found!
         """
        try:
            df = pd.read_csv(f"{self.ticker}.csv")
        except FileNotFoundError:
            print("File of the Ticker Not Found! \n ")
        else:
            return df

    def add_daily_return_to_dataframe(self, df):
        """
        :precondition a valid dataframe
        :param dataframe
        :return dataframe with the data for this ticker + a column that have daily returns + saves changes to
        the csv file

        """
        df['daily_return'] = (df['Adj Close'] / df['Adj Close'].shift(1) - 1)
        # df['Adj Close'].shift(1)-1 is the values from previous day
        # df.to_csv(f"{self.ticker}.csv")
        return df

    def get_return_in_defined_time_from_csv(self):
        """
        :param Nothing
        :return a float value with total return in the specified period

        """
        df = self.add_daily_return_to_dataframe(self.get_dataframe_from_csv())
        start_date = f"{self.starting_year}-{self.starting_month}-{self.starting_day}"
        ending_date = f"{self.ending_year}-{self.ending_month}-{self.ending_day}"

        df['Date'] = pd.to_datetime(df['Date'])
        mask = (df['Date'] >= start_date) & (df['Date'] <= ending_date)
        daily_returns_mean = df.loc[mask]['daily_return'].mean()
        df2 = df.loc[mask]
        number_of_days = df2.shape[0]  # Return a tuple representing the dimensionality of the DataFrame
        return number_of_days * daily_returns_mean

    def mplfinance_plot(self, chart_type):
        """
        :precondition a valid csv file
        :param chart_type
        :return Multiple Charts

        """
        start_date = f"{self.starting_year}-{self.starting_month}-{self.starting_day}"
        ending_date = f"{self.ending_year}-{self.ending_month}-{self.ending_day}"
        df = self.get_dataframe_from_csv()
        df.index = pd.DatetimeIndex(df['Date'])
        df_sub = df.loc[start_date:ending_date]
        # A candlestick chart demonstrates the daily open, high, low and closing price of a stock
        mpf.plot(df_sub, type='candle')

        # Plot price changes
        mpf.plot(df_sub, type='line')

        # Moving averages provide trend information (Average of previous 4 observations)
        mpf.plot(df_sub, type='ohlc', mav=4)

        # Define a built in style
        s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 12})
        # Pass in the defined style to the whole canvas
        fig = mpf.figure(figsize=(12, 8), style=s)
        # Candle stick chart subplot
        ax = fig.add_subplot(2, 1, 1)
        # Volume chart subplot
        av = fig.add_subplot(2, 1, 2, sharex=ax)
        # You can plot multiple MAVs, volume, non-trading days
        mpf.plot(df_sub, type=chart_type, mav=(3, 5, 7), ax=ax, volume=av, show_nontrading=False)
        mpf.show

    def price_plot(self):
        """
        :precondition a valid csv file
        :param None
        :return Chart for specified period

        """
        start_date = f"{self.starting_year}-{self.starting_month}-{self.starting_day}"
        ending_date = f"{self.ending_year}-{self.ending_month}-{self.ending_day}"
        df = self.get_dataframe_from_csv()
        df.index = pd.DatetimeIndex(df['Date'])
        df_sub = df.loc[start_date:ending_date]
        df_np = df_sub.to_numpy()
        np_adj_close = df_np[:, 5]
        date_array = df_np[:, 1]
        fig = plt.figure(figsize=(16, 10), dpi=100)
        axes = fig.add_subplot(111)
        axes.plot(date_array, np_adj_close, color='navy')
        axes.xaxis.set_major_locator(plt.MaxNLocator(8))
        axes.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        axes.set_facecolor('#FAEBD7')
        plt.show()

    def get_return_in_defined_time(self):
        df = self.add_daily_return_to_dataframe(self.get_all_data_in_period())
        daily_returns_mean = df['daily_return'].mean()
        number_of_days = df.shape[0]  # Return a tuple representing the dimensionality of the DataFrame
        return number_of_days * daily_returns_mean
        return

    def print_info_on_stock(self):
        print(self.stock.info)

    def get_price_now(self):
        js = self.stock.get_info()
        return js['currentPrice']

    def get_all_data_in_period(self):
        return pd.DataFrame(yf.download(self.ticker, start=self.starting_date, end=self.ending_date))

    def get_all_data_max_period(self):
        return pd.DataFrame(yf.download(self.ticker, period='max'))

    def save_dataframe_to_csv(self, df):
        df.to_csv(f"{self.ticker}.csv")

    def get_adj_close_in_period(self):
        df = pd.DataFrame(yf.download(self.ticker, start=self.starting_date, end=self.ending_date))
        ds = df["Adj Close"]
        #ds = ds.rename(columns={'': 'Adj Close'})
        return ds

    def get_adj_close_max_period(self):
        df = pd.DataFrame(yf.download(self.ticker, period='max'))
        ds = df["Adj Close"]
        return ds

    def get_daily_return_in_period(self):
        df = pd.DataFrame(yf.download(self.ticker, start=self.starting_date, end=self.ending_date))
        df['daily_return'] = (df['Adj Close'] / df['Adj Close'].shift(1) - 1)
        ds = df['daily_return']
        return ds

    def get_roi_max_period(self):
        adj_close = self.get_adj_close_max_period()
        roi = ((adj_close.iloc[-1] - adj_close.iloc[0]) / adj_close.iloc[0])
        return roi

    def get_roi_in_period(self):
        adj_close = self.get_adj_close_in_period()
        roi = ((adj_close.iloc[-1] - adj_close.iloc[0]) / adj_close.iloc[0])
        return roi

    def get_mean_max_period(self):
        adj_close = self.get_adj_close_max_period()
        return adj_close.mean()

    def get_mean_in_period(self):
        adj_close = self.get_adj_close_in_period()
        return adj_close.mean()

    def get_std_max_period(self):
        adj_close = self.get_adj_close_max_period()
        return adj_close.std()

    def get_std_in_period(self):
        adj_close = self.get_adj_close_in_period()
        return adj_close.std()

    def get_coefficient_max_period(self):
        mean = self.get_mean_max_period()
        sd = self.get_std_max_period()
        return sd / mean

    def get_coefficient_in_period(self):
        mean = self.get_mean_in_period()
        sd = self.get_std_in_period()
        return sd / mean

    def get_balance_sheet_dataframe_yearly(self):
        df = pd.DataFrame(self.stock.get_balance_sheet())
        return df

    def get_balance_sheet_dataframe_quarterly(self):
        df = pd.DataFrame(self.stock.quarterly_balance_sheet)
        return df

    def get_cashflow_dataframe_yearly(self):
        df = pd.DataFrame(self.stock.cashflow)
        return df

    def get_cashflow_dataframe_quarterly(self):
        df = pd.DataFrame(self.stock.quarterly_cashflow)
        return df

    def get_earnings_dataframe_yearly(self):
        df = pd.DataFrame(self.stock.earnings)
        return df

    def get_earnings_dataframe_quarterly(self):
        df = pd.DataFrame(self.stock.quarterly_earnings)
        return df

    def get_dividends(self):
        df = pd.DataFrame(self.stock.get_dividends())
        return df

    def get_splits(self):
        df = pd.DataFrame(self.stock.splits)
        return df

    def get_major_holders(self):
        df = pd.DataFrame(self.stock.major_holders)
        return df

    def get_institutional_holders(self):
        df = pd.DataFrame(self.stock.institutional_holders)
        return df

    def get_recommendations(self):
        return self.stock.recommendations

    def get_news(self):

        return self.stock.news

# ---------------------------------------------------------------------------------------------------------------

# TEST CODE:

# ---------------------------------------------------------------------------------------------------------------

# amazon = StockData("AMZN", 2020, 1, 1, 2021, 1, 1)
# # amazon.save_to_csv_from_yahoo()
# # df_amazon = amazon.get_dataframe_from_csv()
# # df_amazon = amazon.add_daily_return_to_dataframe(df_amazon)
# # print(df_amazon)
# # print(amazon.get_return_in_defined_time())
# # amazon.mplfinance_plot("ohlc")
# # amazon.price_plot()
#
# print(amazon.get_daily_return_in_period())
