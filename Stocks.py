import csv
import matplotlib
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import mplfinance as mpf
from StockData import StockData
import yfinance as yf
from yahoo_fin import stock_info as si
from get_all_tickers import get_tickers as gt
from get_all_tickers.get_tickers import Region


class Stocks:
    def __init__(self, tickers_list, starting_year, starting_month, starting_day, ending_year, ending_month,
                 ending_day):
        self.tickers_list = tickers_list
        self.starting_year = starting_year
        self.starting_month = starting_month
        self.starting_day = starting_day
        self.ending_year = ending_year
        self.ending_month = ending_month
        self.ending_day = ending_day
        self.starting_date = f"{self.starting_year}-{self.starting_month}-{self.starting_day}"
        self.ending_date = f"{self.ending_year}-{self.ending_month}-{self.ending_day}"

    def download_multiple_stocks_to_csv(self):
        """
        :precondition valid dates & tickers
        :param tickers_list which has all symbols and starting date & ending date
        :return list of StockData object & Multiple Csv file for them

        """
        stocks = []
        for x in self.tickers_list:
            stock = StockData(x, self.starting_year, self.starting_month, self.starting_day, self.ending_year,
                              self.ending_month, self.ending_day)
            stock.save_to_csv_from_yahoo()
            stocks.append(stock)
        return stocks

    def stocks_list(self):
        stocks = []
        for x in self.tickers_list:
            stock = StockData(x, self.starting_year, self.starting_month, self.starting_day, self.ending_year,
                              self.ending_month, self.ending_day)
            stocks.append(stock)
        return stocks

    def merge_dataframe_by_column_name(self, col_name, tickers):
        """
        :precondition valid dates & tickers
        :param tickers list
        :return a dataframe that have all the stocks column

        """
        mult_df = pd.DataFrame()

        for x in tickers:
            mult_df[x] = web.DataReader(x, 'yahoo', self.starting_date, self.ending_date)[col_name]

        return mult_df

    def get_adj_close_for_all_stocks(self):
        mult_df = pd.DataFrame()

        for x in self.stocks_list():
            mult_df[x.get_ticker()] = web.DataReader(x.get_ticker(), 'yahoo', self.starting_date, self.ending_date)['Adj Close']


        return mult_df

    def daily_return_for_all_stocks(self):
        mult_df = pd.DataFrame()
        slist = self.stocks_list()
        for x in slist:
            mult_df[x.get_ticker()] = x.get_daily_return_in_period().values
        return mult_df

    def plot_return_mult_stocks(self, investment_amount, stock_df):
        """
           :precondition valid dataframe with adj close value for each stock
           :param dataframe of multiple stocks that have adj close values
           :return plots that show you how would your investment would look like if invested in these companies

           """
        (stock_df / stock_df.iloc[0] * investment_amount).plot(figsize=(15, 6))
        plt.show()

    def get_stock_mean_sd(self, stock_df, ticker):
        return stock_df[ticker].mean(), stock_df[ticker].std()

    def get_mult_stock_mean_sd(self, stock_df):
        for stock in stock_df:
            mean, sd = self.get_stock_mean_sd(stock_df, stock)
            cov = sd / mean
            print("Stock: {:4} Mean: {:7.2f} Standard deviation: {:2.2f}".format(stock, mean, sd))
            print("Coefficient of Variation: {}\n".format(cov))


def get_all_tickers_USA_yahoo():
    df1 = pd.DataFrame(si.tickers_sp500())
    df2 = pd.DataFrame(si.tickers_nasdaq())
    df3 = pd.DataFrame(si.tickers_dow())
    sym1 = set(symbol for symbol in df1[0].values.tolist())
    sym2 = set(symbol for symbol in df2[0].values.tolist())
    sym3 = set(symbol for symbol in df3[0].values.tolist())
    ymbols = set.union(sym1, sym2, sym3)
    x = list(ymbols)
    pf = pd.DataFrame(x)
    pf.to_csv('US.csv')
    return x


def get_all_tickers_KSA_from_csv():
    from csv import reader
    with open('Tasi.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        # Passing the cav_reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
        list_tickers = []
        for x in range(len(list_of_rows)):
            lis = list_of_rows[x]
            list_tickers.append(lis[0])
    return list_tickers


def get_all_tickers_USA_from_csv():
    from csv import reader
    with open('US.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        # Passing the cav_reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
        list_tickers = []
        for x in range(len(list_of_rows)):
            lis = list_of_rows[x]
            list_tickers.append(lis[0])
    return list_tickers


def get_sector_tickers(sector):
    sec_df = pd.read_csv("stock_sectors.csv")
    indus_df = sec_df.loc[sec_df['Sector'] == sector]
    df = indus_df['Symbol'].values.tolist()
    return df
    # Industrials, Health Care,Information Technology,Consumer Staples,Consumer Discretionary,Utilities
    # Financials, Materials, Real Estate, Energy


# ---------------------------------------------------------------------------------------------------------------

# TEST CODE:

# ---------------------------------------------------------------------------------------------------------------


# tickers = ["FB", "AAPL", "NFLX", "GOOG","AMZN"]
# stocks = Stocks(tickers,2020, 1, 1, 2021, 1, 1)
# stocks.download_multiple_stocks()
#
# mdf = stocks.merge_dataframe_by_column_name("Adj Close")
# stocks.plot_return_mult_stocks(100,mdf)
# stocks.get_mult_stock_mean_sd(mdf)
# print(get_all_tickers_KSA())

# US_tickers = get_all_tickers_USA_from_csv()
#
#
# stock = StockData("MRNS",2017,1,3,2017,12,31)
# print(stock.get_adj_close_in_period())
# print(stock.get_mean_in_period())
# print(stock.get_std_in_period())
# print(stock.get_coefficient_in_period())
# print(stock.get_roi_in_period())
# print(get_sector_tickers("Industrials"))
