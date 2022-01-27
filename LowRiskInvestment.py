import numpy as np
import pandas as pd
from Stocks import *
from StockData import *


class LowRiskInvestment:
    def __init__(self, stocks):
        self.s = stocks
        self.stocks = stocks.stocks_list()
        self.df_daily_return = stocks.daily_return_for_all_stocks()
        self.df_adj_close = stocks.get_adj_close_for_all_stocks()

    def get_cov_roi_for_all_stocks(self):
        # it will return return on investments and coefficient variance
        col_names = ["Ticker", "COV", "ROI"]
        df = pd.DataFrame(columns=col_names)
        stocks_list = self.stocks
        for stock in stocks_list:
            df.loc[len(df.index)] = [stock.get_ticker(), stock.get_coefficient_in_period(), stock.get_roi_in_period()]
        return df

    def get_top_n_roi(self, n):
        df = self.get_cov_roi_for_all_stocks().sort_values(by=['ROI'], ascending=False).head(n)
        return df

    def get_correlation_FANNGS(self):
        self.df_daily_return.corr().plot(kind='bar')
        plt.show()

    def get_correlation_FANNGS_for_stock(self, ticker):
        self.df_daily_return.corr()[ticker].plot(kind='bar')
        plt.show()

    def get_Variance_for_stock(self, ticker):
        df = self.df_daily_return
        return df[ticker].var() * len(df)

    def get_covariance_of_stocks(self):
        df = self.df_daily_return
        return df.cov() * len(df)

    def plot_growth_of_investment(self):
        (self.df_adj_close / self.df_adj_close.iloc[0] * 100).plot(figsize=(16, 9))
        plt.show()

    def optimize_portfolio(self, number_combination):
        risk_free_rate = 0.0125  # years bound rate
        portfolio_returns = []
        portfolio_volatility = []
        portfolio_ratio = []
        portfolio_weights = []
        returns = np.log(self.df_adj_close / self.df_adj_close.shift(1))
        for x in range(number_combination):
            p_weights = np.random.random(len(self.stocks))
            p_weights /= np.sum(p_weights)

            ret = np.sum(p_weights * returns.mean()) * 252
            portfolio_returns.append(ret)  # saving return

            vol = np.sqrt(np.dot(p_weights.T, np.dot(returns.cov() * 252, p_weights)))
            portfolio_volatility.append(vol)  # saving volatility

            sr = (ret - risk_free_rate) / vol
            portfolio_ratio.append(sr)  # saving sharpe ratio

            portfolio_weights.append(p_weights)

        portfolio_returns = np.array(portfolio_returns)
        portfolio_volatility = np.array(portfolio_volatility)
        portfolio_ratio = np.array(portfolio_ratio)
        portfolio_weights = np.array(portfolio_weights)

        SR_index = np.argmax(portfolio_ratio)

        for x in range(len(self.stocks)):
            print("Stocks: %s : %2.2f" % (self.stocks[x].get_ticker(), (portfolio_weights[SR_index][x] * 100)))
        print(f"Volatility = {portfolio_volatility[SR_index]}")
        print(f"Return = {portfolio_returns[SR_index]}")

    def calculate_beta(self):
        """
        Beta provides the relationship between an investment and the overall market. Risky investments tend to fall
         further during bad times, but will increase quicker during good times.

        Beta is found by dividing the covariance of the stock and the market by the variance of the overall market.
        It is a measure of systematic risk that can't be diversified away.

        B = 0 no relation to market
        B < 1 less risky than market
        B > 1 More risky than market
        """
        for x in self.stocks:
            Tickers = ["^GSPC"]
            Tickers.append(x.get_ticker())
            ss = Stocks(Tickers,self.s.starting_year, self.s.starting_month, self.s.starting_day, self.s.ending_year,
                              self.s.ending_month, self.s.ending_day)
            daily_df = ss.daily_return_for_all_stocks()
            cov = daily_df.cov() * 252
            cov_vs_market = cov.iloc[0,1]
            sp_var = daily_df['^GSPC'].var() * 252
            beta = cov_vs_market / sp_var
            print(f"Stock : {x.get_ticker()} has a Beta = {beta}")

        tickers = self.s.tickers_list
        tickers.append("^GSPC")
        ss = Stocks(tickers, self.s.starting_year, self.s.starting_month, self.s.starting_day, self.s.ending_year,
                    self.s.ending_month, self.s.ending_day)
        daily_df = ss.daily_return_for_all_stocks()
        cov = daily_df.cov() * 252
        cov_vs_market = cov.iloc[0, 1]
        sp_var = daily_df['^GSPC'].var() * 252
        beta = cov_vs_market / sp_var
        print(f"Beta for whole portfolio : {beta}")
        return beta

    def get_portfolio_roi_tot(self):
        df = self.df_adj_close

        for i in range(len(self.stocks)):
            tic = self.stocks[i].get_ticker()
            amount = int(input(f"How many {tic} shares you want?"))
            df[tic] = df[tic].apply(lambda x: x * amount)
        df["Total"] = df.iloc[:,0:len(self.stocks)].sum(axis=1)
        start_value = df["Total"].iloc[0]
        end_value = df["Total"].iloc[-1]
        roi_tot = (end_value-start_value)/start_value
        return roi_tot

    def get_alhpa(self):
        risk_free_rate = 0.0125
        roi_tot = self.get_portfolio_roi_tot()
        sto = StockData("^GSPC",self.s.starting_year, self.s.starting_month, self.s.starting_day, self.s.ending_year,
                    self.s.ending_month, self.s.ending_day)
        sp_roi = sto.get_roi_in_period()
        print(f"Portfolio ROI: {roi_tot}")
        print(f"S&P ROI: {sp_roi}")
        p_alpha = roi_tot - risk_free_rate -(self.calculate_beta()*(sp_roi - risk_free_rate))
        print(f"ALpha = {p_alpha}")




# ---------------------------------------------------------------------------------------------------------------

# TEST CODE:

# ---------------------------------------------------------------------------------------------------------------


# tickers = ["FB", "AAPL", "NFLX", "GOOG", "AMZN", "RIOT"]
# stocks = Stocks(tickers, 2020, 1, 2, 2020, 12, 31)
# lri = LowRiskInvestment(stocks)
# lri.get_correlation_FANNGS()
# print(lri.get_Variance_for_stock("NFLX"))
# print(lri.get_covariance_of_stocks())

# tickers = get_sector_tickers("Industrials")
# port_list = ["GNRC", "DXCM", "AMD", "NFLX", "COST", "TGT", "AES", "MSCI",
#              "NEM", "SBAC", "HES"]
#port_list = ["GOOG","FB"]
port_list = ["IDEX","DSS","PLTR","URA","NPK","INTC"]
stocks = Stocks(port_list, 2021, 1, 1, 2021, 12, 31)
lri = LowRiskInvestment(stocks)
lri.get_correlation_FANNGS()
print(lri.get_top_n_roi(5))
lri.plot_growth_of_investment()
lri.optimize_portfolio(10000)
lri.get_alhpa()

#get Alpha for Tasi !!