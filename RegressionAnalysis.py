import pandas as pd
from Stocks import *
from StockData import *
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

"""
AutoRegressive Integrated Moving Average (ARIMA) is the basis for many other models. It focuses on trying to fit the data as well as possible by examining differences between values instead of the values themselves.

ARIMA works very well when data values have a clear trend and seasonality. We can only make predictions based on the data we have. Any outside effects not in the data can't be used to make predictions. For example we could make predictions on stock prices, but since we don't know when a recession may occur that event can't be modeled.

There is a seasonal (SARIMA) and a non-seasonal ARIMA. There is also SARIMAX which focuses on exogenous, or external factors. It differs from ARIMA in that it has a set of parameters (P, D, and Q) that focus on seasonality.

AR (Autoregressions) refers to a model that regresses based on prior values.
"""


class RegressionAnalysis:
    def __init__(self, stock):
        self.stock = stock
        self.df = self.stock.get_adj_close_in_period()
        self.df = self.df.asfreq('d')
        self.df = self.df.fillna(method='ffill')  # fill missing with avg
        print("ss",self.df)

    def performe_RA(self,frame):
        sns.set_style('darkgrid')
        pd.plotting.register_matplotlib_converters()  # automatic datetime converters
        sns.mpl.rc('figure', figsize=(15, 10))
        fig, ax = plt.subplots()
        lags = ar_select_order(self.df, maxlag=30)  # remove weekends i.e

        model = AutoReg(self.df, lags.ar_lags)
        model_fit = model.fit()
        model.fit()
        days = len(self.df)
        # take 80% for training and 20% for testing
        trainings = int(days * 0.8)
        train_df = self.df.iloc[30:trainings]
        test_df = self.df.iloc[trainings:]
        # Define training model for "frame" days Change will lead to different results
        print(np.asarray(self.df))

        train_model = AutoReg(self.df, frame).fit(cov_type="HC0")
        start = len(train_df)
        end = len(train_df) + len(test_df) - 1

        prediction = train_model.predict(start=start, end=end, dynamic=True)

        ax = test_df.plot(ax=ax)
        ax = prediction.plot(ax=ax)
        forecast = train_model.predict(start=end, end=end + 60, dynamic=True)
        ax = forecast.plot(ax=ax)
        plt.show()
        # Get starting price of prediction
        s_price = forecast.head(1).iloc[0]

        # Get the last price of prediction
        e_price = forecast.iloc[-1]
        # Get return over prediction
        return (e_price - s_price) / s_price



Ra = RegressionAnalysis(StockData("amzn", 2017, 1, 1, 2021, 8, 2))
print("ss",type(Ra.df))
print(Ra.df)
Ra.performe_RA(750)
