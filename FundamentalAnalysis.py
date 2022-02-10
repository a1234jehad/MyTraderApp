import yfinance as yf
from prettytable import PrettyTable
import textwrap


class FundamentalAnalysis:
    def __init__(self, ticker):
        self.stock = yf.Ticker(ticker)

    def get_basic_info(self):
        info = self.stock.info
        x = PrettyTable()
        x.hrules = 1
        print("Summary of the company:", '\n')
        print()
        print(textwrap.fill(info['longBusinessSummary'], 140))
        print()
        x.field_names = ["Type", "Info", "Description "]
        needed_info = ['sector', "fullTimeEmployees", 'industry', 'ebitdaMargins', 'profitMargins', 'grossMargins',
                       'revenueGrowth', 'operatingMargins', 'earningsGrowth', 'currentRatio', 'returnOnAssets',
                       'debtToEquity', 'returnOnEquity', 'operatingCashflow', 'totalCash', 'totalDebt', 'totalRevenue','bookValue'
                       'currentPrice', 'targetLowPrice','targetMedianPrice','recommendationKey' ]

        description = {
            'ebitdaMargins': 'The EBITDA margin is a measure of a company\'s operating profit as a percentage of its '
                             'revenue \nEBITDA margin = (earnings before interest and tax + depreciation + amortization)'
                             ' / total revenue\nAn EBITDA margin of 10% or more is typically considered good',
            'profitMargins': 'The profit margin is a ratio of a company\'s profit\n (sales minus all expenses) '
                             'divided by its revenue.\nAs a rule of thumb, 5% is a low margin, 10% is a healthy '
                             'margin, and 20% is a high margin',
            'grossMargins': 'amount of money a company retains after incurring the direct costs associated with\n '
                            'producing the goods it sells and the services it provides.\nThe higher the gross '
                            'margin, the more capital a company retains. \ngross profit margin ratio of 50 to 70% '
                            'would be considered healthy ',
            'revenueGrowth': 'rate of increase in total revenues divided by total revenues from the same period in '
                             'the previous year.\nany business with a revenue growth rate of 10% or more is '
                             'considered good. \nHowever, a 2 or 3% growth rate is also regarded as healthy in some '
                             'cases  ',
            'operatingMargins': 'A higher operating margin indicates that the company is earning enough money from\n '
                                'business operations to pay for all of the associated costs involved in maintaining '
                                'that business.\n For most businesses, an operating margin higher than 15% is '
                                'considered good. ',
            'earningsGrowth': 'Earnings growth is the annual compound annual growth rate (CAGR) of earnings from '
                              'investments\n ',
            'currentRatio': 'The current ratio is a liquidity ratio that measures a company\'s ability \nto pay '
                            'short-term obligations or those due within one year\na current ratio below 1.00 could '
                            'indicate that a company might struggle to meet its short-term obligations'
                            '\nwhereas ratios of 1.50 or greater would generally indicate ample liquidity',
            'returnOnAssets': 'ROA shows how efficient a company is at using its assets to generate profits.\n'
                              'A ROA of over 5% is generally considered good and over 20% excellent',
            'debtToEquity': 'reflects the ability of shareholder equity to cover all outstanding debts in the event '
                            'of a business downturn\n '
                            'Generally, a good debt-to-equity ratio is around 1 to 1.5',
            'returnOnEquity': 'Return on equity (ROE) is the measure of a company\'s net income divided by its '
                              'shareholders\' equity\n ROEs of 15–20% are generally considered good. ',
            'operatingCashflow': 'Operating cash flow indicates whether a company can generate \nsufficient positive '
                                 'cash flow to maintain and grow its operations '
        }
        for key in needed_info:
            try:
                val = info[key]
            except Exception:
                continue

            if key in (needed_info and description):
                x.add_row([key, val, description[key]])
            elif key in needed_info:
                x.add_row([key, val, "N/A"])

                # print(key,":",val)
                # print()
        print(x)

    def plot_cashflow_quart(self):
        pass

FA = FundamentalAnalysis("7002.SR")
FA.get_basic_info()
