import matplotlib.pyplot as plt
import yfinance as yf
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

cf.go_offline()
Ticker = "MSFT"
period = "5y"
intervals = "1d"

stock = yf.Ticker(ticker=Ticker)
dr =stock.quarterly_balancesheet
data = dr.transpose()
print(dr)
# invstments = go.Scatter(x= data.index , y= data["Investments"], line=dict(color='orange', width=1), name="Investments")
# Net_Income = go.Scatter(x= data.index , y= data["Net Income"], line=dict(color='orange', width=1), name="Net Income")
# fig = go.Figure()
# fig.add_trace(trace=invstments)
# fig.add_trace(trace= Net_Income)
# fig.update_layout(title = 'Cashflow quarterly')
#
# fig.show()
print()