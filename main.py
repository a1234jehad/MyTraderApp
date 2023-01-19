import matplotlib.pyplot as plt
import yfinance as yf
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from AI import *
from SentimentAnalysis import *
from  TechnicalAnalysis import *
cf.go_offline()

Ticker = "MSFT"
period = "5y"
intervals = "1d"



#my_AI = AI(Ticker,period,intervals)
my_sent = SentimentAnalysis(Ticker)
my_TA = TechnicalAnalysis(Ticker,period,intervals)
# fixing stocks & stockdata toDo
#FA toDo
#gui toDO