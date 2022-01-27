import yfinance as yf

stock = yf.Ticker("^GSPC")
js = stock.get_info()
print(js)