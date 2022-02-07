from stocksent import Sentiment


class SentimentAnalysis:
    def __init__(self, ticker):
        self.ticker = ticker

    def plot_sent(self):
        stocks = Sentiment(self.ticker)
        stocks.plot()

    def get_sent(self):
        stocks = Sentiment(self.ticker)
        sentiment_score = stocks.get_sentiment(days=30)  # Get the sentiment for the past 4 days.
        print(sentiment_score)  # Returns a float with the sentiment score.
        return sentiment_score

    def get_df(self):
        stocks = Sentiment(self.ticker)
        sentiment_score = stocks.get_dataframe(days=30)  # Get the headlines for the past 6 days.
        print(sentiment_score)  # Returns a DataFrame with headlines, source and sentiment scores
        return sentiment_score


s = SentimentAnalysis("AAPL")

s.plot_sent()