from stocksent import Sentiment


class SentimentAnalysis:
    def __init__(self, ticker):
        self.ticker = ticker

    def plot_Sentiment(self):
        stocks = Sentiment(self.ticker)
        stocks.plot()

    def get_Sentiment(self):
        stocks = Sentiment(self.ticker)
        sentiment_score = stocks.get_sentiment(days=30)  # Get the sentiment for the past x days.
        print(sentiment_score)  # Returns a float with the sentiment score.
        return sentiment_score

    def get_df(self):
        stocks = Sentiment(self.ticker)
        sentiment_score = stocks.get_dataframe(days=30)  # Get the headlines for the past x days.
        print(sentiment_score)  # Returns a DataFrame with headlines, source and sentiment scores
        return sentiment_score





