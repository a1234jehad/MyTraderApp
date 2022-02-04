import praw
import os
from textblob import TextBlob


class RedditNLP:
    def __init__(self):
        self.reddit_id = os.environ["PersonalRedditApi"]
        self.Secret = os.environ["RedditApi"]
        self.user = os.environ["user_name"]
        self.password = os.environ["pass"]
        self.reddit = praw.Reddit(
            client_id=self.reddit_id,
            client_secret=self.Secret,
            password=self.password,
            user_agent="USERAGENT",
            username=self.user,
        )

    def strategy1(self,numberofsent,subreddit):
        stat_num = numberofsent

        def Average(lst):
            if len(lst) == 0:
                return 0
            else:
                return sum(lst[-stat_num:]) / stat_num

        # for comment in self.reddit.subreddit("redditdev").comments(limit=50):
        #     print(comment.body)

        # for submission in self.reddit.subreddit("bitcoin").hot(limit=25):
        #     print(submission.title)
        lst = []
        for comment in self.reddit.subreddit(subreddit).stream.comments():
            #print(comment.body)
            redditComment = comment.body
            blob = TextBlob(redditComment)
            # print(blob.sentiment)
            # 1 is pos and -1 is neg 0 netural
            sent = blob.sentiment
            if sent.polarity != 0:
                lst.append(sent.polarity)
                if round(Average(lst) > 0.5 and len(lst) > stat_num):
                    print("buy")
                elif round(Average(lst) < -0.5 and len(lst) > stat_num):
                    print("sell")
            #print(f"*****************Sentiment is: {sent.polarity}*********************")

    def strategy2(self):
        pass #combine RSI with Sent...


RN = RedditNLP()
RN.strategy1(50,"bitcoin")
