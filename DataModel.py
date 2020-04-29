from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yfinance as yf

import tweepy
from tweepy import Cursor
from datetime import datetime, timedelta

yf.pdr_override()
register_matplotlib_converters()

auth = tweepy.OAuthHandler('6eE9ljLVoFw0dO7tzE5LvVuAV', 'uWWH069Y0DN3Pl2J7P50bJ5sh63TkpW9t7KaINUwCd2YrqSKBq')
auth.set_access_token('1076668512-rYvNa96YK7v4ivroXF48kfZPmuOUMH8LfkRrIva', 'iUjkAQ843cyc7zvbRJ56Zwwwv1HFcnfnEpm0NoJIhy4Cs')
api = tweepy.API(auth)

tsla = yf.Ticker("TSLA")
tweets = Cursor(api.user_timeline, id='@Tesla').items()

yesterday = datetime.today() - timedelta(days=1)
start_date = datetime(2019, 1, 1)

dataframe = pdr.get_data_yahoo("TSLA", start=start_date.date(), end=yesterday.date())
dataframefragment = dataframe.loc[:, ["Adj Close"]]
dataframefragment['shift'] = dataframefragment['Adj Close'].shift(-1)

tweet_dates = list()

for tweet in tweets:
    if tweet.created_at < start_date:
        break
    if tweet.created_at < yesterday:
        tweet_dates.append(np.datetime64(tweet.created_at.date()))

tweeted = list()

for i in range(0, len(dataframe.index.values)):
    stock_date = dataframe.index.values[i].astype('datetime64[D]')
    value = 0
    for dates in tweet_dates:
        if stock_date == dates:
            value = 1
        else:
            value = 0
    tweeted.append(value)

for j in range(0, len(tweeted)):
    if j > 0:
        tweeted[j-1] = tweeted[j]
    tweeted[len(tweeted)-1] = 0

dataframefragment['tweeted'] = tweeted

StockPrices = np.array(dataframefragment.drop(['shift'], 1))
# what does this do?

StockPrices = preprocessing.scale(StockPrices)

# Adj Close form yesterday
# Will be used to predict the price of today
StockPrice_yesterday = StockPrices[-1:]
StockPrices = StockPrices[:-1]

StockPrices_shifted = np.array(dataframefragment['shift'])
StockPrices_shifted = StockPrices_shifted[:-1]

X_train, X_test, y_train, y_test = train_test_split(StockPrices, StockPrices_shifted, test_size=0.33)

classifier = make_pipeline(PolynomialFeatures(5), Ridge())
classifier.fit(X_train, y_train)

confidence = classifier.score(X_test, y_test)
print("The polynomial regression confidence is ", confidence)

StockPrice_forecast = classifier.predict(StockPrice_yesterday)
print("StockPrice_forecast: ", StockPrice_forecast[0])

dataframefragment['Adj Close'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



