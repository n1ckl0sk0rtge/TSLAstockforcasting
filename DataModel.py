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
import datetime


yf.pdr_override()
register_matplotlib_converters()

tsla = yf.Ticker("TSLA")


dataframe = pdr.get_data_yahoo("TSLA", start="2019-04-28", end="2020-04-28")
dataframefragment = dataframe.loc[:, ["Adj Close"]]
dataframefragment['shift'] = dataframefragment['Adj Close'].shift(-1)

StockPrices = np.array(dataframefragment.drop(['shift'], 1))
# what does this do?
StockPrices = preprocessing.scale(StockPrices)

# Adj Close form yesterday
# Will be used to predict the price of today
StockPrice_yesterday = StockPrices[-1:]
StockPrices = StockPrices[:-1]

StockPrices_shifted = np.array(dataframefragment['shift'])
StockPrices_shifted = StockPrices_shifted[:-1]

X_train, X_test, y_train, y_test = train_test_split(StockPrices, StockPrices_shifted, test_size=0.3)

classifier = make_pipeline(PolynomialFeatures(4), Ridge())
classifier.fit(X_train, y_train)

confidence = classifier.score(X_test, y_test)
print("The knn regression confidence is ", confidence)

StockPrice_forecast = classifier.predict(StockPrice_yesterday)
print("StockPrice_forecast: ", StockPrice_forecast[0])

dataframefragment['Adj Close'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



