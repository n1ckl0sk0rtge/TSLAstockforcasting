from pandas_datareader import data as pdr
import math
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yfinance as yf
import datetime

yf.pdr_override()
register_matplotlib_converters()

tsla = yf.Ticker("TSLA")
dataframe = pdr.get_data_yahoo("TSLA", start="2019-04-27", end="2020-04-27")

dataframefragment = dataframe.loc[:, ["Adj Close", "Volume"]]

forecast_out = int(math.ceil(0.01 * len(dataframefragment)))
forecast_col = 'Adj Close'
dataframefragment['label'] = dataframefragment[forecast_col].shift(-forecast_out)
X = np.array(dataframefragment.drop(['label'], 1))

X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(dataframefragment['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

confidenceknn = clfknn.score(X_test, y_test)

print("The knn regression confidence is ", confidenceknn)

forecast_set = clfknn.predict(X_lately)
dataframefragment['Forecast'] = np.nan
print(forecast_set, confidenceknn, forecast_out)

last_date = dataframefragment.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=10)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=10)
    dataframefragment.loc[next_date] = [np.nan for _ in range(len(dataframefragment.columns)-1)] + [i]

dataframefragment['Adj Close'].tail(500).plot()
dataframefragment['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# accuracy = classifier_svm.score(X_test, y_test)

# cm = confusion_matrix(y_test, pred)

# plt.plot(x_data)
# plt.show()




