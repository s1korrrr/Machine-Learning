"""
The objective of linear algebra is to calculate relationships of points in vector space.
This is used for a variety of things, but one day, someone got the wild idea to do this with features of a dataset.
We can too! Remember before when we defined the type of data that linear regression was going to work on was
called "continuous" data? This is not so much due to what people just so happen to use linear regression for,
it is due to the math that makes it up. Simple linear regression is used to find the best fit line of a dataset.
If the data isn't continuous, there really isn't going to be a best fit line.
"""

import quandl, math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle

style.use("ggplot")

df = quandl.get("WIKI/GOOGL")

# We need to figure out what features of this data set to use for our stock price prediction.
# We'll choose only somewhat relevant features, in this case the Adjusted open, Adjusted close,
# The adjusted high and low stock price, the Adjusted close and the Adjusted volume.
# Note that 'Adjusted' simply means the price of the stock after such things as  stock splits.
# We could use the non-adjusted features as well, but using both adjusted and non-adjusted is
# not useful as they both essentially describe the same things.

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

# Here, we're going to create two new features computed from the features already given
# in the data set.  # We're going to need these features to track volatility in the stock
# which is an essential feature for predicting a new stock price.

# The first features is the high low percentage, calculated by subtracting the
# adjusted high price from the close and dividing by the close multiplied by 100.0 (to get a proper percentage value)

df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100.0

# Our next features is the Percentage price change obtained by subtracting the adjusted close
# from the adjusted open and dividing  by the adjusted open multiplied by 100.0

df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

# Once we have the features, we want to then get the labels, i.e. for a classifier, we want to know
# whether or not we have a True or a False value.

# Here we're using the Adjusted close as our label name.
forecast_col = "Adj. Close"

# With machine learning, we will replace any NaN values with a real number (even though it's an outlier).
# This is a better choice than getting rid of columns that don't have all the data.
df.fillna(-99999, inplace=True)

# Now we'll define our regression algorithm
# Let's first set how many days back we want to go  as a percentage of the number of days in the data frame.
# Here we'll set it to go back ten percent of the total number of days in the data frame.
forecast_out = int(math.ceil(0.01 * len(df)))

# Here we use the pandas shift method to shift the forecast label values out by the amount
# specified in the forecast_out variable.  Basically, the label will contain the adjusted close
# value ten days (as specified by the forecast_out variable) into the future.

df["label"] = df[forecast_col].shift(-forecast_out)

# Now, from our dataframe, let's get both our test data (X) and our label (y) that we will feed into our
# linear regression model.  Note that the test
X = np.array(df.drop(["label"], 1))


# The preprocessing module from sklearn 'normalizes' our features, that is to say, it tries to
# make sure that for each feature given, the values range roughly by one unit.  So, for example
# having features range in value between -1 and 1 or =0.5 to 0.5.
# The reason for doing this is to hopefully speed up the execution of the linear regression algorithm.
X = preprocessing.scale(X)

# Our X_lately variable contains the most recent features, which we're going to predict against
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df["label"])

# Now, let's get two sets of values.
# Set 1.  The training data for our linear regression algorithm.
# Set 2.  The testing data to see how well our algorithm learned from the training data.

# Here we're going to split up the data we got from Quandl for the training feature data (X_train) and the training label data (y_train)
# and the testing feature data (X_test) and the testing label data (y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Let's choose the SKLearn's Linear Regression algorithnm for our ML example
# clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR()
# clf = svm.SVR(kernel="poly")

# Let's run the training data through the algorithm.
# clf.fit(X_train, y_train)

# with open("linearregression.pickle", "wb") as f:
#     pickle.dump(clf, f)

# With pickle, you can save any Python object, like our classifier.
# After defining, training, and testing your classifier.

pickle_in = open("linearregression.pickle", "rb")
clf = pickle.load(pickle_in)

# We have commented out the original definition of the classifier and are instead loading in the one we saved.

# Now let's see how well the training set compares to the testing data.
accuracy = clf.score(X_test, y_test)

# The forecast_set is an array of forecasts, showing that not only could you just seek out a single prediction,
# but you can seek out many at once
forecast_set = clf.predict(X_lately)
df["Forecast"] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# So here all we're doing is iterating through the forecast set, taking each forecast and day,
# and then setting those values in the dataframe (making the future "features" NaNs).
# The last line's code just simply takes all of the first columns, setting them to NaNs,
# and then the final column is whatever i is (the forecast in this case)
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# A good result is to have an accuracy of 95 per cent or better (usually rather difficult to
# achieve with real world training sets)﻿
print(accuracy)

"""
Finally, while we're on the topic of being efficient and saving time, 
I want to bring up a relatively new paradigm in the last few years, and that is temporary super computers! 
Seriously. With the rise of on-demand hosting services, such as Amazon Webservices (AWS), 
Digital Ocean, and Linode, you are able to buy hosting by the hour. Virtual servers can be set up in about 60 seconds, 
the required modules used in this tutorial can all be installed in about 15 minutes or so at a fairly leisurely pace. 
You could write a shell script or something to speed it up too. Consider that you need a lot of processing, 
and you don't already have a top-of-the-line-computer, or you're working on a laptop. No problem, just spin up a server!

The last note I will make on this method is that, with any of the hosts, generally you can spin up a very small server,
load what you need, then scale UP that server. I tend to just start with the smallest server, then, when I am ready,
I resize the server, and go to town. When done, just don't forget to destroy or downsize the server when done.
"""