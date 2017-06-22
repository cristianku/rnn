# This will use Recurral Neural Network to predict the stock prices in the future

# STOCK: Facebook Inc - https://www.google.com/finance/historical?cid=296878244325128&startdate=Jan+1%2C+2010&enddate=Dec+31%2C+2016&num=30&ei=ncRLWeC_CsrOswGw6JbwCQ

# TRAINING WITH : 18/05/2012 - 31-12-2016

#PREDICTION ON 01/01/2017 -

# Recurral Neural Network trained with 1 step in the future comparison ( next day stock price ),
# and with 1 attribute ( opening stock price )


# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv('facebook-historical-2010-2016.csv')
training_set = training_set.iloc[:,1:2].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
print len(training_set)
training_set = sc.fit_transform(training_set)
# remember that :
# the index of the last element is always : number of elements -1 ( it starts at 0 ! )
# the right border in this notation is excluded ( we dont want the last day )
# Getting the inputs and the ouputs
# the feature Number 1 is the every day  value of the stock
# Remember the right border is excluded
X_train = training_set[0:len(training_set)  -1]

# the prediction is the next day value
# we give a right number out of border to include until the last element ( excluded from the previous )
y_train = training_set[1:len(training_set) ]



# Reshaping
X_train = np.reshape(X_train, (len(training_set) -1 , 1, 1))

#Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
#adam = Gradient Descent loss minimizer
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)
# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
test_set = pd.read_csv('facebook-historical-2017.csv')
real_stock_price = test_set.iloc[:,1:2].values
next_day_real_2017_prices = real_stock_price[1:len(real_stock_price) ]

real_stock_price = sc.fit_transform(real_stock_price)
# Getting the predicted stock price of 2017

input_data_for_prediction = real_stock_price[0:len(real_stock_price) -1]

inputs = np.reshape(input_data_for_prediction, (len(input_data_for_prediction), 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(next_day_real_2017_prices, color = 'red', label = 'Real Facebook Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Facebook Stock Price')
plt.title('Facebook Stock Price Prediction 2017')
plt.xlabel('Time')
plt.ylabel('Facebook Stock Price')
plt.legend()
plt.show()

print predicted_stock_price