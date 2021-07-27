#Recurrent Neural Network

#Data preprocessing
import numpy as np
import matplotlib.pyplot as py
import pandas as pd
#Importing the training set
datasetTrain=pd.read_csv('Google_Stock_Price_Train.csv')
trainingSet=datasetTrain.iloc[:,1:2].values
#Feature scaling
from sklearn.preprocessing import MinMaxScaler as mms
sc=mms(feature_range=(0,1))
trainingSetScaled=sc.fit_transform(trainingSet)
#Creating data structure with 60 time steps and 1 output
xTrain=[]
yTrain=[]
for i in range(60,1258):
    xTrain.append(trainingSetScaled[i-60:i,0])
    yTrain.append(trainingSetScaled[i,0])
xTrain,yTrain=np.array(xTrain),np.array(yTrain)
#Reshape
xTrain=np.reshape(xTrain,(xTrain.shape[0],xTrain.shape[1],1))
#Building the RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# Initializing RNN
regressor=Sequential()
#Adding the first LSTM layer and dropout regularization
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(xTrain.shape[1],1)))
regressor.add(Dropout(0.2))
#Adding the second LSTM layer and dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the third LSTM layer and dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the fourth LSTM layer and dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
#Adding output layer
regressor.add(Dense(units=1))
#Compiling RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
#Fitting the RNN to the training set
regressor.fit(xTrain,yTrain,epochs=100,batch_size=32)
#Making predictions and visualizing results
#Getting real stock price
datasetTest=pd.read_csv('Google_Stock_Price_Test.csv')
realStockPrice=datasetTest.iloc[:,1:2].values
#Getting predicted stock price
datasetTotal=pd.concat((datasetTrain['Open'],datasetTest['Open']),axis=0)
inputs=datasetTotal[len(datasetTotal)-len(datasetTest)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
xTest=[]
for i in range(60,80):
    xTest.append(inputs[i-60:i,0])
xTest=np.array(xTest)
xTest=np.reshape(xTest,(xTest.shape[0],xTest.shape[1],1))
predictedStockPrice=regressor.predict(xTest)
predictedStockPrice=sc.inverse_transform(predictedStockPrice)
#Visualizing the results
py.plot(realStockPrice,color='red',label='Real Prices')
py.plot(predictedStockPrice,color='blue',label='Predicted Prices')
py.title('Stock Price Predictions')
py.xlabel('Time')
py.ylabel('Stock Price')
py.legend()
py.show()