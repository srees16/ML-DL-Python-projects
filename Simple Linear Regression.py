#Simple Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

#Importing the data set with pandas and taking the necessary variables
dataset = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
'''
x = dataset['YearsExperience']
y = dataset['Salary']
'''
# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest= train_test_split(x,y,test_size=0.2,random_state=11)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
xTrain=scX.fit_transform(xTrain)
xTest=scX.fit_transform(xTest)

# Fitting Simple Linear Regression model to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xTrain,yTrain)

#Predicting results from test set
yPred=regressor.predict(xTest)

'''
# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))
rmse(yPred, yTest)
'''

#Visualizing the results of training set
plot.scatter(xTrain,yTrain,color='red')
plot.plot(xTrain,regressor.predict(xTrain),color='blue')
plot.title('Salary vs Experience(Training Set)')
plot.xlabel('Yrs of Experience')
plot.ylabel('Salary')
plot.show()

#Visualizing the results of test set
plot.scatter(xTest,yTest,color='red')
plot.plot(xTrain,regressor.predict(xTrain),color='blue')
plot.title('Salary vs Experience(Test Set)')
plot.xlabel('Yrs of Experience')
plot.ylabel('Salary')


from sklearn.metrics import mean_squared_error
from math import sqrt

result = sqrt(mean_squared_error(yTest, yPred))
