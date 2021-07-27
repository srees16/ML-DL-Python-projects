#Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv(r'D:\Machine Learning Data Science\Projects\Datasets\50_Startups.csv')
x=dataset.iloc[::-1].values
y=dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX=LabelEncoder()
x[:,3]=labelEncoderX.fit_transform(x[:,3])
hotEncoder=OneHotEncoder(categorical_features=[3])
x=hotEncoder.fit_transform(x).toarray()

# Avoiding the dummy variable trap
x=x[:, 1:]

# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest= train_test_split(x,y,test_size=0.2,random_state=10)

# Fitting the multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xTrain,yTrain)

# predicting the test set
yPred=regressor.predict(xTest)

'''
# Building optimal model using backward elimination
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
xOpti=x[:,[0,1,2,3,4,5]]
regressorOLS=sm.OLS(endog=y,exog=xOpti).fit()
regressorOLS.summary()
xOpti=x[:,[0,3,4,5]]
regressorOLS=sm.OLS(endog=y,exog=xOpti).fit()
regressorOLS.summary()
xOpti=x[:,[0,3,5]]
regressorOLS=sm.OLS(endog=y,exog=xOpti).fit()
regressorOLS.summary()
xOpti=x[:,[0,3]]
regressorOLS=sm.OLS(endog=y,exog=xOpti).fit()
regressorOLS.summary()
'''