#Polynomial Regression

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

#Importing the data set with pandas and taking the necessary variables
datasets = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Position_Salaries.csv')
x = datasets.iloc[:,1:2].values
y = datasets.iloc[:,2].values

#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
linReg=LinearRegression()
linReg.fit(x,y)

#Visualising the linear regression results
plot.scatter(x,y,color='red')
plot.plot(x,linReg.predict(x),color='green')
plot.title('Salary Vs Experience (Linear Regression)')
plot.xlabel('Experience')
plot.ylabel('Salary')
plot.show()

#Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg=PolynomialFeatures(degree=4)
xPoly=polyReg.fit_transform(x)
linReg2=LinearRegression()
linReg2.fit(xPoly,y)

#Visualising the polynomial regression results
xGrid=np.arange(min(x),max(x),0.1)
xGrid=xGrid.reshape(len(xGrid),1)
plot.scatter(x,y,color='red')
plot.plot(xGrid,linReg2.predict(polyReg.fit_transform(xGrid)),color='green')
plot.title('Salary Vs Experience (Polynomial Regression)')
plot.xlabel('Experience')
plot.ylabel('Salary')
plot.show()

#Predicting new results with Linear regression
linReg.predict(6.5)

#Predicting new results with Polynomial regression
linReg2.predict(polyReg.fit_transform(6.5))