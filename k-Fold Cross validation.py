#k-Fold Cross validation
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values
#Splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
xTrain,xTest,yTrain,yTest= train_test_split(x,y,test_size=0.25,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xTrain=ss.fit_transform(xTrain)
xTest=ss.transform(xTest)
#Fitting logistic regression model to the dataset
from sklearn.linear_model import LogisticRegression as lr
classifier=lr(random_state=0)
classifier.fit(xTrain,yTrain)
#Predicting the test set
yPred=classifier.predict(xTest)
#Evaluating using Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yTest,yPred)
#Visualization of the train results
from matplotlib.colors import ListedColormap
xSet,ySet=xTrain,yTrain
x1,x2=np.meshgrid(np.arange(start=xSet[:,0].min()-1,stop=xSet[:,0].max()+1,step=0.01),
                  (np.arange(start=xSet[:,1].min()-1,stop=xSet[:,1].max()+1,step=0.01))

#Visualization of the test results