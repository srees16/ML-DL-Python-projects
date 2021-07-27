#Support Vector Machine
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values
#Splitting the data into test and training set
from sklearn.model_selection import train_test_split as tts
xTrain,yTrain,xTest,yTest=tts(x,y,test_size=0.25,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xTrain=ss.fit_transform(xTrain)
xTest=ss.transform(xTest)
xTest.reshape(1,-1)
#Fitting SVM model to training set
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(xTrain,xTest)
#Predicting the results of test set
yPred=classifier.predict(xTest)
yPred.reshape(1,-1)
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yTest,yPred)