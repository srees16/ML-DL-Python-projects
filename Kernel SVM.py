#Kernel SVM
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:, [2,4]].values
y=dataset.iloc[:, 4].values
#Splitting the data into test and training set
from sklearn.model_selection import train_test_split as tts
xTrain,yTrain,xTest,yTest=tts(x,y,test_size=0.25,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sx=StandardScaler()
sy=StandardScaler()
x=sx.fit_transform(x)
y=sy.fit_transform(y)
#Fitting the classifier to the training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf')
classifier.fit(x,y)
#Predicting the test set results
yPred=classifier.predict(52210)
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yTest,yPred)