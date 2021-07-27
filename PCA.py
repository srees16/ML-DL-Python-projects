#PCA
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Wine.csv')
x=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values
#Splitting the data into test and training set
from sklearn.cross_validation import train_test_split as tts
xTrain,yTrain,xTest,yTest=tts(x,y,test_size=0.2,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xTrain=ss.fit_transform(xTrain)
xTest=ss.fit_transform([xTest])
#Applying PCA
'''
#Checking for the most significant components by trial & error
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
xTrain=pca.fit_transform(xTrain)
xTest=pca.fit_transform(xTest)
variance=pca.explained_variance_ratio_
'''
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
xTrain=pca.fit_transform(xTrain)
xTest=pca.fit_transform(xTest)
#Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(xTrain,xTest)
#Predicting the test set results
yPred=classifier.predict(xTest)
#Evaluating model performance by using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yTest,yPred)