#Kernel PCA
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values
#Splitting the data into test and training set
from sklearn.cross_validation import train_test_split as tts
xTrain,yTrain,xTest,yTest=tts(x,y,test_size=0.25,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xTrain=ss.fit_transform(xTrain)
xTest=ss.transform(xTest)
#Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca=KernelPCA(n_components=2,kernel='rbf')
xTrain=kpca.fit_transform(xTrain)
xTest=kpca.fit_transform(xTest)
#Checking for the most significant components by trial & error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
ld=lda(n_components=2)
xTrain=ld.fit_transform(xTrain,yTrain)
xTest=ld.transform(xTest)
#Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(xTrain,yTrain)
#Predicting the test set results
yPred=classifier.predict(xTest)
#Evaluating model performance by using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yTest,yPred)