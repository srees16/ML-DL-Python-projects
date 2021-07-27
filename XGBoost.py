#XGBoost
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:, 13].values
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lEncoder=LabelEncoder()
x[:,1]=lEncoder.fit_transform(x[:,1])
x[:,2]=lEncoder.fit_transform(x[:,2])
hotEncoder=OneHotEncoder(categorical_features=[1])
x=hotEncoder.fit_transform(x).toarray()
x=x[:,1:]
#Splitting the data into test and training set
from sklearn.model_selection import train_test_split as tts
xTrain,yTrain,xTest,yTest=tts(x,y,test_size=0.2,random_state=0)
#Fitting XGBoost to training set
from xgboost import XGBClassifier
