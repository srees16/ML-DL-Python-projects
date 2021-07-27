#DATA PREPROCESSING FOR DATA.CSV FILE
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values
dataset.isnull().sum()#/len(dataset)*100 #checking for no of missing values. unblock for % of them
#Handling missing data. Method for imputing object type dataset
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
'''
#2nd way to handle missing data, but not working for imputation of dataframe type dataset
dataset['Age'].fillna(dataset['Age'].mean,inplace=True)
dataset['Salary'].fillna(dataset['Salary'].mode()[0],inplace=True)
dataset.isnull().sum()#verifying if all missing values are treated
'''
#Encoding categorical data and removing relational weights among them
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX=LabelEncoder()
x[:,0]=labelEncoderX.fit_transform(x[:,0])
hotEncoder=OneHotEncoder(categorical_features=[0])
x=hotEncoder.fit_transform(x).toarray()
labelEncoderY=LabelEncoder()
y=labelEncoderY.fit_transform(y)
#Splitting dataset into training and test set
from sklearn.model_selection import train_test_split as tts
xTrain,xTest,yTrain,yTest= tts(x,y,test_size=0.2,random_state=10)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
xTrain=scX.fit_transform(xTrain)
xTest=scX.fit_transform(xTest)