# Exploratory Data Analysis

'''Assumptions to test: Normality, Homoscedasticity, Linearity, Absence of correlated errors

Normality can be probed by:
1. Histogram (Kurtosis and skewness)
2. Normal probability plot (data need to be distributed along the diagonal that represents the normal distribution)

If the data is not normally distributed, i.e., if it shows 'peakedness', positive skewness and does not follow the
diagonal line, perform log transformation of the data

Kurtosis:-
Zero kurtosis: If the kurtosis is close to 0, then it’s assumed to have normal distribution.
Positive kurtosis: Indicates that the distribution has heavier tails and a sharper peak than normal distribution.
Negative kurtosis: Dataset has as much data in each tail as it does in the peak i.e., the distribution is flatter
than a normal curve with the same mean and standard deviation.
Skew: Longer left tail is left skewed aka -ve skew & longer right tail is right skewed aka +ve skew.

EDA methods:
    Univariate analysis & visualization: provides summary statistics for each field in the raw data set
    Bivariate analysis & visualisation: find relationship of each variable in the dataset and the target variable
    Multi variate analysis & visualization: to understand interactions between different fields in the dataset
    Dimensionality Reduction
'''
import os
import sys
import time # use time.time() to calculate start & end times for operations
import string
import numpy as np
import pandas as pd
from numpy.random import randint, randn, sample
from numpy import array, count_nonzero
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import seaborn as sns
sns.set()
import pyforest
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
#init_notebook_mode(connected = True) # specific to Jupyter
cf.go_offline() # allows to use cufflinks offline
import math
from math import pi, sin
import category_encoders as ce
from collections import namedtuple, defaultdict, OrderedDict, Counter
from scipy import stats
import scipy.cluster.hierarchy as sch
from scipy.stats import norm, chi2 as c2, ttest_1samp
from scipy.sparse import csr_matrix
import statsmodels.api as sm
from statsmodels.stats import weightstats as stests
from statsmodels.formula.api import ols
import sklearn
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve, r2_score, mean_squared_error, mean_absolute_error, log_loss)
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn import datasets
from sklearn.datasets import load_breast_cancer, load_iris, load_digits
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, LabelBinarizer, MultiLabelBinarizer, OrdinalEncoder, LabelEncoder, OneHotEncoder)
from sklearn.feature_selection import (SelectFromModel, VarianceThreshold, SelectKBest, SelectPercentile, chi2, RFE, RFECV, f_classif)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from dython.nominal import associations
import prince # for multiple correspondence analysis
import xgboost
from xgboost import XGBClassifier as xgb, plot_importance
from ipywidgets import Image
from io import StringIO
import pdb
import re
import json
import joblib
import pickle
import requests
import pydotplus
from functools import reduce
import flask
from flask import Flask, jsonify, request, render_template, abort
from flask_restful import Resource, Api
import pymongo as pym
import psycopg2 as pgs
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, InputLayer, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import tensorflow as tf
import cv2
print('Numpy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sys: {}'.format(sys.version))
print('Sci-kit Learn: {}'.format(sklearn.__version__))
print('Open CV: {}'.format(cv2.__version__))
print('TensorFlow: {}'.format(tf.__version__))
print('Keras: {}'.format(keras.__version__))
print('PyMongo: {}'.format(pym.__version__))
print('XGBoost: {}'.format(xgboost.__version__))
print('PostgreSql: {}'.format(pgs.__version__))
print('Flask: {}'.format(flask.__version__))
print('Regex: {}'.format(re.__version__))
print(plotly.__version__)
# pwd # python working directory
'''
import warnings
warnings.filterwarnings("ignore")
'''

'''
Pandas Series & DataFrame objects have attributes like inplace, columns, index etc
'''
'''''''''''''''''''''''''''''''''''''''''EDA Adults'''''''''''''''''''''''''''''''''''''''''''''''''''

# https://www.kaggle.com/overload10/income-prediction-on-uci-adult-dataset/notebook
dataset = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\adult.csv')
data = [dataset]
dataset.head(3)
dataset.info()
dataset.shape
dataset.apply(np.max) # max values of dataset
dataset['capital-gain'].apply(np.max) # max value in a single column
# Convert salary to integer
salary_map = {' <=50K' : 1, ' >50K' : 0}
dataset['salary'] = dataset['salary'].map(salary_map).astype(int)

# convert sex into integer
dataset['sex'] = dataset['sex'].map({' Male' : 1, ' Female' : 0}).astype(int)
dataset.head(3)

# Find correlation between columns
def plot_correlation(data, size = 15):
    corr = data.corr(method = 'pearson')
    fig, ax = plt.subplots(figsize = (0.4 * size, 0.4 * size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
plot_correlation(dataset)

# Categorise in US and Non-US candidates
dataset[['country', 'salary']].groupby(['country']).mean()

# dataset.isna().sum().sum() # gives count of all null values across all the columns
# Drop empty value marked as '?'
dataset['country'].value_counts() # gives count of each of the values in a specific column
dataset['country'] = dataset['country'].replace(' ?', np.nan)
dataset['country'].isna().sum()#/len(df)*100

dataset['workclass'].value_counts()
dataset['workclass'].replace(' ?', np.nan, inplace = True) # instead of using assignment to dataframe, inplace can be used
dataset['workclass'].isnull().sum()

dataset['occupation'].value_counts()
dataset['occupation'] = dataset['occupation'].replace(' ?', np.nan)
dataset['occupation'].isnull().sum()

# OR, below is a single step solution for all 3 above features
dataset[['country', 'workclass', 'occupation']] = dataset[['country', 'workclass', 'occupation']].replace(' ?', np.nan)
# dataset[['country', 'workclass', 'occupation']].replace(' ?', np.nan, inplace = True) # same as above

dataset['country'] = dataset['country'].fillna(lambda x: x.mean())
dataset['workclass'] = dataset['workclass'].fillna(lambda x: x.mean())
dataset['occupation'] = dataset['occupation'].fillna(lambda x: x.mean())

# OR, below is a single step solution for all 3 above features
dataset[['country', 'workclass', 'occupation']] = dataset[['country', 'workclass', 'occupation']].fillna(lambda x: x.mean())
'''
dataset['country'] = dataset['country'].fillna(method = 'ffill')
dataset['workclass'] = dataset['workclass'].fillna(method = 'ffill')
dataset['occupation'] = dataset['occupation'].fillna(method = 'ffill')

'''
dataset.dropna(how = 'any', inplace = True)
dataset.shape
dataset.head(3)

for dataset in data:
    dataset.loc[dataset['country'] != ' United-States', 'country'] = 'Non-US'
    dataset.loc[dataset['country'] == ' United-States', 'country'] = 'US'
dataset.head(3)

# Convert country in integer
dataset['country'] = dataset['country'].map({'US' : 1, 'Non-US' : 0}).astype(int)
dataset.head(3)

# Data visualisation using histogram
x = dataset['hours-per-week']
plt.hist(x, bins = None, density = True, histtype = 'bar')
plt.show()

dataset[['relationship', 'salary']].groupby(['relationship']).mean()
dataset[['marital-status', 'salary']].groupby(['marital-status']).mean()

# Categorise marital-status into single and couple
single = [' Divorced', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed']
married = [' Married-AF-spouse', ' Married-civ-spouse']
dataset['marital-status'] = dataset['marital-status'].replace(single, 'Single')
dataset['marital-status'] = dataset['marital-status'].replace(married, 'Couple')
dataset.head(3)

dataset[['marital-status', 'salary']].groupby(['marital-status']).mean()

dataset[['marital-status', 'relationship', 'salary']].groupby(['marital-status', 'relationship']).mean() * 100
# dataset[['marital-status', 'relationship', 'salary']].groupby(['relationship', 'marital-status']).mean() * 100
dataset['marital-status'] = dataset['marital-status'].map({'Couple' : 0, 'Single' : 1})
dataset.head(3)

relation_map = {' Unmarried' : 0, ' Wife' : 1, ' Husband' : 2, ' Not-in-family' : 3, ' Own-child' : 4,
           ' Other-relative' : 5}
dataset['relationship'] = dataset['relationship'].map(relation_map)
dataset.head(3)

# Analyse race
dataset[['race', 'salary']].groupby('race').mean()

race_map = {' White' : 0, ' Amer-Indian-Eskimo' : 1, ' Asian-Pac-Islander' : 2, ' Black' : 3, ' Other' : 4}
dataset['race'].replace(race_map, inplace = True)
dataset.head(3)

dataset[['occupation', 'salary']].groupby(['occupation']).mean()
dataset[['workclass', 'salary']].groupby(['workclass']).mean()

def workclass(x):
    if x['workclass'] == ' Federal-gov' or x['workclass'] == ' Local-gov' or x['workclass'] == ' State-gov':
        return 'govt'
    elif x['workclass'] == ' Private':
        return 'private'
    elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc':
        return 'self_employed'
    else:
        return 'without_pay'

dataset['employment_type'] = dataset.apply(workclass, axis = 1)
dataset.head(3)

dataset[['employment_type', 'salary']].groupby(['employment_type']).mean()

employment_map = {'govt' : 0, 'private' : 1, 'self_employed' : 2, 'without_pay' : 3}
dataset['employment_type'] = dataset['employment_type'].map(employment_map)
dataset.head(3)

'''
# plot to show but freezes the console
x = dataset['education']
plt.hist(x, bins = None)
plt.show()
'''

dataset[['education', 'salary']].groupby(['education']).mean() * 100
dataset.drop(labels = ['education', 'occupation', 'workclass'], axis = 1, inplace = True)
dataset.head(3)

x = dataset['capital-gain']
plt.hist(x, bins = None, normed = None)
plt.show()

dataset.loc[(dataset['capital-gain'] > 0), 'capital-gain'] = 1
dataset.loc[(dataset['capital-gain'] == 0) , 'capital-gain'] = 0
dataset.head(3)

x = dataset['capital-loss']
plt.hist(x, bins = None)
plt.show()

dataset.loc[(dataset['capital-loss'] > 0),'capital-loss'] = 1
dataset.loc[(dataset['capital-loss'] == 0 ,'capital-loss')] = 0
dataset.head(3)

dataset['age'].count()

X = dataset.drop(['salary'], axis = 1)
y = dataset['salary']

# Creation of Train and Test dataset
split_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_size, random_state = 22)

# Creation of Train and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = split_size, random_state = 5)

print("Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
print("Validation dataset: {0}{1}".format(X_val.shape, y_val.shape))
print("Test dataset: {0}{1}".format(X_test.shape, y_test.shape))

# probing for classification algorithms with better accuracy
names = ['LR', 'Random Forest', 'Neural Network', 'GaussianNB', 'DecisionTreeClassifier', 'XGBoost']
models = []
models.append(LogisticRegression(solver = 'liblinear'))
models.append(RandomForestClassifier(n_estimators = 100))
models.append(MLPClassifier())
models.append(GaussianNB())
models.append(DecisionTreeClassifier())
models.append(xgb())
models

kfold = model_selection.KFold(n_splits = 5)

for i in range(0, len(models)):
    cv_result = model_selection.cross_val_score(models[i], X_train, y_train, cv = kfold, scoring = 'accuracy')
    score = models[i].fit(X_train, y_train)
    prediction = models[i].predict(X_val)
    acc_score = accuracy_score(y_val, prediction)
    print('-' * 40)
    print('{0}: {1}'.format(names[i], acc_score))

# predict our test data and see prediction results
randomForest = RandomForestClassifier(n_estimators = 100)
randomForest.fit(X_train, y_train)
prediction = randomForest.predict(X_test)

print('-' * 40)
print('Accuracy score:')
print(accuracy_score(y_test, prediction))
print('-' * 40)
print('Confusion Matrix:')
print(confusion_matrix(y_test, prediction))
print('-' * 40)
print('Classification Matrix:')
print(classification_report(y_test, prediction))

'''Using Random Forest for important features''' # Taken from EDA Wine, make required changes

# Using in-built feature_importance function. Convert the data into numeric by OneHotEncoding
model = RandomForestRegressor(random_state = 1, max_depth = 10)
dataset = pd.get_dummies(dataset) # use drop_first = True
model.fit(X_train, Y_train)

# After fitting the model, plot the feature importance graph
features = dataset.columns
importances = model.feature_importances_

# top 6 features
indices = np.argsort(importances)[-6:]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color = 'b', align = 'center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Naive Bayes classifier
NBclassifier = GaussianNB()
NBclassifier.fit(X_train, Y_train)
nb_yPred = NBclassifier.predict(X_test)

# accuracy of the classification
accuracy_nb = accuracy_score(nb_yPred, Y_test) * 100
confusion_matrix(nb_yPred, Y_test)
print(accuracy_nb)

'''Using XGBoost classifier for important features'''
xgbclassifier = XGBClassifier()
xgb_yPred = xgbclassifier.fit(X_train, Y_train).predict(X_test)
accuracy_xgb = accuracy_score(xgb_yPred, Y_test)
confusion_matrix(xgb_yPred, Y_test)
print(accuracy_xgb)

# After fitting the model,plot histogram feature importance graph
fig, ax = plt.subplots(figsize = (10, 4))
plot_importance(xgbclassifier, ax = ax)

# Marginal plot allows to study the relationship between 2 numeric variables. The central chart display their correlation
sns.set(style = "white", color_codes = True) #Not working, need to probe~~~~~
sns.jointplot(x = x['alcohol'], y = y, kind = 'kde', color = 'skyblue')

'''''''''''Adult'''''''''''

# https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset

adults = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\adult.csv')
adults.sample(5)
adults.columns
adults.apply(np.max)
adults['sex'].value_counts()
adults.loc[adults['sex'] == 'Female', 'age'].mean()
adults[adults['country'] == ' Germany'].count() / adults.shape[0]
age1 = adults.loc[adults['salary'] == ' <=50K', 'age']
age2 = adults.loc[adults['salary'] == ' >50K', 'age']
round(np.mean(age1)), round(np.nanstd(age1), 2), round(np.mean(age2)), round(np.nanstd(age2), 2)

high_school = [1, 3, 6]
above_high_school = ' Bachelors', ' Masters', ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' Doctorate', ' Prof-school'
adults.loc[adults['salary'] == ' >50K', 'education'].unique()
adults['age'].describe().T
stats = adults[['race', 'sex', 'age']].groupby(['race', 'sex']).describe().T

adults['marital-status'].replace([' Never-married', ' Divorced', ' Separated', ' Widowed'], 'bachelors', inplace = True)
adults['marital-status'] = adults['marital-status'].replace([' Married-civ-spouse', ' Married-AF-spouse', ' Married-spouse-absent'], 'married')

# Among whom is the proportion of those who earn a lot (>50K) greater: married or single men
adults.loc[(adults['sex'] == ' Male') & (adults['marital-status'].isin([' Never-married', ' Separated', ' Divorced', ' Widowed'])), 'salary'].value_counts()
adults.loc[(adults['sex'] == ' Male') & (adults['marital-status'].str.startswith(' Married')), 'salary'].value_counts()
adults['marital-status'].value_counts()

max_load = adults['hours-per-week'].max()
no_of_workaholics = adults[adults['hours-per-week'] == max_load].shape[0]
round(adults[(adults['salary'] == ' >50K') & (adults['hours-per-week'] == max_load)].shape[0] / no_of_workaholics * 100, 2)

for (country, salary), sub_df in adults.groupby(['country', 'salary']):
    print(country, salary, round(sub_df['hours-per-week'].mean(), 2))

sal_per_week = pd.crosstab(adults['country'], adults['salary'], values = adults['hours-per-week'], aggfunc = np.mean).T

adults['country'].unique()
adults['country'].count()
adults['country'].str.startswith(' H').value_counts() # Find specific values in a column using str.startwith
adults[adults['country'].str.startswith(' H')]

'''Cross tab/Contingency table: To see how observations in a sample are distributed in the context of two variables

Syntax:
pd.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False,
margins_name='All', dropna=True, normalize=False)'''

df = pd.read_excel('D:\Programming Tutorials\Machine Learning\Projects\Datasets\survey.xls')
pd.crosstab(df['Nationality'], df['Handedness'])
pd.crosstab(df['Handedness'], df['Nationality'], margins = True, margins_name = 'Total') # for total
pd.crosstab(df['Sex'], [df['Handedness'], df['Nationality']], margins = True)
pd.crosstab([df['Nationality'], df['Sex']], [df['Handedness']], margins = True)
pd.crosstab(df['Nationality'], df['Handedness'], normalize = 'index') # normalizes over each row values
pd.crosstab(df['Nationality'], df['Handedness'], normalize = 'columns') # normalizes over each column values
pd.crosstab(df['Nationality'], df['Handedness'], normalize = 'index', margins = True) # normalizes over each row values and margin values
pd.crosstab(df['Nationality'], df['Handedness'], values = df['Age'], aggfunc = np.mean)


'''''''''''''''''''''''''''''''''''''''''EDA Back Pain'''''''''''''''''''''''''''''''''''''''''''''''''''

'''Ref: https://towardsdatascience.com/an-exploratory-data-analysis-on-lower-back-pain-6283d0b0123
https://www.kaggle.com/nasirislamsujan/exploratory-data-analysis-lower-back-pain?scriptVersionId=5589885'''

#importing dataset
dataset = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Dataset_spine.csv')
dataset.head(3)
#deleting unnecessary column
del dataset['Unnamed: 13']
#change the Column names. Same job done by below method as list
'''1st way to rename the columns, no need to follow any order since they are dict entries'''
dataset.rename(columns = {"Col1" : "pelvic_incidence", "Col2" : "pelvic_tilt", "Col3" : "lumbar_lordosis_angle",
                          "Col4" : "sacral_slope", "Col5" : "pelvic_radius", "Col6" : "degree_spondylolisthesis",
                          "Col7" : "pelvic_slope", "Col8" : "direct_tilt", "Col9" : "thoracic_slope",
                          "Col10" : "cervical_tilt", "Col11" : "sacrum_angle", "Col12" : "scoliosis_slope",
                          "Class_att" : "class"}, inplace = True)

'''2nd way to rename the columns, order needs to be followed since is a list of titles'''
dataset.columns = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius',
                   'degree_spondylolisthesis', 'pelvic_slope', 'direct_tilt','thoracic_slope','cervical_tilt',
                   'sacrum_angle', 'scoliosis_slope', 'class']

'''Summary of the dataset, gives descriptive statistics with the central tendency, dispersion and shape
of dataset distribution excluding NaN values.Works with numeric values but not categorical values'''
dataset.shape
dataset.describe()
dataset.info()
# checking for missing values in each feature column, unblock for % count
dataset.isnull().sum()#/len(df)*100
#shows count as barchart w.r.t specified class, checks for imbalanced dataset
dataset['class'].value_counts().plot.bar()
# 1st way to find correlation between features as heatmap
plot.subplots(figsize = (10, 6)) #doesnt clutter unlike plot.figure(figsize=(10,6))
sns.heatmap(dataset.corr(), annot = True, cmap = 'viridis')
# 2nd way to find correlation between features as heatmap that gives only 1 diagonal
mask = np.array(dataset.corr())
mask[np.tril_indices_from(mask)] = False
fig, ax = plot.subplots(figsize = (10, 8))
sns.heatmap(dataset.corr(), vmax = .8, square = True, annot = True, cmap = 'viridis', mask = mask)\
# 3rd way of custom correlation between each pair of features w.r.t output
sns.pairplot(dataset, hue = 'class')
# Histogram of distribution of each feature
dataset.hist(figsize = (10, 12), bins = 20, color = '#007959AA')
plot.title("Features Distribution")
plot.show()

'''1st way of boxplot for outliers'''
plot.subplots(figsize = (15, 6))
dataset.boxplot(patch_artist = True, sym = "k.")
plot.xticks(rotation = 45)

'''2nd way of boxplot for outliers'''
# Label encoding of the output variable. Algorithms like XGBoost takes numerical values
encoder = LabelEncoder()
dataset['class'] = encoder.fit_transform(dataset['class'])

# boxplot for outliers
feature_space = dataset.columns.values
number_of_columns = 12
number_of_rows = len(feature_space) - 1 / number_of_columns
plot.figure(figsize = (number_of_columns, 5 * number_of_rows))
for i in range(0, len(feature_space)):
    plot.subplot(number_of_rows + 1, number_of_columns, i + 1)
    sns.set_style('whitegrid')
    sns.boxplot(dataset[feature_space[i]], color = 'green', orient = 'v')
    plot.tight_layout()

# To check distribution-Skewness
plot.figure(figsize = (2 * number_of_columns, 5 * number_of_rows))
for k in range(0, len(feature_space)):
    plot.subplot(number_of_rows + 1, number_of_columns, k + 1)
    sns.distplot(dataset[feature_space[k]], kde = True)

# Visualization with barplot and normal distribution plot
for j, features in enumerate(list(dataset.columns)[:-1]):
    fg = sns.FacetGrid(dataset, hue = 'class', height = 5)
    fg.map(sns.distplot, features).add_legend()
dataset.pelvic_slope[dataset.scoliosis_slope == 1].median()
sns.boxplot(data = dataset, x = 'class', y = 'pelvic_slope', color = 'g')

'''3rd way to detect & remove outliers by function'''

# Function to detect outliers
minimum = 0
maximum = 0
def detect_outlier(feature):
    first_q = np.percentile(feature, 25)
    third_q = np.percentile(feature, 75)
    IQR = third_q-first_q #IQR is the distance between 3rd Quartile and 1st Qartile
    IQR *= 1.5
    minimum = first_q - IQR #acceptable minimum value
    maximum = third_q + IQR #acceptable maximum value
    flag = False
    if(minimum > np.min(feature)):
        flag = True
    if(maximum < np.max(feature)):
        flag = True
    return flag

# Detecting outliers using above function
X = dataset.iloc[:, :-1] #taking all the columns except the output column
for i in range(len(X.columns)):
    if(detect_outlier(X[X.columns[i]])):
        print('"', X.columns[i], '"', 'contains Outliers!')

# Function to remove outliers
def remove_outlier(feature): #use tukey method to remove outliers. whiskers are set at 1.5 times IQR
    first_q = np.percentile(X[feature], 25)
    third_q = np.percentile(X[feature], 75)
    IQR = third_q-first_q
    IQR *= 1.5
    minimum = first_q - IQR #acceptable minimum value
    maximum = third_q + IQR #acceptable maximum value
    median = X[feature].median()
    #values beyond the acceptance range are considered outliers. replace them with median of that feature
    X.loc[X[feature] < minimum, feature] = median
    X.loc[X[feature] > maximum, feature] = median

# Removing outliers
for i in range(len(X.columns)):
    for i in range(len(X.columns)):
        remove_outlier(X.columns[i])

'''Re-checking using the same outlier detection methods above'''

# 1st way of boxplot after removing outliers to verify
plot.subplots(figsize = (15, 6))
X.boxplot(patch_artist = True, sym = "k.")
plot.xticks(rotation = 45)

# 2nd way of boxplot for outliers
plot.figure(figsize = (number_of_columns, 5 * number_of_rows))
for i in range(0, len(feature_space)):
    plot.subplot(number_of_rows + 1, number_of_columns, i + 1)
    sns.set_style('whitegrid')
    sns.boxplot(dataset[feature_space[i]], color = 'green', orient = 'v')
    plot.tight_layout()

# To check distribution-Skewness
plot.figure(figsize = (2 * number_of_columns, 5 * number_of_rows))
for k in range(0, len(feature_space)):
    plot.subplot(number_of_rows + 1, number_of_columns, k + 1)
    sns.distplot(dataset[feature_space[k]], kde = True)

# Visualization with barplot and normal distribution plot
for j, features in enumerate(list(dataset.columns)[:-1]):
    fg = sns.FacetGrid(dataset, hue = 'class', height = 5)
    fg.map(sns.distplot, features).add_legend()
dataset.pelvic_slope[dataset.scoliosis_slope == 1].median()
sns.boxplot(data = dataset, x = 'class', y = 'pelvic_slope', color = 'g')

'''Recheck complete'''

# Feature Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)

# converting the scaled data into pandas dataframe
scaled_dataset = pd.DataFrame(data = scaled_data, columns = X.columns)
scaled_dataset.head(3)

# Splitting into training & test dataset
X = scaled_dataset
y = dataset['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# probing for the best classification algorithm using KFold CV
'''
models = []
models.append(LogisticRegression(solver = 'liblinear'))
models.append(RandomForestClassifier(n_estimators = 100))
models.append(MLPClassifier())
models.append(GaussianNB())
models.append(DecisionTreeClassifier())
models.append(xgb())
names = ['Logistic Regression', 'Random Forest', 'ANN', 'Gaussian NB', 'DecisionTree Classifier',
         'XGBClassifier']
models
'''
# Or
models = [LogisticRegression(solver = 'liblinear'), RandomForestClassifier(n_estimators = 100), MLPClassifier(),
          GaussianNB(), DecisionTreeClassifier(), xgb()]
models
names = ['Logistic Regression', 'Random Forest', 'ANN', 'Gaussian NB', 'DecisionTree Classifier',
         'XGBClassifier']

kfold = model_selection.KFold(n_splits = 5, random_state = 7)

for i in range(0, len(models)):
    cv_result = model_selection.cross_val_score(models[i], X_train, y_train, cv = kfold, scoring = 'accuracy')
    score = models[i].fit(X_train, y_train)
    prediction = models[i].predict(X_test)
    acc_score = accuracy_score(y_test, prediction)
    print ('-' * 40)
    print ('{0}: {1}'.format(names[i], acc_score))

'''Fitting dataset to the appropriate ML model to predict & compare with test data as per above accuracy'''

# Random Forest classifier
randomForest = RandomForestClassifier(n_estimators = 100)
y_pred_rf = randomForest.fit(X_train, y_train).predict(X_test)
print ('-' * 40)
print ('Accuracy score:')
print (accuracy_score(y_test, y_pred_rf))
print ('-' * 40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test, y_pred_rf))
print ('-' * 40)
print ('Classification Matrix:')
print (classification_report(y_test, y_pred_rf))

# Naive Bayes classifier
classifier_gnb = GaussianNB()
y_pred_gnb = classifier_gnb.fit(X_train, y_train).predict(X_test)
# accuracy of the classification
accuracy_score(y_test, y_pred_gnb)
confusion_matrix(y_test, y_pred_gnb)

# MLP classifier
classifier_mlp = MLPClassifier()
y_pred_mlp = classifier_mlp.fit(X_train, y_train).predict(X_test)
# accuracy of the classification
accuracy_score(y_test, y_pred_mlp)
confusion_matrix(y_test, y_pred_mlp)

# SVM classifier
classifier_svc = SVC(kernel = 'linear')
y_pred_svc = classifier_svc.fit(X_train, y_train).predict(X_test)
# accuracy of the classification
accuracy_score(y_test, y_pred_svc)
confusion_matrix(y_test, y_pred_svc)

# XGBoost classifier
classifier_xgb = XGBClassifier()
y_pred_xgb = classifier_xgb.fit(X_train, y_train).predict(X_test)
# accuracy of the classification
accuracy_score(y_test, y_pred_xgb)
confusion_matrix(y_test, y_pred_xgb)

'''Tuning for optimal hyper parameters using Grid Search '''

# probing optimal batch size
batch_Size = [8, 16, 32, 50, 64, 100, 128]
# probing optimal no of epochs
epochs = [10, 50, 100, 150, 200]
# probing for best optimizer
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# probing for optimizer learning rate
learn_rate = [0.001, 0.01, 0.1, 0.2 ,0.3]
# probing for momentum
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# probing for weight initialization mode
initialization = ['normal', 'zero', 'uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform',
                  'lecun_uniform']
# probing for optimal activation
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# dropout is best combined with a weight constraint such as the max norm constraint
weights = [1, 2, 3, 4, 5]
# probing for best dropout rate
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# probing for no of neurons in hidden layers
no_of_neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(batch_size = batch_Size, epochs = epochs, optimizer = optimizers, learn_rate = learn_rate,
                  momentum = momentum, init = initialization, activation = activation, weight_constraint = weights,
                  dropout_rate = dropout_rate, neurons = no_of_neurons)
grid = GridSearchCV(estimator = MLPClassifier(), param_grid = param_grid, n_jobs = -1)
gSearch = grid.fit(X, y)
best_params = gSearch.best_params_
best_accuracy = gSearch.best_score_

# summarize results
print("Best score: %f using params %s" % (gSearch.best_score_, gSearch.best_params_))
means = gSearch.cv_results_['mean_test_score']
stds = gSearch.cv_results_['std_test_score']
params = gSearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Histogram for important features
fig = plot.subplots(figsize = (10, 4))
plot_importance(classifier_xgb)

# Marginal plot to study relationship between 2 numeric variables. Central chart display their correlation
sns.set(style = 'white', color_codes = True)
sns.jointplot(x = X['pelvic_slope'], y = y, kind = 'kde', color = 'skyblue')

'''Using Random Forest for important features''' # Taken from EDA Wine, make required changes

# Using in-built feature_importance function.Convert the data into numeric by OneHotEncoding
model = RandomForestRegressor(random_state = 1, max_depth = 10)
dataset = pd.get_dummies(dataset) # use drop_first = True
model.fit(X_train, Y_train)

# After fitting the model,plot the feature importance graph
features = dataset.columns
importances = model.feature_importances_

# top 6 features
indices = np.argsort(importances)[-6:]
plot.title('Feature Importances')
plot.barh(range(len(indices)), importances[indices], color = 'b', align = 'center')
plot.yticks(range(len(indices)), [features[i] for i in indices])
plot.xlabel('Relative Importance')
plot.show()

# Naive Bayes classifier
NBclassifier = GaussianNB()
NBclassifier.fit(X_train, Y_train)
nb_yPred = NBclassifier.predict(X_test)

# accuracy of the classification
accuracy_nb = accuracy_score(nb_yPred, Y_test) * 100
confusion_matrix(nb_yPred, Y_test)
print(accuracy_nb)

'''Using XGBoost classifier for important features'''
xgbclassifier = xgb()
xgb_yPred = xgbclassifier.fit(X_train, Y_train).predict(X_test)
accuracy_xgb = accuracy_score(xgb_yPred, Y_test)
confusion_matrix(xgb_yPred, Y_test)
print(accuracy_xgb)

# After fitting the model,plot histogram feature importance graph
fig, ax = plot.subplots(figsize = (10, 4))
plot_importance(xgbclassifier, ax = ax)

# Marginal plot allows to study the relationship between 2 numeric variables. The central chart display their correlation
sns.set(style = "white", color_codes = True) #Not working, need to probe~~~~~
sns.jointplot(x = x['alcohol'], y = y, kind = 'kde', color = 'skyblue')


'''''''''''''''''''''''''''''''''''''''''EDA Breast Cancer'''''''''''''''''''''''''''''''''''''''''''''''

'''Ref: https://github.com/bacemtayeb/EDA/blob/master/Haberman.ipynb'''

# Loading the dataset
dataset = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\breast_cancer.csv')
# lists the column names
dataset.columns
# for list view
list(dataset.columns)
# rename the columns
dataset.columns = ['Age', 'Operation_Year', 'Axil_nodes', 'Surv_status']
# view top 3 rows
dataset.head(3)
# view bottom 3 rows
dataset.tail(3)
# Useful statistical insights, descriptive statistics
dataset.describe()
# A quick look at data types of features
dataset.info()
# 1st way of checking for missing values, could also be checked through the count row
dataset.isna().sum()#/len(df)*100
# 2nd way of checking for missing values in each feature column, unblock for % count
dataset.isnull().sum()#/len(df)*100
# Mapping numerical categories with standard categories. Redundant in this context
dataset.Surv_status = dataset.Surv_status.map({1 : 1, 2 : 0})
'''mapping standard categories with string categories and adding the data frame. However this gives error
 in the 2nd way of box plot since the column contains strings and need to be converted to numeric type'''
# dataset['Status'] = dataset.Surv_status.map({1 : 'Alive', 2 : 'Dead'})

# verify if the output column is updated with the mapping
dataset.head()
# 1st way to show count w.r.t specified class, checks for imbalanced dataset. Unblock for bar plot visualization
dataset.Surv_status.value_counts()#.plot.bar()
# 2nd way to show the unique values as an array
dataset.Surv_status.unique()
# multi collinearity check
dataset.corr(method = 'pearson')
# 1st way for correlation for multi correlation between features as heatmap
plot.figure(figsize = (4, 3))
sns.heatmap(dataset.corr(method = 'pearson'), annot = True)
#2nd way for correlation between features as heatmap that gives only 1 diagonal
mask = np.array(dataset.corr())
mask[np.tril_indices_from(mask)] = False
fig, ax = plot.subplots(figsize = (4, 3))
sns.heatmap(dataset.corr(), mask = mask, vmax = .8, square = True, annot = True, cmap = 'viridis')
# Custom correlation between each pair of features w.r.t output
sns.pairplot(dataset, hue = 'Surv_status')
# Histogram of distribution of each feature
dataset.hist(figsize = (10, 12), bins = 50, color = '#007959AA')
plot.title("Features Distribution")
# 1st way of boxplot for outliers method 1
plot.subplots(figsize = (6, 4))
dataset.boxplot(patch_artist = True, sym = "k.")
plot.xticks(rotation = 90)
# 2nd way of boxplot for outliers
features = dataset.columns
number_of_columns = len(features)
number_of_rows = number_of_columns - 1 / number_of_columns
plot.figure(figsize = (number_of_columns, 5 * number_of_rows))
for i in range(0, number_of_columns):
    plot.subplot(number_of_rows + 1, number_of_columns, i + 1)
    #sns.set_style('whitegrid')
    sns.boxplot(dataset[features[i]], color = 'blue', orient = 'v')
    plot.tight_layout()

# To check distribution-Skewness
plot.figure(figsize = (2 * number_of_columns, 5 * number_of_rows))
for i in range(0, number_of_columns):
    plot.subplot(number_of_rows + 1, number_of_columns, i + 1)
    sns.distplot(dataset[features[i]], kde = True)

# Visualization with barplot and normal distribution plot
for i, features in enumerate(list(dataset.columns)[:-1]):
    fg = sns.FacetGrid(dataset, hue = 'Surv_status', height = 5)
    fg.map(sns.distplot, features).add_legend()

# to verify what the graphs above showed
dataset.Operation_Year[dataset.Surv_status == 1].median()
# to know the impact of the number of positive axillary nodes detected and the patient status
sns.boxplot(data = dataset, x = 'Surv_status', y = 'Axil_nodes', color = 'g')
# select the feature columns
x = dataset.iloc[:, :3].values
# select the target column
y = dataset.iloc[:, 3].values
# OR
y = dataset['Surv_status']
# Splitting dataset
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size = 0.30, random_state = 10)
# Naive Bayes classifier
classifier_gnb = GaussianNB()
y_pred_gnb = classifier_gnb.fit(X_train, Y_train).predict(X_test)
# accuracy of the classification
acc = accuracy_score(y_pred_gnb, Y_test)
confusion_matrix(y_pred_gnb, Y_test)
print(acc)
# XGBoost classifier
classifier_xgb = xgb()
y_pred_xgb = classifier_xgb.fit(X_train, Y_train).predict(X_test)
acc = accuracy_score(y_pred_xgb, Y_test)
confusion_matrix(y_pred_xgb, Y_test)
print(acc)
# Histogram for important features
fig, ax = plot.subplots(figsize = (8, 3))
plot_importance(classifier_xgb, ax = ax)

# Marginal plot allows to study the relationship between 2 numeric variables. The central chart display their correlation
sns.set(style = "white", color_codes = True) #Not working, need to probe~~~~~
sns.jointplot(x = x["Operation_Year"], y = y, kind = 'kde', color = "skyblue")


'''''''''''''''''''''''''''''''''''''''''EDA Wine'''''''''''''''''''''''''''''''''''''''''''''''

'''Ref- http://nbviewer.jupyter.org/github/PBPatil/Exploratory_Data_Analysis-Wine_Quality_Dataset/blob/master/winequality_white.ipynb
Ref- https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15'''

# Import dataset
dataset = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\winequality-white.csv', sep = ';')
# Finding the null/Nan values in the columns 1st way
dataset.isnull().sum()#/len(df)*100 #checking for missing values in each feature column, unblock for % count
# Finding the null/Nan values in the columns 2nd way
for i in dataset.columns:
    print("Null values in {} = {}".format(i, dataset[i].isnull().sum()))
# Visualizing the null/Nan values in the columns 1st way. If there were,it'll show different shade on color background
sns.heatmap(dataset.isnull(), cbar = True, yticklabels = False, cmap = 'viridis')
# returns first five observations of the data set
dataset.head(3)
# To find matrix dimension of features of the dataset
dataset.shape
# info() gives data types of columns and if there are null values
dataset.info()
# gives summary of the data statistics like count, mean, stddev, min & max values, quantiles
dataset.describe()
# insights of categories of dependent variable
dataset.quality.unique()
# shows count w.r.t specified class, checks for imbalanced dataset. Unblock for bar plot visualization
dataset.quality.value_counts()#.plot.bar()
# Visualizing as histogram for dataset imbalance
pd.value_counts(dataset['quality']).plot.bar()
plot.title('Wine Quality histogram')
plot.xlabel('Quality Classes')
plot.ylabel('Frequency')
# can be ignored, shows the unique values as an array
uniq_vals = np.unique(dataset)
'''1st way of correlation matrix w.r.t no specific feature using Pearson by default, so no need to mention'''
plot.figure(figsize = (10, 6))
sns.heatmap(dataset.corr(method = 'pearson'), cmap = 'viridis', annot = True)
'''2nd way of multi correlation between features as heatmap that gives only 1 diagonal'''
mask = np.array(dataset.corr())
mask[np.tril_indices_from(mask)] = False
fig,ax = plot.subplots(figsize = (10, 6))
sns.heatmap(dataset.corr(), mask = mask, vmax = .8, square = True, annot = True, cmap = 'viridis')
'''3rd way of multi correlation matrix heatmap w.r.t specific feature "Quality"'''
# number of features for heatmap
k = 12
#gives correlation using Pearson by default
cols = dataset.corr(method = 'pearson').nlargest(k, 'quality')['quality'].index
corr_matrix = dataset[cols].corr()
plot.figure(figsize = (10, 6))
sns.heatmap(corr_matrix, cmap = 'viridis', annot = True)
#Custom correlogram between each pair of features w.r.t output
sns.pairplot(dataset, hue = 'quality')
#Histogram of distribution of each feature
dataset.hist(figsize = (10, 12), bins = 20, color = "#007959AA")
plot.title("Features Distribution")
plot.show()

'''Plots for outliers and distribution skewness'''
#Boxplot for outliers method 1
#plot.subplots(figsize=(15,6)) #unblock for bigger plot dimensions
dataset.boxplot(patch_artist = True, sym = "k.")
plot.xticks(rotation = 90)

# Boxplot for outliers method 2
features = dataset.columns.values
number_of_columns = len(dataset.columns)
number_of_rows = len(features) - 1 / number_of_columns
plot.figure(figsize = (number_of_columns, 5 * number_of_rows))
for i in range(0, len(features)):
    plot.subplot(number_of_rows + 1, number_of_columns, i + 1)
    sns.set_style('whitegrid')
    sns.boxplot(dataset[features[i]], color = 'green', orient = 'v')
    plot.tight_layout()

# To check distribution-Skewness
plot.figure(figsize = (2 * number_of_columns, 5 * number_of_rows))
for i in range(0, len(features)):
    plot.subplot(number_of_rows + 1, number_of_columns, i + 1)
    sns.distplot(dataset[features[i]], kde = True)

# Visualization with barplot and normal distribution plot
for i,features in enumerate(list(dataset.columns)[:-1]):
    fg = sns.FacetGrid(dataset, hue = 'quality', height = 12)
    fg.map(sns.distplot, features).add_legend()
dataset.alcohol[dataset.quality == 0].median() # to verify what the graphs above showed
sns.boxplot(data = dataset, x = 'quality', y = 'alcohol', color = 'g') #to know the impact of the no of positive alcohol nodes detected and the patient status

# select the feature columns & the target column
dataset.shape
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x.shape
y.shape

# Splitting dataset
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size = 0.8, random_state = 10)

'''Using Random Forest for important features'''

# Using in-built feature_importance function.Convert the data into numeric by OneHotEncoding
model = RandomForestRegressor(random_state = 1, max_depth = 10)
dataset = pd.get_dummies(dataset) # use drop_first = True
model.fit(X_train, Y_train)

# After fitting the model,plot the feature importance graph
features = dataset.columns
importances = model.feature_importances_

# top 6 features
indices = np.argsort(importances)[-6:]
plot.title('Feature Importances')
plot.barh(range(len(indices)), importances[indices], color = 'b', align = 'center')
plot.yticks(range(len(indices)), [features[i] for i in indices])
plot.xlabel('Relative Importance')
plot.show()

# Naive Bayes classifier
NBclassifier = GaussianNB()
NBclassifier.fit(X_train, Y_train)
nb_yPred = NBclassifier.predict(X_test)

# accuracy of the classification
accuracy_nb = accuracy_score(nb_yPred, Y_test) * 100
confusion_matrix(nb_yPred, Y_test)
print(accuracy_nb)

'''Using XGBoost classifier for important features'''
xgbclassifier = xgb()
xgb_yPred = xgbclassifier.fit(X_train, Y_train).predict(X_test)
accuracy_xgb = accuracy_score(xgb_yPred, Y_test)
confusion_matrix(xgb_yPred, Y_test)
print(accuracy_xgb)

# After fitting the model,plot histogram feature importance graph
fig, ax = plot.subplots(figsize = (10, 4))
plot_importance(xgbclassifier, ax = ax)

# Marginal plot allows to study the relationship between 2 numeric variables. The central chart display their correlation
sns.set(style = "white", color_codes = True) #Not working, need to probe~~~~~
sns.jointplot(x = x['alcohol'], y = y, kind = 'kde', color = 'skyblue')

'''''''''''EDA House Price'''''''''''
# https://www.kaggle.com/pavansanagapati/a-simple-tutorial-on-exploratory-data-analysis
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# https://www.kaggle.com/pavansanagapati/simple-tutorial-how-to-handle-missing-data

url = 'D:\Programming Tutorials\Machine Learning\Projects\Datasets\House Price Adv Regression train.csv'
train = pd.read_csv(url)
test = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\House Price Adv Regression test.csv')
stats = train.describe().T
train.info()
train.head()
train.sample(5, random_state = 4)
train.shape
train.columns.tolist()
len(train.columns)
list(set(train.dtypes.tolist()))
numeric_features = train.select_dtypes(['int64', 'float64'])
numeric_features.columns
categorical_features = train.select_dtypes(include = [np.object])
categorical_features.columns
'''deleting irrelevant features'''
del(train['Id'])
# finding columns that have values less than 15%
consolidated_train = train[[cols for cols in train if train[cols].count() / len(train.index) >= .85]]
# eliminating the features having less than 30% values
print('Eliminated columns are: ')
for c in train.columns:
    if c not in consolidated_train.columns:
        print(c)
train = consolidated_train
train.columns
# missing data
total = train.isna().sum().sort_values(ascending = False)
percent = (train.isna().sum() / train.isna().count() * 100).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.index.name = 'Feature'
missing_data.head(8)
train = train.drop((missing_data[missing_data['Total'] > 1].index), 1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
# methond 1: n-strongly correlated features
correlation = train.corr(method = 'pearson')
print(correlation['SalePrice'].sort_values(ascending = False))
# correlation heat map
f, ax = plt.subplots(figsize = (16, 16))
plt.title('Correlation of Numeric Features with Sale Price', y = 1, size = 16)
sns.heatmap(correlation, annot = False, square = False, vmax = 0.8)
plt.show()
# zoomed heat map
k = 11
cols = correlation.nlargest(k, 'SalePrice')['SalePrice'].index
print(cols)
cor_map = np.corrcoef(train[cols].values.T)
sns.set(font_scale = 1.25)
plt.subplots(figsize = (14, 12))
sns.heatmap(cor_map, cbar = True, vmax = .8, lw = 0.1, square = True, annot = True, cmap = 'viridis', linecolor = 'w',
            xticklabels = cols.values, yticklabels = cols.values, annot_kws = {'size': 12})
# methond 2: n strongly correlated features
df_num_corr = train.corr(method = 'pearson')['SalePrice'][:-1]
gold_features = df_num_corr[abs(df_num_corr) > .5].sort_values(ascending = False).index
type(gold_features)
# Univariate analysis
skew = train.skew()
kurtosis = train.kurtosis()
# standardizing data
sale_price_scaled = MinMaxScaler().fit_transform(train['SalePrice'][:, np.newaxis])
low_range = sale_price_scaled[sale_price_scaled[:, 0].argsort()][:10]
high_range = sale_price_scaled[sale_price_scaled[:, 0].argsort()][-10:]
print(low_range)
print(high_range)
# probing for normality by skew and kurtosis
train['SalePrice'].skew()
train['SalePrice'].kurtosis()
train['SalePrice'].describe()
# probing for normality by skew and kurtosis. method 1 to verify for normal distribution by few best fit methods
sns.distplot(train['SalePrice'])
plt.figure(1)
plt.title('Johnson SU')
sns.distplot(train['SalePrice'], kde = False, fit = st.johnsonsu)
plt.figure(2)
plt.title('Normal')
sns.distplot(train['SalePrice'], kde = False, fit = st.norm)
plt.figure(3)
plt.title('Log Normal')
sns.distplot(train['SalePrice'], kde = False, fit = st.lognorm)
# SalePrice  has to be transformed as it doesn't follow normal distribution, best fit is unbounded Johnson dist
plt.figure(figsize = (10, 8))
sns.distplot(skew, color = 'green', axlabel = 'skewness')
plt.figure(figsize = (12, 8))
sns.distplot(kurtosis, color = 'r', axlabel = 'kurtosis', norm_hist = False, kde = True, rug = False)
plt.show()
plt.hist(x = train['SalePrice'], orientation = 'vertical', histtype = 'bar', color = 'b', bins = 30)
plt.show()
# probing for normality by skew and kurtosis. method 2 to verify normal distribution by histogram and normal probability plot
sns.distplot(train['SalePrice'], fit = norm, kde = True)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
# applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])
# verifying again with above normal dist and prob plots
# method 1
sns.distplot(train['SalePrice'], fit = norm, kde = True)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
# method 2
plt.hist(train['SalePrice'], color = 'b')
# probing for normality of GrLivArea
sns.distplot(train['GrLivArea'], fit = norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot = plt)
# applying log transformation
train['GrLivArea'] = np.log(train['GrLivArea'])
# verifying again with above normal dist and prob plots
sns.distplot(train['GrLivArea'], fit = norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot = plt)
# probing for normality of TotalBsmtSF
sns.distplot(train['TotalBsmtSF'], fit = norm)
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot = plt)
# create column for new variable (one is enough as it's a binary categorical feature)
train['Has_basement'] = pd.Series(len(train['TotalBsmtSF']), index = train.index)
train['Has_basement'] = 0
train.loc[train['TotalBsmtSF'] > 0, 'Has_basement'] = 1
# log transformation
train.loc[train['Has_basement'] == 1, 'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
# histogram and normal probability plot
sns.distplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit = norm)
fig = plt.figure()
res = stats.probplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot = plt)
# bivariate analysis
feature = 'GrLivArea'
train_feature = pd.concat([train['SalePrice'], train[feature]], axis = 1)
train_feature.plot.scatter(x = feature, y = 'SalePrice', ylim = (0, 800000))
# deleting outliers
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(train[train['GrLivArea'] == 1298].index)
train = train.drop(train[train['GrLivArea'] == 523].index)
train.plot.scatter(x = feature, y = 'SalePrice', ylim = (0, 800000))
# probing for homoscedasticity
plt.scatter(train['GrLivArea'], train['SalePrice'])
# 'SalePrice' with 'TotalBsmtSF'
plt.scatter(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], train[train['TotalBsmtSF'] > 0]['SalePrice'])
feature = 'TotalBsmtSF'
train_feature = pd.concat([train['SalePrice'], train[feature]], axis = 1)
train.plot.scatter(x = feature, y = 'SalePrice', ylim = (0, 800000))
feature = 'YearBuilt'
train_feature = pd.concat([train[feature], train['SalePrice']], axis = 1)
f, ax = plt.subplots(figsize = (16, 8))
fig = sns.boxplot(x = feature, y = 'SalePrice', data = train)
fig.axis(ymin = 0, ymax = 800000)
# Pair plots between SalePrice & correlated variables
sns.set()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(train[columns], kind = 'scatter', height = 2, diag_kind = 'kde')
plt.show()
# scatter plots between the most correlated variables
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows = 3, ncols = 2, figsize = (14, 10))

OverallQual_scatter_plot = pd.concat([train['SalePrice'],train['OverallQual']], axis = 1)
sns.regplot(x = 'OverallQual', y = 'SalePrice', data = OverallQual_scatter_plot,scatter = True, fit_reg = True, ax = ax1)

TotalBsmtSF_scatter_plot = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis = 1)
sns.regplot(x = 'TotalBsmtSF', y = 'SalePrice', data = TotalBsmtSF_scatter_plot, scatter = True, fit_reg = True, ax = ax2)

GrLivArea_scatter_plot = pd.concat([train['SalePrice'], train['GrLivArea']], axis = 1)
sns.regplot(x = 'GrLivArea', y = 'SalePrice', data = GrLivArea_scatter_plot, scatter = True, fit_reg = True, ax = ax3)

GarageArea_scatter_plot = pd.concat([train['SalePrice'], train['GarageArea']], axis = 1)
sns.regplot(x = 'GarageArea', y = 'SalePrice', data = GarageArea_scatter_plot, scatter = True, fit_reg = True, ax = ax4)

FullBath_scatter_plot = pd.concat([train['SalePrice'], train['FullBath']], axis = 1)
sns.regplot(x = 'FullBath', y = 'SalePrice', data = FullBath_scatter_plot, scatter = True, fit_reg = True, ax = ax5)

YearBuilt_scatter_plot = pd.concat([train['SalePrice'], train['YearBuilt']], axis = 1)
sns.regplot(x = 'YearBuilt', y = 'SalePrice', data = YearBuilt_scatter_plot,scatter = True, fit_reg = True, ax = ax6)

YearRemodAdd_scatter_plot = pd.concat([train['SalePrice'], train['YearRemodAdd']], axis = 1)
YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd', 'SalePrice')
# Box plot - OverallQual
feature = 'OverallQual'
train_feature = pd.concat([train[feature], train['SalePrice']], axis = 1)
f, ax = plt.subplots(figsize = (8, 6))
fig = sns.boxplot(x = feature, y = 'SalePrice', data = train)
fig.axis(ymin = 0, ymax = 800000)

saleprice_overall_quality= train.pivot_table(index = 'OverallQual', values = 'SalePrice', aggfunc = np.median)
saleprice_overall_quality.plot(kind = 'bar', color = 'blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.show()
'''Categorical Variables'''
# exploring the categorical variables w.r.t. SalePrice
feature = 'Neighborhood'
train = pd.concat([train['SalePrice'], train[feature]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x = feature, y = 'SalePrice', data = train)
fig.axis(ymin = 0, ymax = 800000);
xt = plt.xticks(rotation = 45)
# PointPlot
plt.figure(figsize = (8, 10))
g1 = sns.pointplot(x = 'Neighborhood', y = 'SalePrice', data = train, hue = 'LotShape')
g1.set_xticklabels(g1.get_xticklabels(), rotation = 90)
g1.set_title('Lotshape Based on Neighborhood', fontsize = 15)
g1.set_xlabel('Neighborhood')
g1.set_ylabel('Sale Price', fontsize = 12)
plt.show()
# count plot
plt.figure(figsize = (12, 6))
sns.countplot(x = 'Neighborhood', data = train)
xt = plt.xticks(rotation = 45)
# SalePrice w.r.t variable values and enumerate them
def boxplot(x, y, **kwargs):
    sns.boxplot(x = x, y = y)
    x = plt.xticks(rotation = 90)
f = pd.melt(train, id_vars = ['SalePrice'], value_vars = categorical_features.columns)
g = sns.FacetGrid(f, col = 'variable', col_wrap = 2, sharex = False, sharey = False, height = 5)
g = g.map(boxplot, 'value', 'SalePrice')
# Housing Price vs Sales - 1: Sale Type & Sale Condition 2: Sales Seasonality
feature = 'SaleType'
train = pd.concat([train['SalePrice'], train[feature]], axis = 1)
f, ax = plt.subplots(figsize = (16, 10))
fig = sns.boxplot(x = feature, y = "SalePrice", data = train)
fig.axis(ymin = 0, ymax = 800000);
xt = plt.xticks(rotation = 45)
# box plot SaleCondition vs SalePrice
feature = 'SaleCondition'
train = pd.concat([train['SalePrice'], train[feature]], axis = 1)
f, ax = plt.subplots(figsize = (16, 10))
fig = sns.boxplot(x = feature, y = 'SalePrice', data = train)
fig.axis(ymin = 0, ymax = 800000)
xt = plt.xticks(rotation = 45)
# ViolinPlot - Functional vs.SalePrice
sns.violinplot('Functional', 'SalePrice', data = train)
# FactorPlot - FirePlaceQC vs. SalePrice
sns.factorplot('FireplaceQu', 'SalePrice', data = train, color = 'm', estimator = np.median, order = ['Ex', 'Gd', 'TA', 'Fa', 'Po'], size = 4.5,  aspect = 1.35)
# Facet Grid Plot - FirePlace QC vs.SalePrice
g = sns.FacetGrid(train, col = 'FireplaceQu', col_wrap = 3, col_order = ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
g.map(sns.boxplot, 'Fireplaces', 'SalePrice', order = [1, 2, 3], palette = 'Set2')

'''''''''''Telecom Churn'''''''''''

# https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas

churn = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\churn_telecom.csv')
churn.sample(frac = .003, random_state = 3) # looking for sample df
churn.shape # probing for dataset size by rows n cols
churn.columns # looking for col names to rename if needed
churn.info()
numerical_cats = churn.select_dtypes(['int64', 'float64']) # probing for numerical cols
n_stats = numerical_cats.describe().T # descriptive statistics of the numerical cols
categorical_cats = churn.select_dtypes(['object', 'bool']) # probing for categorical cols OR
c_stats = categorical_cats.describe() # descriptive statistics of the categorical cols
c_stats1 = churn.describe(include = ['object', 'bool'])
churn['State'].unique()
pd.get_dummies(churn['State']).shape
churn['Churn'].value_counts() # probing if the dataset is imbalanced
round(churn['Churn'].value_counts(normalize = True) * 100, 2) # to calculate fractions, pass normalize=True
churn.sort_values(by = 'Number vmail messages', ascending = True)
churn.sort_values(by = ['Number vmail messages', 'Customer service calls'], ascending = [True, False])
churn[churn['Churn']].mean()
churn[churn['Churn'] == 1].mean()
churn[churn['Churn'] == 1]['Total day minutes'].mean()
churn[(churn['Churn'] == 0) & (churn['International plan'] == 'No')]['Total intl minutes'].max()
slice = churn.loc[0: 5, 'State': 'Area code'] # picking data by index slicing
slice1 = churn.iloc[0: 6, 0: 3]
churn[:-1]
churn[-1:]
max1 = churn[churn['Churn']].max()
max2 = churn.apply(np.max)
churn[churn['State'].apply(lambda x : x[0] == 'W')].head()
churn['International plan']
dicts = {'No': 0, 'Yes': 1}
churn['International plan'].map(dicts)
churn.replace({'Voice mail plan': dicts}).head()

'''''''''''Olympics'''''''''''
'''
Practice:
https://towardsdatascience.com/exploratory-statistical-data-analysis-with-a-real-dataset-using-pandas-208007798b92
https://towardsdatascience.com/olympics-kaggle-dataset-exploratory-analysis-part-2-understanding-sports-4b8d73a8ec30
https://github.com/StrikingLoo/Olympics-analysis-notebook/blob/master/Olympics_2.ipynb
https://www.kaggle.com/marcogdepinto/let-s-discover-more-about-the-olympic-games
'''

athlete = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\athlete_events.csv')
noc = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\noc_regions.csv')
df = athlete
df.head(5)
df.info()
df.columns
df.describe()
noc.head()
numerical = df.select_dtypes(['float64', 'int64'])
numerical_stats = numerical.describe().T
category = df.select_dtypes('object')
categorical_stats = category.describe().T
# missing values in each col
df.isna().sum()
# filtering the column with multiple categories
winners = df[df['Medal'].fillna('None') != 'None'] # first method
winners = df[df['Medal'].isin(['Gold', 'Silver'])] # second method
# percentage of values missing in each column
def nan_percent(df, column):
    rows = df[column].shape[0]
    empty_vals = rows - df[column].count()
    return round((empty_vals / rows) * 100, 2)
for i in list(df):
    print(i + ': ' + str(nan_percent(df, i)) + '%')
# how many unique people actually won a medal since 1900
tot_rows = df.shape[0]
unique_athletes = len(df['Name'].unique())
medal_winners = len(df[df['Medal'].fillna('None') != 'None']['Name'].unique())
print('There are', unique_athletes, 'unique athletes out of which', medal_winners, 'are medal winners')
# how many medals in each medal categories have actually been earned throughout these 120 years?
df[df['Medal'].fillna('None') != 'None']['Medal'].value_counts()
df['Medal'].value_counts().sum()
# each type of medal won by each country
team_medal_count = df.groupby(['Team', 'Medal'])['Medal'].count()
team_medal_count = team_medal_count.reset_index(name = 'count').sort_values(by = 'count', ascending = False)
def country_stats(country):
    return team_medal_count[team_medal_count['Team'] == country]
country_stats('India')
# what was female representation like in the Olympics throughout the century
unique_women = df[df['Sex'] == 'F']['Name'].unique()
unique_men = df[df['Sex'] == 'M']['Name'].unique()
women_medals = df[df['Sex'] == 'F']['Medal'].count()
men_medals = df[df['Sex'] == 'M']['Medal'].count()
print(len(unique_women), 'unique women won', women_medals, 'medals and', len(unique_men), 'unique men won',
      men_medals, 'medals')
df[df['Sex'] == 'F']['Year'].min()
# are women actually growing in participation faster than men or is it because of world population?
f_year_count = df[df['Sex'] == 'F'].groupby('Year').count()['Name']
m_year_count = df[df['Sex'] == 'M'].groupby('Year').count()['Name']
(sns.scatterplot(data = f_year_count), sns.scatterplot(data = m_year_count))
# what sports have the heaviest and tallest players in men
df_men = df[df['Sex'] == 'M']
men_sports_metrics = df_men.groupby('Sport')['Weight', 'Height'].agg(['min', 'max', 'mean'])
men_sports_metrics['Height'].dropna().sort_values(by = 'mean', ascending = False)[:5] # tallest
men_sports_metrics['Weight'].dropna().sort_values(by = 'mean', ascending = False)[:5] # heaviest
# which have the lightest or shortest
men_sports_metrics['Height'].dropna().sort_values(by = 'mean', ascending = True)[:5] # shortest
men_sports_metrics['Weight'].dropna().sort_values(by = 'mean', ascending = True)[:5] # lightest
# visualizing the height & weight distributions by mean
sns.distplot(men_sports_metrics['Weight'].dropna()['mean'], kde = True)
sns.distplot(men_sports_metrics['Height'].dropna()['mean'], kde = True)
# scatter plot of mean in weight for all sports
means = list(men_sports_metrics['Weight'].dropna()['mean'])
sports = list(men_sports_metrics['Weight'].dropna().index)
plot_data = sorted(zip(sports, means), key = lambda x: x[1])
plot_data_dict = {'x': [i for i, _ in enumerate(plot_data)], 'y': [v[1] for i, v in enumerate(plot_data)],
                        'group': [v[0] for i, v in enumerate(plot_data)]}
sns.scatterplot(data = plot_data_dict, x = 'x', y = 'y')
# to do: scatter plot of mean in height for all sports

# what ‘build’ (weight/height) each sport has
mean_heights = men_sports_metrics['Height'].dropna()['mean']
mean_weights = men_sports_metrics['Weight'].dropna()['mean']
avg_build = mean_weights / mean_heights
builds = list(avg_build.sort_values(ascending = True))
plot_dict = {'x': [i for i, _ in enumerate(builds)], 'y': builds}
sns.lineplot(data = plot_dict, x = 'x', y = 'y')
# to do: find the least and most heavily built sports

# Find the min no of new sports have been introduced to the Olympics, and when
sport_min_year = df_men.groupby('Sport')['Year'].agg(['min', 'max'])['min'].sort_values('index')
year_count = Counter(sport_min_year)
year = list(year_count.keys())
new_sports = list(year_count.values())
data = {'x': year, 'y': new_sports}
# shows how many sports were practiced in the Olympics for the first time for each year
sns.scatterplot(data = data, x = 'x', y = 'y')
# to do: Find the most no of new sports introduced to the Olympics, and when

# sports introduced before the year 2000
sport_min_year[sport_min_year < 2000]
# sports introduced after the year 1936
sport_min_year[sport_min_year > 1936]
# joining the dataframes
merged = pd.merge(df, noc, on = 'NOC', how = 'left')
dff = merged
dff.head()
# distribution of the age of gold medalists
golds = dff[dff['Medal'] == 'Gold']
golds.isnull().sum()
golds = golds[np.isfinite(golds['Age'])]
plt.figure(figsize = (12, 10))
sns.countplot(golds['Age'])
# men gold winners who are elder
golds[golds['Age'] > 50]['Sport'].count()
elder_golds = golds[golds['Age'] > 50]['Sport']
elder_golds.value_counts()
sns.countplot(elder_golds)
# Women is summer olmpics
women = merged[(merged['Sex'] == 'F') & (merged['Season'] == 'Summer')]
women.head()
sns.countplot(data = women, x = 'Year')
women[women['Year'] == 1900].head()
women[women['Year'] == 1900]['ID'].count()
# medals per country
golds_per_country = golds['region'].value_counts().reset_index(name = 'Medal').head()
sns.catplot(data = golds_per_country, kind = 'bar', x = 'index', y = 'Medal')
# Disciplines with the greatest number of Gold Medals in USA
golds_usa = golds[golds['NOC'] == 'USA']
golds_usa['Event'].value_counts().reset_index(name = 'Medals').head()
male_basketball_usa_golds = golds_usa[(golds_usa['Sport'] == 'Basketball') & (golds_usa['Sex'] == 'M')].sort_values(['Year'])
male_basketball_usa_golds.groupby(['Year']).first()
# median height n weight of olympic medalists
medalists = dff[dff['Medal'].fillna('None') != 'None']
medalists['Height'].median()
medalists['Weight'].median()
not_null = dff[(dff['Height'].notnull()) & (dff['Weight'].notnull())]
not_null.info()
sns.scatterplot(data = not_null, x = 'Height', y = 'Weight')

Open:
    https://www.kaggle.com/agrawaladitya/step-by-step-data-preprocessing-eda
    https://www.kaggle.com/ajay1216/practical-guide-on-data-preprocessing-in-python