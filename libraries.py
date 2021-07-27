# Libraries of ML/DL

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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve, r2_score, mean_squared_error,
                             mean_absolute_error, log_loss)
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn import datasets
from sklearn.datasets import load_breast_cancer, load_iris, load_digits
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, LabelBinarizer, MultiLabelBinarizer, OrdinalEncoder,
                                   LabelEncoder, OneHotEncoder)
from sklearn.feature_selection import (SelectFromModel, VarianceThreshold, SelectKBest, SelectPercentile, chi2, RFE,
                                       RFECV, f_classif)
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