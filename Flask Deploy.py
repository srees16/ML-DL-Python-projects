'''Flash Basics & Deployment'''

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
request.data: to get the raw data, when the incoming request data contains string. Works when data couldn't be parsed as form data.
To get the raw post body regardless of the content type, use request.get_data(). request.data calls request.get_data(parse_form_data = True)

request.args: for URL query parameters, key/value pairs in URL query string
search = request.args.get("search")

request.form: the key/value pairs from HTML post form
email = request.form.get('email') # use get if the key might not exist
password = request.form['password'] # use indexing if you know the key exists
request.form.getlist('name'): use getlist if the key is sent multiple times and you want a list of values. get only returns the first value

request.values: combined args and form, preferring args if keys overlap

request.files: the files in the body, which Flask keeps separate from form. HTML forms must use enctype = multipart/form-data or files will not be uploaded.

request.json(): parsed JSON data. Request must have application/json content type header, or use request.get_json(force = True) to ignore the content type
data = request.get_json()

If you're not sure how data will be submitted, you can use an or chain to get the first one with data:
def get_request_data():
    return (request.args or request.form or request.get_json(force = True, silent = True) or request.data)

Use jsonify to display dict items
Use {'key': list of items} to display list items
use return statement to display if items are already with a key and value


Ideal design hierarchy for Python APIs:

Model
model.py trains and saves the model to the disk.

Server
server.py contains all the required for flask and to manage APIs.

Request
request.py contains the python code to process POST request to server.

Ref:
    https://www.tutorialspoint.com/flask/flask_sqlite.htm
    https://stackoverflow.com/questions/19794695/flask-python-buttons
'''
# Simple flash example with 1 route
host = '127.0.0.1'
port = 8000
url = '/'
methods = ['GET']
app = Flask(__name__) # runs on http://127.0.0.1:8000 as get
@app.route(url, methods = methods)
def hello():
   return 'Hello World, Sree here!'
if __name__ == '__main__':
    app.run(host = host, debug = True, port = port, use_reloader = False)

# Simple flask snippet
host = '127.0.0.1'
port = 8000
url = '/users/<string:username>'
methods = ['GET', 'POST']
app = Flask(__name__) # runs on http://127.0.0.1:8000/users/Sree as both get and post
@app.route(url, methods = methods)
def hellobud(username = None):
    return ('Hello {}!'.format(username))
if __name__ == '__main__':
    app.run(debug = True, port = port, use_reloader = False)

# Multiple routes flask
host = '127.0.0.1'
port = 8000
app = Flask(__name__) # runs on 127.0.0.1:8000/multi/5
@app.route('/multi/<int:num>', methods = ['GET'])
def getMultiply(num):
    return jsonify({'Result is' : num * 12})
if __name__ == '__main__':
    app.run(debug = True, port = port, host = host, use_reloader = False)

# Works on http://127.1.2.3:1212/emp/sreekanth/11.9/464511
host = '127.0.0.1'
port = 8000
app = Flask(__name__)
@app.route('/emp/<string:name>/<float:exp>/<int:id>', methods = ['GET', 'POST'])
def emp(name, exp, id):
    return jsonify("I'm %s having %2.2f years of experience. My emp ID is %i"%(name, exp, id))
if __name__ == '__main__':
    app.run(host = host, port = port, debug = True, use_reloader = False)

# Multiple protocols in flask
host = '127.0.0.1'
port = 8000
app = Flask(__name__) # runs on http://127.0.0.1:8000/ as POST with Json input as {"Name is ": "sree"}
@app.route('/', methods = ['GET', 'POST'])
def helloji():
    if request.method == 'POST':
        jsonReply = request.get_json()
        return jsonify(jsonReply), 201
    else:
        return jsonify('Thanks for trying!')
if __name__ == '__main__':
    app.run(debug = True, port = port, use_reloader = False, host = host)

# Flask RESTful example
host = '127.0.0.1'
port = 8000
app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):

    def get(self): # runs on http://127.0.0.1:9000/ as GET
        return {'About Us ' : 'We are smart!'}

    def post(self): # runs on http://127.0.0.1:9000/ as POST with Json input as {"Name is ": "sree"}
        myJson = request.get_json()
        return {'You sent ' : myJson}, 201

class Multi(Resource):

    def get(self, num): # runs on http://127.0.0.1:9000/ as GET
        return {'result' : num * 12}

api.add_resource(HelloWorld, '/', methods = ['GET', 'POST'])
api.add_resource(Multi, '/multi/<int:num>', methods = ['GET'])

if __name__ == '__main__':
    app.run(debug = True, port = port, host = host, use_reloader = False)

# Flask XOR API. Gives internal server error on http://127.0.0.1:1000/xor_prediction?x1=1&x2=0
url = r'D:\Programming Tutorials\Machine Learning\Model Deployment\Flask API Tutorial\xorModel.pkl'
def predict_xor_predict(n1, n2):
    output = {'Output prediction of XOR' : 0}
    x_input = np.array([n1, n2]).reshape(1, 2)
    pFile = url
    m1 = pickle.load(open(pFile, 'rb'))
    output['Output prediction of XOR'] = m1.predict(x_input)[0]
    print(output)
    return output
host = '127.0.0.1'
port = 8000
app = Flask(__name__)
@app.route('/')
def index():
    return 'XOR Prediction!'
@app.route('/xor_prediction', methods = ['GET'])
def calc_xor_predict():
    body = request.get_data()
    header = request.headers
    try:
        n1 = int(request.args['x1'])
        n2 = int(request.args['x2'])
        if(n1 != None) and (n2 != None) and ((n1 == 0) or (n1 == 1)) and ((n2 == 0) or (n2 == 1)):
            res = predict_xor_predict(n1, n2)
        else:
            res = {'Success' : False, 'message' : 'Input data incorrect'}
    except:
        res = {'Success' : False, 'message' : 'Unknown errors!'}
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug = True, port = port, use_reloader = False, host = host)

# Fitting a classifier to pickle. This needs to be tested
myClassifier = pickle.load(open('picklefile.pkl','rb'))
app = Flask(__name__)
@app.route('/api', methods = ['POST'])
def doPredict():
    data = request.get_json(force = True)
    predictInput = [data['sl'], data['sw'], data['pl'], data['pw']]
    predictInput = np.array(predictInput)
    yPred = myClassifier.predict(predictInput)
    output = [yPred[0]]
    return jsonify(results = output)

'''''''''''''''''''''''''''''''''''''''Titanic''''''''''''''''''''''''''''''''''''''''''''''

# Titanic Flask API. This code works for deployment
'''Ref: https://www.datacamp.com/community/tutorials/machine-learning-models-api-python'''

'''model.py'''
# Import dependencies
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from flask import Flask, request, jsonify
import traceback
import sys

# Load the dataset in a dataframe object and include only four features as mentioned
url = 'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Titanic_train.csv'
dataset = pd.read_csv(url)
include = ['Age', 'Sex', 'Embarked', 'Survived'] # Only four features
dataset_refined = dataset[include]

# Data Preprocessing
categoricals = []
for col, col_type in dataset_refined.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          dataset_refined[col].fillna(0, inplace = True)

dataset_refined_ohe = pd.get_dummies(dataset_refined, columns = categoricals, dummy_na = True)

# Logistic Regression classifier
dependent_variable = 'Survived'
x = dataset_refined_ohe[dataset_refined_ohe.columns.difference([dependent_variable])]
y = dataset_refined_ohe[dependent_variable]
logistic_regression = LogisticRegression()
logistic_regression.fit(x, y)

# Save your model
joblib.dump(logistic_regression, 'model.pkl')
print('Model dumped!')

# Load the model that you just saved
logistic_regression = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print('Models columns dumped!')

'''api.py'''

# Dependencies
# Your API definition
host = '127.0.0.1'
port = 8000
app = Flask(__name__)
@app.route('/titanic_predict', methods = ['POST'])
def predict():
    try:
        logistic_regression = joblib.load('model.pkl') # Load "model.pkl"
        print ('Model loaded')
        model_columns = joblib.load('model_columns.pkl') # Load "model_columns.pkl"
        print ('Model columns loaded')
        json_ = request.json # request.get_json() also works the same
        print(json_)
        query = pd.get_dummies(pd.DataFrame(json_))
        query = query.reindex(columns = model_columns, fill_value = 0)
        prediction = list(logistic_regression.predict(query))
        return jsonify({'prediction': str(prediction)})
    except:
        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = port # If you don't provide any port the port will be set to 9090
    app.run(host = host, port = port, debug = False)

'''
Test the above API with the below raw Json in Postman via url 127.0.0.2:9000/titanic_predict
[
	{"Age": 85, "Sex": "male", "Embarked": "S"},
	{"Age": 29, "Sex": "female", "Embarked": "C"},
	{"Age": 65, "Sex": "male", "Embarked": "S"},
	{"Age": 10, "Sex": "male", "Embarked": "S"}
]

'''

'''''''''''''''''''''''''''''''''''Loan Prediction''''''''''''''''''''''''''''''

https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask
https://github.com/pratos/flask_api/blob/master/notebooks/ML%20Models%20as%20APIs%20using%20Flask.ipynb

'''

# importing the dataset
url = 'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Pickles\Loan Prediction train.csv'
data = pd.read_csv(url)

# probing for descriptive statistics
data.head(3)
list(data.columns)
data.shape

# probing for missing values
for _ in data.columns:
    print('No of null values in:{}= {}'.format(_, data[_].isnull().sum()))

# splitting the dataset into training & test data
pred_var = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'], test_size = 0.25, random_state = 42)

'''
missing_predictors = [ 'Gender', 'Married', 'Dependents', 'Self_Employed', LoanAmount, 'Loan_Amount_Term', Credit_History]
for _ in missing_predictors:
    print('List of unique labels for {}:{}'.format(_, set(data[_])))

# Imputing the missing values
X_train['Dependents'] = X_train['Dependents'].fillna('0')
X_train['Self_Employed'] = X_train['Self_Employed'].fillna('No')
X_train['Loan_Amount_Term'] = X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mean())
X_train['Credit_History'] = X_train['Credit_History'].fillna(1)
X_train['Married'] = X_train['Married'].fillna('No')
X_train['Gender'] = X_train['Gender'].fillna('Male')
X_train['LoanAmount'] = X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean())
label_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
for _ in label_columns:
    print("List of unique labels {}:{}".format(_,set(X_train[_])))
#Mapping the numerical to categorical values
gender_values={'Female':0,'Male':1}
married_values={'No':0,'Yes':1}
education_values={'Graduate':0,'Not Graduate':1}
employed_values={'No':0,'Yes':1}
property_values={'Rural':0,'Urban':1,'Semiurban':2}
dependent_values={'3+':3,'0':0,'2':2,'1':1}
X_train.replace({'Gender':gender_values,'Married':married_values,'Education':education_values,'Self_Employed':employed_values,'Property_Area':property_values,'Dependents':dependent_values},inplace=True)
X_train.head(3)
X_train.info()
for _ in X_train.columns:
    print("The number of null values in:{}= {}".format(_,X_train[_].isnull().sum()))
X_train=X_train.as_matrix()
'''

# create custom pre-processing estimator that would help in writing better pipelines and in future deployments
class PreProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def transform(self, df):
        #Regular transform() that is a help for training, validation & testing datasets (NOTE: The operations performed here are the ones that we did prior to this cell)
        pred_var = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        df = df[pred_var]
        df['Dependents'] = df['Dependents'].fillna(0)
        df['Self_Employed'] = df['Self_Employed'].fillna('No')
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(self.term_mean_)
        df['Credit_History'] = df['Credit_History'].fillna(1)
        df['Married'] = df['Married'].fillna('No')
        df['Gender'] = df['Gender'].fillna('Male')
        df['LoanAmount'] = df['LoanAmount'].fillna(self.amt_mean_)
        gender_values = {'Female' : 0, 'Male' : 1}
        married_values = {'No' : 0, 'Yes' : 1}
        education_values = {'Graduate' : 0, 'Not Graduate' : 1}
        employed_values = {'No' : 0, 'Yes' : 1}
        property_values = {'Rural' : 0, 'Urban' : 1, 'Semiurban' : 2}
        dependent_values = {'3+' : 3, '0' : 0, '2' : 2, '1' : 1}
        df.replace({'Gender' : gender_values, 'Married' : married_values, 'Education' : education_values, 'Self_Employed' : employed_values, 'Property_Area' : property_values, 'Dependents' : dependent_values}, inplace = True)
        return df.values
    def fit(self, df, y = None, **fit_params):
        #Fitting the Training dataset & calculating the required values from train e.g: We will need the mean of X_train['Loan_Amount_Term'] that will be used in transformation of X_test
        self.term_mean_ = df['Loan_Amount_Term'].mean()
        self.amt_mean_ = df['LoanAmount'].mean()
        return self
# Convert y_train & y_test to np.array:
y_train = y_train.map({'Y' : 1, 'N' : 0})
y_test = y_test.map({'Y' : 1, 'N' : 0})

'''
X_train,X_test,y_train,y_test=train_test_split(data[pred_var],data['Loan_Status'],test_size=0.25,random_state=42)
X_train.head(3)
for _ in X_train.columns:
    print("The number of null values in:{}= {}".format(_,X_train[_].isnull().sum()))
preprocess=PreProcessing()
preprocess.fit(X_train)
X_train_transformed=preprocess.transform(X_train)
X_train_transformed.shape
X_test_transformed=preprocess.transform(X_test)
X_test_transformed.shape
y_test=y_test.replace({'Y':1,'N':0}).as_matrix()
y_train=y_train.replace({'Y':1, 'N':0}).as_matrix()
param_grid={"randomforestclassifier__n_estimators":[10,20,30], "randomforestclassifier__max_depth":[None,6,8,10], "randomforestclassifier__max_leaf_nodes":[None,5,10,20], "randomforestclassifier__min_impurity_split":[0.1,0.2,0.3]}

'''

pipe = make_pipeline(PreProcessing(), RandomForestClassifier())
pipe
param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30], "randomforestclassifier__max_depth" : [None, 6, 8, 10], "randomforestclassifier__max_leaf_nodes" : [None, 5, 10, 20], "randomforestclassifier__min_impurity_decrease":[0.1, 0.2, 0.3]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 3)
print('Finding the optimal parameters...')
grid.fit(X_train, y_train)
print("Best parameters: {}".format(grid.best_params_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))

# Load test dataset
test_df = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Pickles\Loan Prediction test.csv')
test_df.head(3)
predictions = grid.predict(test_df)

# Creating a pickle file using dill library
filename = 'model_v1.pk'
with open(filename, 'wb') as file:
	pickle.dump(grid, file)
with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)
loaded_model.predict(test_df)

# Creating a joblib file
filename2 = 'model_v2.jl'
with open(filename2, 'wb') as file:
	joblib.dump(grid, file)
with open(filename2, 'rb') as f:
    loaded_model2 = joblib.load(f)
loaded_model2.predict(test_df)

'''Creating an API using Flask'''
app = Flask(__name__)
@app.route('/predict', methods = ['POST'])
def apicall():
    try:
        #API Call. Pandas dataframe (sent as a payload) from API Call
        test_json = request.get_json()
        test = pd.read_json(test_json, orient = 'records')
        #To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
        test['Dependents'] = [str(x) for x in list(test['Dependents'])]
        loan_ids = test['Loan_ID']  #Getting the Loan_IDs separated out
    except Exception as e:
        raise e
    clf = 'model_v1.pk'
    # clf = 'model_v2.jl'
    if test.empty:
        return(bad_request())
    else:
        print("Loading the model...") #Load the saved model
        loaded_model = None
        with open('./models/' + clf, 'rb') as f:
            loaded_model = pickle.load(f)
        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(test)
        #Add the predictions as Series to a new pandas dataframe or depending on the use-case, the entire test data appended with the new files
        prediction_series = list(pd.Series(predictions))
        final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))
        #We can be as creative in sending the responses.But we need to send the response codes as well
        responses = jsonify(predictions = final_predictions.to_json(orient = "records"))
        responses.status_code = 200
        return(responses)

#Setting the headers to send and accept json responses
header = {'Content-Type' : 'application/json', 'Accept' : 'application/json'}

'''
if __name__ == '__main__':
    app.run(host = '127.0.0.1', debug = False, port = 3000)
'''
# Reading test batch
df = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Pickles\Loan Prediction test.csv',encoding='utf-8-sig')
# data = df  # uncomment this for all the records to be parsed to data.to_json()
data = df.head(1) # passing only 1 record from CSV file

'''
# Just 1 of the records of the csv
data = [{"Loan_ID":"LP001015","Gender":"Male","Married":"Yes","Dependents":"0","Education":"Graduate",
         "Self_Employed":"No","ApplicantIncome":5,"CoapplicantIncome":0,"LoanAmount":110.0,
         "Loan_Amount_Term":360.0,"Credit_History":1.0,"Property_Area":"Urban"}]
'''

#Converting Pandas Dataframe to json
data = data.to_json(orient = 'records')

# POST <url>/predict
'''
resp = requests.post("http://127.0.0.1:3000/predict", \
                    data = json.dumps(data),\
                    headers= header)
resp.status_code
resp.json()'''

resp = requests.post("http://127.0.0.1:3000/predict", data = json.dumps(data), headers = header)
resp.status_code
resp.json()

# THIS IS AN OPPORTUNITY TO FIX THE MULTIPLE INPUTS AND RETURN A JSON OUTPUT

'''Try below in PyCharm for both front end n backend predictions'''

# https://medium.com/@dvelsner/deploying-a-simple-machine-learning-model-in-a-modern-web-application-flask-angular-docker-a657db075280
# read iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
# fit model
clf = svm.SVC(C=1.0,probability=True,random_state=1)
clf.fit(X, y)
print('Training Accuracy: ', clf.score(X, y))
print('Prediction results: ', clf.predict_proba([[5.2,  3.5,  2.4,  1.2]]))
# declare constants
HOST = '127.0.0.1'
PORT = 8085
app = Flask(__name__)
@app.route('/api/train', methods = ['POST'])
def train():
    parameters = request.get_json() # get parameters from request
    iris = datasets.load_iris() # read iris data set
    X, y = iris.data, iris.target
    clf = svm.SVC(C = float(parameters['C']), probability = True, random_state = 1)
    clf.fit(X, y)
    joblib.dump(clf, 'model.pkl') # persist model
    return jsonify({'accuracy': round(clf.score(X, y) * 100, 2)})
if __name__ == '__main__':
    app.run(host = HOST, debug = False, port = PORT) # automatic reloading enabled