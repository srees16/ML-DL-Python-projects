'''Python OOP, Numpy, Pandas, Strings et al

Numpy
Pandas
EDA
Matplotlib
Seaborn
Flask
Postgres
OOP:
    class & methods
    inheritance
    dunder methods/special methods/default functions/magic methods
Exceptions
List comprehension: gives a list of elements by iterating thru another list with a simple function
Nested list comprehension
Tuple
Named tuple
Dict: dictionaries are implementation of hash table
default dict
ordered dict
Dict comprehension
Lambda expression: a simple 1 line anonymous function
Filter: filters out every element in a sequence that meets the condition in a predefined function
Map: maps a function/lambda expression to an iterable
**Filter is applied to check boolean condition, map is applied for performing lambda operation**
Zip: gives a list of tuples when multiple lists/sets/tuples are zipped. Opp of tuple unpacking
Reduce: iterable is reduced to a single value based on the function that we pass to it
Enumerate: enumerates thru an iterable with a counter
counter: counts the no of occurances of characters in a string
default dict: generates default keys with values
ordered dict: preserves the order in which items r inserted in a dict
Named tuple: acts as function
Itertools
Functools
Decorators: it’s a function that takes another function as argument and extends the behavior of it without explicitly modifying it, meaning, it allows adding new functionality to existing function.
Generators: used for efficient memory management to store elements instead of a list which consumes memory. Yield is the keyword.

Iterables are strings, lists, sets, tuples, dicts. Encapsulate string in iter() to replicate behaviour of an iterable

break, continue & pass: can be used in while & for loops:
    Break statement terminates the current loop and resumes execution at the next statement

    Continue stops/skips current execution, and proceeds with next iteration (returns control to the beginning of loop), rejects all remaining statements in current iteration of loop and the control goes back to the start of the loop

    Pass statement is used when a statement is required syntactically but no command/code needs to execute.

    Else statement is executed when the loop has exhausted iterating. can be used with for & while loops. It's executed only when For/While loops are NOT terminated by a break statement.

For comparing iterables:
Pandas Series has ~, isin, union1d, intersection1d, str.startswith(), str.contains(), str.endswith()
Python Set has union, intersection, difference, symmetric difference, update along with -,^,& symbols

Why Numpy Instead Of Python Lists?
    Numpy arrays are more compact than lists
    Access in reading and writing items is faster with Numpy
    Numpy is more convenient due to vector and matrix operations
    Numpy can be more efficient as they are implemented efficiently

In numpy, axis 0 for vertical index and axis 1 for horizontal columns
In pandas, axis 0 for horizontal index and axis 1 for vertical columns

Pandas:
    drop() and dropna()
    drop() is for dropping row or column values based on condition or without condition
    dropna() is for dropping nan values from specific rows/columns or from entire data frame

Machine learning algorithms that use gradient descent as an optimization technique, esp distance based algorithms
like PCA, KNN, K-means, SVM require data to be scaled. If features have different scale/range, it takes longer for
gradient descent to converge at the global minimum faster and accurate convergence since they use distances
between data points to determine their similarity.

Naive Bayes and Tree based algorithms are not sensitive to the scale of the features. Decision Tree splits a node based on a
single feature that increases homogeneity of the node. Hence feature scaling is not required.

Min Max Scaler works based on Normalization. Values bound to range of 0 and 1.
Standard Scaler works based on Standardization. Values are unbounded.

Normalization is useful when distribution of data is not Gaussian distribution. Ex: KNN and ANN which do not assume any distribution of data
Standardization is useful when data follows Gaussian distribution. No bounding range so outliers will not be affected by this

OneHot encoded features can be Normalized as they are already in the range of 0 to 1. It wont affect their value.
OneHot encoded features cant be Standardized as it means assigning a distribution to categorical features.

To find hyperparameters of any model/algorithm, use get_params() of that object which instantiates that model/algorithm

Decision Tree/Random Forest hyperparameters:
n_estimators = number of trees in the foreset
max_features = max number of features considered for splitting a node
max_depth = max number of levels in each decision tree
min_samples_split = min number of data points placed in a node before the node is split
min_samples_leaf = min number of data points allowed in a leaf node
n_iter = controls no of different combinations to try
bootstrap = method for sampling data points (with or without replacement)

Choose ML algorithm based on:
    Data:
        size: some algorithms perform better with larger data. for small dataset, algorithms with high bias/low variance, 
            like Naïve Bayes are better
        characteristics: if data is spread linearly, GLM models like Linear/logistic regression or SVM are better. for 
            complex data, random forest works better
        behavior: for sequential or chained features (weather/stock market), decision tree works better
    Accuracy: trade off between accuracy & speed since high accuracy means low speed due to training & processing. SVM, ANN, 
            random forest give better accuracy but high latency
    Features & Parameters
    Interpretability (blackbox/whitebox)

Scaling an ML solution:
    horizontal scaling
    if batch/realtime predictions
    dimensionality reduction
    use built in libraries & functions
    decide trade off between accuracy & speed
    building pipelines to stream new input data

create requirements.txt file automatically:
    open terminal and navigate to root folder of the code
    run pipreqs ./ (pip install pipreqs)

Path params identify specific resource/resources, query parameters sort/filter those resources

'''
import os #dir(os): gives all functions in that library/module as list
import sys #dir(sys)
import time # use time.time() to calculate start & end times for operations
import string
import numpy as np
import pandas as pd #dir(pandas)
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
import unittest
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

'''Running Notes:
.describe() gives descriptive analysis of the dataset or a specific column.
.info() gives details of features like dtype, no of non-null values.
Plot them using sns.distplot(dataframe or series/column) to view its kurtosis and skew which can be
verified from df.skew() and df.kurtosis().
'''
'''========================================================================================='''

# String library has functions for alphabet, numerals & punctuations among others
string.digits
string.ascii_letters
string.punctuation

# Convert a variable to different data type by nesting
string_outcome = str(numeric_input) # Converts numeric_input to string_outcome
integer_outcome = int(float_input) # Converts float_input to integer_outcome
float_outcome = float(numeric_input) # Converts numeric_input to float_outcome

for i in 'forloop':
    if i == 'l':
        break
    print(i)

for i in 'forloop':
    if i == 'l':
        continue
    print(i)

for i in 'forloop':
    if i == 'l':
        pass
    print(i)

val = 10
while val > 0:
    print(val)
    val -= 1
    if val == 5:
        break
print('Bye')

val = 10
while val > 0:
    print(val)
    val -= 1
    if val == 5:
        continue
print('Bye')

val = 10
while val > 0:
    print(val)
    val -= 1
    if val == 5:
        pass
print('Bye')

'''args & kwargs''' # useful if input size is unknown, kwargs is for key words & values as input

def adds(*args):
    return args

adds(9, 2)

def addition(*args):
    return sum(args)

addition(3, 1, 5)

def kwargs_method(**kwargs):
    return kwargs

kwargs_method(last_name = 'bond', first_name = 'james')
''''''
2+9
# for
for nums in alist:
    print(nums)
# while
i = 1
while i <= 5:
    print(i)
    i += 1

for item in range(0, 5):
    print(item)

range(10) # same as saying range(0, 10)
range(2, 10)
list(range(10))
list(range(4, 10))
list(range(0, 20, 2)) # 3rd arguement is step size
list(range(20, 0, -2)) # iteration in reverse

out = []
for num in range(5):
    out.append(num ** 2)
print(out)
# or
[x ** 2 for x in range(5)]
''' functions'''
def my_func(param):
    print(param)
my_func(45)

def wish(name = 's'):
    print('hello ' + name)
wish()
wish('sree sir')

def name_age(name = 'ss', age = 89):
    print('my name is ' + name + ' and aged {} years'.format(age))
name_age()
name_age('sree', 34)

def addition(*args):
    return sum(args)
addition(24, 53, 66, 44, 71, 89, 19, 44)

def greater(*args):
    return max(args)
greater(35, 78, 74, 22)
greater(24, 53, 66, 44, 71, 89, 19, 44)

def person_details(**kwargs):
    return kwargs
person_details(name = 'rob', age = 12, grade = 'B', class_ = '10th')

alist = [24, 53, 66, 44, 71, 89, 19, 44]
alist.pop()
last = alist.pop()
pop_any = alist.pop(3)

# pop all specific integers
alist = [24, 53, 66, 44, 71, 89, 24, 19, 53, 44, 53, 24]

def remove_all(lis, *args):
    for i in lis:
        if i in args:
            lis.remove(i)
    return lis

remove_all(alist, 24, 53)

1 in alist
89 in alist

def get_user_name(email):
    print(email.split('@')[0])
def get_domain_name(email):
    print(email.split('@')[-1])
get_domain_name('user@domain.com')

def findDog(input):
    return 'dog' in input
findDog('Is there a dog here?')

def findDog(input):
    return 'dog' in input.lower().split()
findDog('Is there a dog here?')

def count_dogs(string):
    count = 0
    for word in string.lower().split():
        if word == 'dog':
            count += 1
    return count
count_dogs('This dog runs faster than the other dog dude!')

seq = ['soup','dog','salad','cat','great']
set(filter(lambda word: word[0] == 's' or word[0] == 'c', seq))

# remove empty values
a = [35, None, 0, 'fast', None]
[i for i in a if i]

# take screenshot of screen
import pyautogui as p
ss = p.screenshot()
ss.save('ss.jpg')

# using google search api
from googlesearch import search
query = 'Geeksforgeeks' # to search

for i in search(query, tld = 'com', num = 5, stop = 10, pause = 2):
    print(i)
# Or
[i for i in search(query, tld = 'com', num = 5, stop = 10, pause = 2)]

'''''''''''''''''''''''''''''''''Data Types'''''''''''''''''''''''''''''''''
#Python data types: Lists, Tuples, Dict, Set

'''''''''''''''''''''''''''Lists'''''''''''''''''''''''''''
# Split and join are opp of each other i.e., string.split() is opp of ''.join(). strip() removes specified character

my_list = [1, 4.2, 'bae']
my_list.append('w')
my_list
my_list[::]
my_list[1:]
my_list[:2]
my_list[0:3]
my_list[1] = 'jack'
my_list
my_list + ['ean'] + [46]
my_list
my_list = my_list + ['ean'] + [53]
my_list
my_list = my_list * 2
my_list.index('bae', 3, len(my_list)) # searches for occurance of element from the specified index
len(my_list)
nest = ['hey',53,[3,'fwa']]
nest[2][1][1]
nested = [54,'xe',[22,6.3,'awfds',['rt',4.7,78,[3.6,'yblo',74]]]]
nested[2][3][0]
# grab 't' from the above list
nested[2][3][0][1]
# grab 'o'
nested[2][3][3][1][3]

# explore difference between .copy() aka shallow copy and direct linked copy in lists
ma_list = ['ss', 'tt', 46, 5.32, 2e3, -43]
ma_list[-1:] # or ma_list[-1]
copy_list = ma_list # direct link to the previous list
copy_list1 = ma_list[:] # shallow copy of the previous list
copy_list2 = ma_list.copy() # shallow copy of list
ma_list[1] = -31
ma_list[0] = 'sk'
ma_list
copy_list
copy_list1
copy_list2
ma_list[4:5] = [] # deleting single item
ma_list
ma_list[3:5] = [] # deleting specific items based on their index
ma_list[1:3] = ['bt', 64]
ma_list
ma_list[:] = [] # deleting entire list
ma_list[0] = 'sk'
ma_list

my_list = [10, 15, 3, 7, 63, 25]
head, *body, tail = my_list
print(head)
print(body)
print(tail)

x = list(range(0, 11))

# converting a string into list/set of chars
string = 'jackas'
new_string = list(string) # or
[i for i in new_string]
new_string[5] = 'ss '
name = ''.join(new_string)
name.strip() # removes white spaces before or after a string or a specified character

# splitting n joining a string
long_string = 'here is a list'
lis = list(long_string)
back_to_long_string = ''.join(lis)
back_to_long_string

li = [54,'xe',[22,6.3,'awfds',['rt',4.7,78,[3.6,'yblo',74]]]]
li.remove(54) # removes element by its name, not by index
li.remove(li[1]) # removes element by its index. OR use li.pop(1) to remove by index
li
li.clear() # clears all elements from list

nations = ['India', 'US', 'Nepal', 'Bhutan', 'Japan', 'US', 'France', 'Germany', 'Croatia']
nations.index('US', 2, len(nations)) # searches for occurance of element after specified index til specified index
nations.sort(reverse = True) # same as nations.reverse()
sorted(nations, reverse = True)
nations[5:6] = [nations[5], 'UK']
nations
# Using the list, make a for loop to display each of the individual students marks. Hint: use tuples
marks = [['Jack', 80, 39, 51], ['Ass', 70, 92, 81], ['Dude', 92, 74, 64]]
stu1 = marks[0]
stu1_marks = marks[0][1:]
for mark in marks:
    stu = mark[0]
    stu_marks = mark[1:]
    print(stu, stu_marks)

# any/all
nations = ['India', 'US', 'Nepal', 'Bhutan', 'Japan', 'US', 'France', 'Germany', 'Croatia']
all(len(n) >= 2 for n in nations)
any('apa' in n for n in nations)

'''''''''''''''''''''''''''List comprehension'''''''''''''''''''''''''''

evens = [x for x in range(11) if x % 2 == 0]
evens
out = [num ** 2 for num in range(5)]

# create a list of the 1st letters of every word in the string
strings = 'create a list of the first letters of every word in the string'
# using function
def list_of_first_letters(string):
    first_letters = []
    for s in strings.split():
        first_letters.append(s[0])
    print(first_letters)
list_of_first_letters(strings)
# using list comprehension
first_letters1 = [word[0] for word in strings.split()]

'''Using normal for loop'''
lister = []
my_string = 'animosity'
for letter in my_string:
    lister.append(letter)
print(lister)

'''Using for loop in list comprehension'''
my_string = 'animosity'
lister = [letter for letter in my_string]
print(lister)
re_my_string = ''.join(lister) # re-adding the string back
my_string == re_my_string # checking if both the strings are same

''' Nested list comprehension '''
def cube_three_multiples(): # typical function
    multiples_of_three = []
    for x in range(1, 31):
        result = x ** 3
        if result % 3 == 0:
            multiples_of_three.append(result)
    print(multiples_of_three)
cube_three_multiples()

# above function has be condensed to nested list comprehension
[round(sqrt(x), 2) for x in range(1, 30) if x % 3 == 0] # square root of multiples of 3
[x ** 3 for x in [x for x in range(1, 31) if x % 7 == 0]] # cube of multiples of 7
factors_of_six = [x % 6 == 0 for x in [x for x in range(25) if x % 2 == 0]]
factors_of_six

# List append vs extend
x = [3,8,7,9,1]
x.append([5, 8])
x
x1 = [3,8,7,9,1]
x1.extend([5, 8])
x1

# create list of n empty data structures
[[] for x in range(3)] # list
[() for x in range(3)] # tuple
[{} for x in range(3)] # dict
[set() for x in range(3)] # set
([] for x in range(3)) # says generator object

# Split A Python List Into Evenly Sized tuples/chunks
x = [1,2,3,4,5,6,7,8,9]
# Split in chunks of 3
y = list(zip(*[iter(x)] * 3))

# Flatten Lists Out Of Lists
nested_list = [[1,2,3], [4,5,6], [7,8,9]]
# using list comprehension
[i for j in nested_list for i in j]
# above can be written as function below
list1 = []
for sublist in lis:
  for item in sublist:
    list1.append(item)

sum(nested_list, []) # Flatten out original list of lists with sum()
lis = [[3,'s'], ['sre'], [7.2,1,85,'as'], [2, 'zxe']]
sum(lis, [])
# using reduce
print(reduce(lambda x, y: x + y, lis))

# Flatten list Of tuples
list_tups = [(19.27, 12.28), (20, 70), (2.52, 65.52)]
list(sum(list_tups, ()))
[item for sublist in list_tups for item in sublist] # using list comprehension
import itertools
list(itertools.chain(*list_tups)) # using itertools

'''''''''''''''''''''''''''Tuples'''''''''''''''''''''''''''
t = (2, 'dfs', 67.3)
t[1][1]
t[0] = 543
# list of tuples
tup = [(1, 9, 's','r'), ('x', 'y')]
tup[0] = (7, 'q', 8.9)
tup[0][1] = 'a'
# tuple of lists
tup_list = ([1, 9, 's','r'], ['x', 'y'])
tup_list[0] = (7, 'q', 8.9)
tup_list[0][1] = 10
tup_list

# Tuple unpacking
'''Tuple unpacking is opp of Zip'''

x = [(4,2),('a',7),(3.1,'df')]
for item in x:
    print(item)

my_tuples = []
tuple_list = [(452, 255), (24, 63), (53, 13)]
for t1, t2 in tuple_list:
    if t1 > t2:
        print(t2)
        my_tuples.append(t1)
print(my_tuples)

a, b, c = ('me', 53, 4.12, False)[1:] # indicates from which index unpacking needs to happen
a, b, c = ['me', 53, 4.12, False][:3] # same for list also
print(a, c)
grades = (53, 24, 36, 48, 29)
x, y, z = grades[0:3]
print(x, y, z)

# changing elements in tuple by coverting into list and back to tuples
scores = ('me', 53, 4.12, False)
re_score = list(scores)
re_score[-1] = 'wow'
re_score
scores = tuple(re_score)
scores

name1 = ('a', 'b')
name2 = ('c', 'd')
names = name1 + name2
names + (46, 85)

# to make it a tuple, put a comma after the single element , without which it would be a string, not tuple
nax = ('tes') # same as 'tes'
type(nax)
nae = ('tes',)
type(nae)
names = names + nax
names

t1 = ('a', 'b')
t2 = ('c', 'd')
# there is no tuple.append() method bcoz tuple is immutable unlike list

'''Named Tuple'''
# assigns names as well as numerical index to member in tuple
t = (5,2,6,1,7,3)
t[3]
dog = namedtuple('Doggy', 'age breed name')
sam = dog(23, 'Dober', 'viks')
sam.name
sam.age
sam[0]
sam.breed
sam.index

cat = namedtuple('Catty', 'name age')
c = cat('cutes', 32)

'''''''''''''''''''''''''''Dictionary

while loop keywords: break, continue, pass
break: breaks out of current closest enclosing loop
continue: goes to the top of the closest enclosing loop
pass: does nothing

'''
my_dict = {}
my_dict[2] = 'zet'
my_dict['ar'] = 45
my_dict
my_dict[2] = 'xet'
my_dict
d = {1 : 'ar', 2 : 'ba', 3 : [53,'lke',6.2]}
d.keys()
d.values()
d1 = {'vs' : {53 : 'yrs', 'had' : 89}, 91 : {3 : [53,'lke',6.2]}}
d1.items()
d[3][1][0]
dict_list = d[3]
d1['vs'][53]
d1[91][3][1][0]

# Merge two dictionaries
d1 = dict(bob = 9, sob = 4)
d2 = dict(job = 4, rob = 7)
d1.keys()
d2.values()
{*d1}
{**d1}
{*d1.values()}
{*d1.items()}
{**d1, **d2}

#upserting/upating dict object with new keys & values
d1 = dict(a = 53, b = 60)
d1['d'] = '4'
d1.update(key = 'value!')
d1.update(key = 5) # cant update dict with a numeric key or key with quotes: d1.update('key' = 5) or d1.update(5 = 'key')
d1.__setitem__(23, 'sdfd')
d1.__setitem__('23', 23)
{**d1, **{'ass': 0}}

'''dict.fromkeys() creates/updates a dict with entire iterable or a single int/string as values for each keys
if no iterable is provided, it creates None as values for keys'''
# ex 1
keys = ('key1', 'key2', 'key3')
dict.fromkeys(keys)
# ex 2
keys = ('key1', 'key2', 'key3')
value = 'valuable'
dict.fromkeys(keys, value)
# ex 3
keys = ['Apple', 'Orange']
values = [3000, 4000]
dict.fromkeys(keys, values)
# ex 4
keys = ['Apple', 'Orange']
vals = {'Ten': 10, 'Twenty': 20, 'Thirty': 30}
dict.fromkeys(keys, vals)
# ex 5
dict1 = {'Ten': 10, 'Twenty': 20, 'Thirty': 30}
dict2 = {'Thirty': 30, 'Fourty': 40, 'Fifty': 50}
dict.fromkeys(dict1, dict2)

# merge above 2 dicts
dict1.update(dict2)

dict1.setdefault('x', 56) # setting default key & values

# min & max in a dict
_dict = {'Physics': 82, 'Math': 65, 'history': 75}
_dict1 = {'Physics': 'Kelly', 'Math': 'John', 'history': 'Emma'}
min(sampleDict, key = _dict.get) # or min(sampleDict)
min(sampleDict, key = _dict1.get)

# rename specific key
dicy = {'name': 'Kelly', 'age': 25, 'salary': 8000}
dicy['CTC'] = dicy.pop('salary')

# return & delete specified keys from a dict
_dict = {'name': 'Kelly', 'age': 25, 'salary': 8000, 'city': 'New york'}
keys = ["name", "salary"]
{x: _dict[x] for x in keys} # returns only specified keys
{x: _dict[x] for x in _dict.keys() - keysToRemove} # removes only specified keys & returns the rest

# checking for specific keys/values in a dict
my_dict = {'a': 100, 'b': 200, 'c': 300}
'c' in my_dict
'a' in my_dict.keys()
200 in my_dict.values()

d1 = {'a': 1, 'b': 2, 'a': 3}
d1
d2 = {'a': 'x', 'b': 'y', 'a': 'z'}
d2

# Dictionary comprehension
{x: x ** 2 for x in range(11)}
{k: v for k, v in zip(['a','b'], range(2))}

'''Default Dict''' # https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work

someddict = defaultdict(int)
print(someddict[3])

somedict = {}
print(somedict[3]) # KeyError

dics = defaultdict(object)
dics['rest']
dix['keys'] = 'vals'
def default_value():
    return 'Bad key'
dix = defaultdict(default_value)
dix['as']
dix = defaultdict(lambda : 'Bad key')
dix['wassup']
dix

def default_string():
    return 'bad key'
some_string_dict = defaultdict(default_string)
print(some_string_dict['s'])

dictionary1 = { 'A': 'Geeks', 'B': 'For', 'C': 'Geeks'}
# using setdefault() method
third_value = dictionary1.setdefault('C')
print("Dictionary:", dictionary1)
print("Third_value:", third_value)

# Ordered Dictionary: Remembers the order in which the entries are added
normal_dict = {}
normal_dict[5] = 'df'
normal_dict[7] = 'jh'
normal_dict[2] = 'er'
normal_dict
for k, v in normal_dict.items():
    print(k, v)

normal_dict = OrderedDict()
normal_dict[4] = 'df'
normal_dict[1] = 'jh'
normal_dict[8] = 'er'
normal_dict
for k, v in normal_dict.items():
    print(k, v)

d1 = {}
d1['a'] = 1
d1['b'] = 2
d2 = {}
d2['b'] = 2
d1['a'] = 1
d1 == d2

x = 0
while x < 10:
    print('Value of x: ', x)
    x += 1
    if x == 5:
        print('We are halfway!')
        break
    else:
        continue

'''''''''''''''''''''''''''Sets'''''''''''''''''''''''''''
# to create an empty set, declare a variable with set(), not empty {} bcoz its for dict not tuple
# Set does not support indexing

setx = {24,66,44,24,66}
setx
alist = [24,66,44,24,44,66]
set(alist)
setx.add(99)
setx

s = set()
s.add(5)
s.add(1)
s.add(3)
s.add(1)
ss = s.copy()
ss.add(2)
ss
s.add(9)
s.add(4)
s.add(7)
s.add(6)
s
ss
s.union(ss) # add all common and uncommon elements
s.intersection(ss) # add only common elements
s.intersection_update(ss) # changing elements in 1st set with those in 2nd set
s.difference(ss)
s.difference_update(ss) # changing elements in 1st set with those in 2nd set
s.symmetric_difference(ss) # exclusive 'OR' which retains uncommon elements in 2 sets
s.discard(4)
s.clear()

'''same as above but with symbols'''
a = {24,66,44,24,66}
b = {14,44,24,75,82}

# unique to a
a - b
# unique to b
b - a
# common in both sets
a & b
# not common in both sets
a ^ b

names = ['chemistry', 'history', 'biology', 'physics', 'maths', 'history', 'physics']
set_names = set(names)
set_names[0] # gives error bcoz set object doesnt support indexing and is not subscriptable
# convert it back to list to use indexing
names = ['chemistry', 'history', 'biology', 'physics', 'maths', 'history', 'physics']
names = list(set(names)) # or use sorted(set(names))
names[1]
set_names.add('economics')
set_names.remove('biology')
set_names
word1 = set('hepburn')
word2 = set('audrey')
word1.union(word2)
word1 = word1.union(word2) # updates word1 with elements in word2, however we can use the below instead
word1.update(word2)
word1.union(word2) # add all common and uncommon elements
word1.intersection(word2) # add only common elements
word1.intersection_update(word2) # changing elements in 1st set with those in 2nd set
word1.difference(word2)
word1.difference_update(word2) # changing elements in 1st set with those in 2nd set
word1.symmetric_difference(word2) # exclusive 'OR' which retains uncommon elements in 2 sets
word1.discard(4)
word1.clear()

# Calculate the Quantile of a List in Python
a = np.array(range(10)) # Make a NumPy array
# Return the 25th percentile of our NumPy array
p = np.percentile(a, 25)
print(p)

'''''''''''''''''''''''''''Strings'''''''''''''''''''''''''''
# string.split() is opp of ''.join(). strip()  removes white spaces before or after a string or a specified character
string = 'This is just a test string, to play around with string functions. So lets see!'
string1 = 'This is just a test string..\tto play around with string functions.\nSo lets see!'

string.lower()
string.upper()
string[5]
string[-4:]
string[:-4]
string[-4:-2]

string.split('.')
words = string.split() or string.split('.') or string.split(',') # splits string wit multiple conditions
words
string[:] # same as string[::]
string[::2] # step size
string[::-1] # reverse entire sentence
# reverse only words but retain sentence structure
reversed_words = [rev[::-1] for rev in string.split()]
# reverse entire sentence but retain word structure
reversed_sentence = [rev for rev in string.split()[::-1]]
# reverse sentence along with words
reversed_word_sentence = [rev[::-1] for rev in string.split()[::-1]]

name = 'jack ryan'
name[5]
name[1:] # grab from specified index till the end
name[::] # same as name[:] but former is preferred
name[:3] # grab from index 0 till specified index
name[2:8] # grab from specified index till the specified index

s1 = 'tear'
s2 = 'fear'
s3 = 'train'
if len(s1) == len(s2) and len(s1) == len(s3):
    print('same')
else:
    print('not same')

# Assessment
strings = 'Print only the words that start with s in this sentence'
for word in strings.split():
    if word[0] == 's':
        print(word)

# If the length of a word is even in the string, print even
strings = 'print every word in this sentence that has an even number of letters'
for word in strings.split():
    if len(word) % 2 == 0:
        print(word)

letter = 'S'
letter * 4
letter + 'rees' # concatenation to the string object works but not replacing the element in string

'''Print formatting'''

name = 'jack ryan'
number = 40
floats = 12.2164

"my name is {} and i'm {} years old".format(name, number)
"my name is {a} and i'm {b} years old".format(b = number, a = name)
"my name is {x} and friends call me {x}".format(x = 'jackass')
"my name is %s and i'm %i years old" %(name, number) # %s for string, %i for integer
"my name is %s and i'm %i years old and ROI is %1.2f" %(name, number, floats)
#https://www.youtube.com/watch?v=vTX3IwquFkc
lis = ['jack', 23]
print('my name is {0[0]} and age is {0[1]}'.format(lis))

person = {'name': 'Jack', 'age': 32}
print('my name is {0[name]} and age is {1[age]}'.format(person, person)) #or
print('my name is {0[name]} and age is {0[age]}'.format(person)) #or
print('my name is {name} and age is {age}'.format(**person))

class Person:
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

p1 = Person('jack', 34)
print('my name is {0.name} and age is {0.age}'.format(p1))

'''Note: 1.2f means, at least 1 digit before decimal and 2 digits after decimal'''

'''''''''''''''''''''''''''''''''Python Functions'''''''''''''''''''''''''''''''''

# Lambda, Map, Reduce, Filter, Zip, Enumerate, Counter, Cut
'''''''''''''''''''''''''''Lambda'''''''''''''''''''''''''''

def squaring(num):
    return num ** 2
lambda num : num ** 2

rev = lambda x : x[::-1]
rev('this is funny')

(lambda x: x ** 2)(3) # can be used with filter for effective implementation

'''''''''''''''''''''''''''Map'''''''''''''''''''''''''''

'''Map function maps a predefined function to every element in a sequence/list.
Filter function filters out every element in a sequence that meets the condition in a predefined function
Filter is applied to check boolean condition, map is applied for performing lambda operation
'''

def square_roots(x):
    return x ** 2
lis = [2, 4, 6, 8]
list(map(square_roots, lis)) # Or
list(map(lambda x: round(sqrt(x), 2), lis))

alist = [24, 53, 66, 44, 71, 89, 19, 44]
map(squaring, alist)
list(map(squaring, alist))
list(map(lambda num: num * 2, alist))

def cube_root(x):
    return x ** 3
list(map(cube_root, lis))
list(map(lambda x: x ** 3, lis))

a = [3,6,9]
b = [1,7,3]
c = [6,2,7]
list(map(lambda x, y, z : round(x + y / z, 2), a, b, c))

'''''''''''''''''''''''''''Reduce'''''''''''''''''''''''''''
from functools import reduce

lis = [2, 4, 6, 8, 1, 5, 7, 10]
reduce(lambda x, y: x + y, lis)
max_find = lambda a, b: a if (a > b) else b
max_find(56, 24)
reduce(max_find, lis)

'''''''''''''''''''''''''''Filter'''''''''''''''''''''''''''

list(filter(lambda num : num % 2 == 0, alist))
set(filter(lambda num : num % 2 == 0, alist))

list(filter(lambda x: x % 2 != 0, lister))
set(filter(lambda y: y % 5 == 0, lister))

def even_check(x):
    return x % 2 == 0
list(filter(even_check, lister))
list(filter(lambda x: x > 3, lister))

'''Zip (opp of tuple unpacking) & Unzip

https://www.youtube.com/watch?v=KssW4mF0h0c
https://www.youtube.com/watch?v=bOGmYvtw-kk
'''

a = [323,743,224,654,737]
b = [253,733,264,921,355]
for x, y in zip(a, b):
    if x > y:
        print(x)
    else:
        print(y)
        #OR
for pair in zip(a, b):
    print(max(pair))
    #OR
list(map(lambda pair: max(pair), zip(a, b)))

l1 = [3, 7, 2, 1, 3]
l2 = [36, 88, 32, 56]

z = zip(l1, l2)
print(list(z))

a = [323,743,224,654,737] # or
a = [323,743,224,654,737][2:]
lis = [2, 4, 6]
list(zip(a, lis))

keyss = ['a','b','c','d','e']
lst = list(zip(keyss, [x ** 2 for x in range(5)]))

d1 = {'a': 1, 'b': 2}
d2 = {'c': 1, 'd': 2}
list(zip(d1, d2))
list(zip(d2, d1.values()))
list(zip(d1.values(), d2))

person = ['ram', 'shyam', 'dram', 'kaam', 'abc']
person = ['ram', 'shyam', 'dram', 'kaam', 'abc'][1:]
age = [53, 22, 62, 78]
person_age = list(zip(person, age))
person_age
person_age_unzip = list(zip(*person_age))
persons = person_age_unzip[0]
ages = person_age_unzip[1]
print(persons)
print(ages)

for i in zip(range(len(person)),person):
    print(i)
# Or
for i in enumerate(person):
    print(i)

'''''''''''''''''''''''''''Enumerate'''''''''''''''''''''''''''

'''The enumerate() method adds counter to an iterable and returns it (the enumerate object). This enumerate
object can then be used directly in for loops or be converted into a list of tuples using list() method'''

a = [323,743,224,654,737]
for x, y in enumerate(a):
    print(x, y)

for i in enumerate(a, start = 10):
    print(i)

for x, y in enumerate(a):
    if(x > 2):
        break
    else:
        print(y)

person = ['ram', 'shyam', 'dram', 'kaam', 'abc']
for i in enumerate(person):
    print(i)

example = ['left', 'right', 'up', 'down']
for i in range(len(example)):
    print(i, example[i])
for i, j in enumerate(example):
    print(i, j)

new_dict = dict(enumerate(example))
print(new_dict)

[print(i, j) for i, j in enumerate(new_dict)]
[print(i, j) for i, j in enumerate(new_dict.values())]

grocery = ['bread', 'milk', 'butter']
enumerateGrocery = enumerate(grocery)

print(type(enumerateGrocery))
# converting to list
print(list(enumerateGrocery))
# changing the default counter
enumerateGrocery = enumerate(grocery, start = 11)
print(list(enumerateGrocery))

'''''''''''''''''''''''''''Counter'''''''''''''''''''''''''''
'''subclass of dictionary, that keeps a count of the hashable objects. elements in counter are stored as
dictionary keys and count of objects are stored as values'''

lis = [2,4,7,4,3,5,7,4,2,5,7]
count = Counter(lis)
count[4]

string = 'sreekanth'
count = Counter(string)
count['e']

string = 'I just wanna practice collections class in python so that i just can use python collections'
words = string.split(' ')
count = Counter(words)
# Playing wit some methods in Counter object
list(count.elements())
list(count)
set(count)
dict(count)
count.get('just')
count.items()
count.keys()
count.values()
count.most_common(3)
count.most_common()[:-4-1:-1] # 4 least common elements
count.pop('just')
count.popitem()
count.setdefault('have')
count.setdefault('has')
count
count.clear()

'''''''''''''''''''''''''''Cut''''''''''''''''''''''''''' # Ref: https://www.youtube.com/watch?v=8idAqRe0oiI

score_card = [50,75,25,100,86,13,34,88,51,10]
bins = [0, 25, 50, 75, 100]
rank = ['poor', 'avg', 'good', 'better']
pd.cut(score_card, bins = bins, labels = rank)

test = pd.DataFrame({'days': [0,31,45]})
test['range'] = pd.cut(test.days, [0,30,60], include_lowest = True)
test

# method to switch key to values and values to keys in a dictionaries
a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}

def switch_keys_values(dic):
    k = list(a_dict.keys())
    v = list(a_dict.values())
    new_dict = {x: y for x, y in zip(v, k)}
    return new_dict
switch_keys_values(a_dict)

# method to switch key to values and values to keys in two distinct dictionaries
d1 = {'a': 1, 'b': 2}
d2 = {'c': 3, 'd': 4}

def switch_kv(d1, d2):
    keys1 = list(d1)
    keys2 = list(d2)
    val1 = list(d1.values())
    val2 = list(d2.values())
    new_d1 = {x: y for x, y in zip(val2, keys1)}
    new_d2 = {x: y for x, y in zip(val1, keys2)}
    return new_d1, new_d2
switch_kv(d1, d2)

'''''''''''''''''''''''''''''''''''Python Object Oriented Programming, Class & Methods'''''''''''''''''''''''''''''''''''

print('First module is: {}'.format(__name__))
#meaning of __name__ == '__main__' https://www.youtube.com/watch?v=sugvnHA7ElY

'''
In a class, function is called method, attribute is called property.

When a class is instantiated, it creates a blank object and gets passed to __init__ method as self. It can be modified by creating new variables inside the blank object.
When an instance calls a method of a class, the instance automatically populates self as the object which called the function. But when the method is accessed using the class directly, python can not understand, on which object it should apply the function upon. Hence pass instance of the class in the method arguement. First parameter is always self as it refers to the object we are currently acting on/referring to.

Correct order of defining parameter in function are:
    positional parameter or non-default parameter i.e (a,b,c)
    keyword parameter or default parameter i.e (a = 'b',r= 'j')
    keyword-only parameter i.e (*args)
    var-keyword parameter i.e (**kwargs)

ex: def example(a, b, c=None, r="w" , d=[], *args,  **kwargs):
    (a, b) are positional parameter
    (c = none) is optional parameter
    (r = "w") is keyword parameter
    (d = []) is list parameter
    (*args) is keyword only
    (*kwargs) is var-keyword parameter

In a class:
2 types of variables - instance variables and class variables
4 types of methods - regular methods, class methods, static methods, special/dunder/magic methods

class attributes/variables are specific to class i.e., they are shared among all instances of class, not just to the instance of an object. declared outside methods (used for constants).
instance variables are specific to a method in the class

regular methods take instance (self) as 1st arguement
class methods take cls as 1st arguement, declared with @classmethod
static methods do not take any default arguements (viz. self, cls), declared with @staticmethod
'''

# Class keyword

# ex: 1
class Sample(object): # object keyword does not need to be passed since class takes object by default
    pass
x = Sample()
type(x)

# ex: 2
class Dog:

    # class object attribute aka class variable
    species = 'mammal'
    # self accepts instance of a class
    def __init__(self, name, age, breed = 'labrador', fur = True): #aka instance variables
        self.breed = breed
        self.name = name
        self.age = age
        self.fur = fur

doggy = Dog(breed = 'chihuahua', name = 'Robby', age = 2, fur = False)
doggy #gives memory location of object, use __repr__ or __str__ dunder methods for string representation 
doggy.breed
doggy.species
doggy.name
doggy.age
doggy.__dict__ #namespace of doggy object
Dog.__dict__

# ex: 3
class Circle(object):

    # class object attribute
    pi = 3.14

    def __init__(self, radius = 1, perimeter = 2):
        self.radius = radius
        self.perimeter = perimeter
        
    def __repr__(self):
        return 'This is a circle class!'

    def area(self):
        return Circle.pi * (self.radius ** 2)

    def set_radius(self, new_radius):
        '''
        This method takes new radius and resets the existing to the user provided radius
        '''
        self.radius = new_radius

    def get_radius(self):
        return self.radius

    def Perimeter(self):
        return 2 * Circle.pi * self.radius

circle = Circle()
print(circle)
circle.pi
circle.radius
circle.perimeter
circle = Circle(4)
circle.area()
circle.set_radius(23)
circle.radius
circle.get_radius()
circle.Perimeter()

#ex: 4
class Lottery_Player:

    def __init__(self, name, numbers = (3, 8, 1, 5)):
        self.name = name
        self.numbers = numbers

    def total(self):
        return sum(self.numbers)

lot = Lottery_Player('sree')
lot.name, lot.numbers, lot.total()
tot = Lottery_Player('vicky')
tot.numbers = [12, 53, 74, 92]
tot.name, tot.numbers, tot.total()

# ex: 5.1
class Student1: #class hardcoded with variable in function (10)

    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def get_grade(self):
        return self.grade
    
    def get_bonus(self):
        self.grade = self.grade + 10

s0 = Student1('Pill', 57)
s0.get_bonus()
s0.get_grade()

# ex: 5.2
class Student2: #class with class variable

    bonus = 10
    
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def get_grade(self):
        self.grade = self.grade + Student2.bonus

s0 = Student2('Pill', 57)
s0.get_grade()

#ex: 5.3 - using class method
class Student3:
    
    bonus = 10
    no_of_students = 0

    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
        Student3.no_of_students += 1
    
    def get_grade(self):
        return self.grade

    def get_bonus(self):
        self.grade = self.grade + Student3.bonus #can be accessed thru Student3.bonus
        return self.grade

#class variable can be changed directly: from class using class method (Class.classmethod()), by instance of class thru class method (class.classmethod()) or directly accessing class variable from class (Class.class_variable)
    @classmethod
    def set_bonus(cls, new_bonus):
        cls.bonus = new_bonus

Student3.no_of_students # class variable

s0 = Student3('Pill', 57)
s1 = Student3('Bill', 61)
s0.get_bonus()
s1.get_bonus()
s0.__dict__
Student3.__dict__

Student3.bonus = 5 #modifying class variable (and replicating it in instances)
s0.bonus
s1.bonus

s0.bonus = 2 #modifiying instance variable, modifies only for that specific instance
Student3.bonus
s0.bonus
s1.bonus

s0.get_bonus()
s1.get_bonus()

Student3.no_of_students

Student3.set_bonus(12) #same as saying Student3.bonus = 12
Student3.bonus
s0.bonus
s1.bonus
s0.get_bonus()
s1.get_bonus()

s0.set_bonus(14) #even instance of a class can change class variable if called from class method
Student3.bonus
s0.bonus
s1.bonus

#ex: 5.4
class Student4: # using alternative constructor for parsing strings and returning as class variables

    bonus = 10
    
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    @classmethod
    def name_parser(cls, student_string):
        name, grade = student_string.split('-')
        return cls(name, grade)

    def get_bonus(self):
        self.grade = int(self.grade) + Student4.bonus #can be accessed thru Student3.bonus
        return self.grade

stu1 = 'Jack-35'
s1 = Student4.name_parser(stu1)
s1.name
s1.grade
s1.get_bonus()

#ex 5.5
class Person:

    no_of_ppl = 0
    
    def __init__(self, name):
        self.name = name
        Person.no_of_ppl += 1
        
    @classmethod
    def get_total_ppl(cls):
        return cls.no_of_ppl
    
    @property
    def total_ppl(self):
        return Person.no_of_ppl

Person.no_of_ppl = 5
p1 = Person('bill')
p1.no_of_ppl
Person.no_of_ppl
p3 = Person('chill')
Person.no_of_ppl
Person.get_total_ppl()
Person.total_ppl()
p3.total_ppl()
Person.total_ppl
p3.total_ppl

#ex 5.6
class Person:
    
    no_of_ppl = 0
    
    def __init__(self, name = 'John Doe'):
        self.name = name
        Person.add_ppl()
        
    @classmethod
    def get_total_ppl(cls):
        return cls.no_of_ppl
    
    @classmethod
    def add_ppl(cls):
        cls.no_of_ppl += 1

p = Person()
Person.get_total_ppl()
p1 = Person('bill')
p2 = Person('jill')
Person.get_total_ppl()

#ex 5.7 staticmethod - use @staticmethod to organise multiple methods for maintainability
class My_static_methods:

    @staticmethod
    def adder(*args):
        return sum(args)
    
    @staticmethod
    def multiplier(*args):
        prod = 1
        for i in args:
            prod *= i
        return prod

# access above methods as below
My_static_methods.multiplier(3,4,5,2)

#ex: 5.8
class Student: #class hardcoded with class variable in function
    
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
    
    def get_grade(self):
        return self.grade

class Course:
    
    def __init__(self, subject, max_students):
        self.subject = subject
        self.max_students = max_students
        self.students = []
        
    def add_students(self, student):
        if len(self.students) < self.max_students:
            self.students.append(student)
            return True
        return False
    
    def get_average_grade(self):
        val = 0
        for student in self.students:
            val += student.get_grade()
        return val / len(self.students)

    def get_students(self):
        return self.students

s1 = Student('Bill', 21, 45)
s2 = Student('Gill', 19, 65)
s3 = Student('Chill', 20, 86)
s4.get_grade() # same is written as below
Student.get_grade(s3)

course = Course('Maths', 2)
course.add_students(s1)
course.add_students(s2)
course.add_students(s3)
course.add_students(s4)
course.students[1].name
course.students[0].grade
course.get_students()
course.get_average_grade()

#ex 5.9 for @classmethod
class FixedFloat:
    
    def __init__(self, amount):
        self.amount = amount
        
    def __repr__(self):
        return 'Fixed float {}'.format(self.amount)
    
    @staticmethod
    def sums(val1, val2):
        return FixedFloat(val1 + val2)
    
    @classmethod
    def sumss(cls, val1, val2):
        return cls(val1 + val2)

ff = FixedFloat(3454.3) # without @staticmethod decorator
ff.sums(75.8, 35.2)
FixedFloat.sums(75.8, 35.2) # after adding @staticmethod decorator

class Dollar(FixedFloat):
    
    def __init__(self, amount):
        super().__init__(amount)
        self.symbol = '$'
        
    def __repr__(self):
        return 'Dollar {} {}'.format(self.amount, self.symbol)

ff = FixedFloat(34.3)
ff
dol = Dollar(68.2)
dol
dol = Dollar.sums(75.8, 35.2) # prints __repr__ from parent class, not child after using @staticmethod
dol
dol = Dollar.sumss(75.8, 35.2) # prints __repr__ from child class, not parent after using @classmethod
dol

#ex: 6
'''@property gives class attributes getter, setter and deleter functionality. methods can be accessed like attributes esp those that has no attributes apart from self, only returns simple function, dont use to connect to db, file, webservice etc'''
class Employee:
    
    raise_sal = 30
    
    def __init__(self, first = None, last = None, pay = None):
        self.first = first
        self.last = last
        self.pay = pay
    
    @property
    def email(self):
        return self.first + '.' + self.last + '@email.com'
    
    @property
    def fullname(self):
        return self.first + ' ' + self.last

    @fullname.setter #decorator name used for setter is same as method name
    def fullname(self, name):
        self.first, self.last = name.split(' ')
        
    @fullname.deleter
    def fullname(self):
        print('Name deleted!')
        self.first = None
        self.last = None

emp1 = Employee('jack', 'ass', 34)
emp2 = Employee('stupid', 'ass', 26)
emp3 = Employee()

emp1.first
emp1.last
emp1.email
emp1.fullname() #before adding property decorator emp1.fullname()

emp2.first
emp2.last
emp2.email
emp2.fullname #after adding property decorator emp1.fullname()
emp2.last = 'moron' #after using property decorator
emp2.email
emp2.fullname

emp3.fullname = 'dumb ass' # gives AttributeError: can't set attribute. use setter to set this input as attributes
emp3.first
emp3.last
emp3.email

del emp1.fullname #uses deleter decorator
emp1.first
emp1.last

'''''''''Inheritance'''''''''

'''inheritance can be achieved by:
    super().__init__(attributes)
    Parent.__init__(self, attributes)
    use super() for single parent class, use Parent1.__init__() or Parent2.__init__() for multiple parent classes
'''
#ex: 1
class Animal(object):

    def __init__(self):
        print ('Animal created')

    def whoamI(self):
        print('Animal')

    def eat(self):
        print('Eating')

ani = Animal()
ani.whoamI()
ani.eat()

class Dog(Animal):

    def __init__(self):
        Animal.__init__(self) # or use super().__init__()
        print('Dog created')

    def whoamI(self):
        print('Dog')

    def bark(self):
        print('Bow Bow')

dog = Dog()
dog.eat()
dog.whoamI()
dog.bark() # or Dog.bark(dog)

help(Dog) #shows Method resolution order

#ex: 2
class Vehicle:
    
    color = 'white'
    
    def __init__(self, name, max_speed, mileage):
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage

    def __repr__(self):
        return 'Color: {} Vehicle: {} Speed: {} Mileage: {}'.format(Vehicle.color, self.name, self.max_speed, self.mileage)

    def seating_capacity(self, capacity):
        return f'The seating capacity of a {self.name} is {capacity} passengers'

class Bus(Vehicle):
    
    def __init__(self, name, max_speed, mileage):
        super().__init__(name, max_speed, mileage)
    
    def seats_capacity(self, capacity = 50):
        return super().seating_capacity(capacity = 50) #inheriting methods from parent class (with default parameter)

vehicle1 = Vehicle('Audi', 280, 7)
print(vehicle1)
vehicle2 = Bus('school volvo', 180, 12)
print(vehicle2)
vehicle1.seating_capacity(5)
vehicle2.seats_capacity()

#ex: 3
class Employee:
    
    raise_sal = 30
    
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first+'.'+last+'@email.com'

    def fullname(self):
        return self.first+' '+self.last
    
    def apply_raise(self):
        self.pay = self.pay + Employee.raise_sal

emp1 = Employee('jack', 'ass', 34)
emp2 = Employee('stupid', 'ass', 26)

emp1.email

class Developer(Employee):
    
    raise_sal = 45
    
    def __init__(self, first, last, pay, prog_lang):
        Employee.__init__(self, first, last, pay)
        self.prog_lang = prog_lang

dev1 = Developer('moron', 'ass', 34, 'python')
dev2 = Developer('dumb', 'ass', 26, 'java')

dev1.email
dev2.prog_lang

dev1.raise_sal
emp1.raise_sal

class Manager(Employee):

    def __init__(self, first, last, pay, employees = None):
        Employee.__init__(self, first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees
    
    def add_employees(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)
    
    def remove_employees(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)
    
    def print_employees(self):
        for i in self.employees:
            print('--> ' + i.fullname())

mgr1 = Manager('danny', 'son', 3400, [dev1])
mgr1.email
mgr1.add_employees(emp1)
mgr1.print_employees()
mgr1.remove_employees(dev1)
mgr1.print_employees()

isinstance(mgr1, Employee) # checks if an object is an instance of a specified class
isinstance(mgr1, Developer)
isinstance(mgr1, Manager)
issubclass(Developer, Employee) #checks if a class is a child class of specified class
issubclass(Manager, Developer)

# Inheritance: multiple, hierarchical
class Person:

    def __init__(self, personName, personAge):
        self.name = personName
        self.age = personAge
        print('Person')

class Student:

    def __init__(self, stuId):
        self.stuId = stuId
        self.major = 'physics'
        print('Student')

class Resident(Person, Student): # if a child class inherits 2 parent classes, do not use super().__init__(), instead use specific parent class names and attributes that child class needs to use

    def __init__(self, names, ages, _ids, school):
        Person.__init__(self, names, ages) # or use super().__init__(names, ages)
        Student.__init__(self, _ids)
        self.school = school
        print('Resident person & student')

res = Resident('John', 30, 'WQ102', 'Xaviers')
res.age, res.name, res.stuId, res.school

# conflicts with multiple inheritance
class A:

    def __init__(self):
        self.name = 'John'
        self.age = 23

    def getName(self):
        return self.name

class B:

    def __init__(self):
        self.name = 'Richard'
        self.id = '32'

    def getName(self):
        return self.name

class C(A, B):
    # first constructor called is of A. So, value of name in C becomes same as of name in A.
    # But when constructor of B is called, value of name in C is overwritten by value of name in B
    def __init__(self):
        A.__init__(self)
        B.__init__(self)

    def getName(self):
        return self.name

c = C()
c.getName() # hierarchy is dependent on order of __init__() calls inside the subclass, so

# MRO works in a depth first, left to right

'''
super() in __init__ method indicates class that's in next hierarchy. At first the super() of C indicates A.
Then super() in constructor of A searches for its superclass. If it doesnt find any, it executes rest of the
code and returns. So order of constructor calls is: C -> A -> B.
Once constructor of A is called and attribute 'name' is accessed, it doesn’t access attribute 'name' in B
'''
class A:

    def __init__(self):
        super().__init__()
        self.name = 'John'
        self.age = 23

    def getName(self):
        return self.name

class B:

    def __init__(self):
        super().__init__()
        self.name = 'Richard'
        self.id = '32'

    def getName(self):
        return self.name

class C(A, B):

    def __init__(self):
        super().__init__()

    def getName(self):
        return self.name

c = C()
c.getName()

# MRO method
class First(object):
    
    def __init__(self):
        super(First, self).__init__()
        print("first")

class Second(object):
    
    def __init__(self):
        super(Second, self).__init__()
        print("second")

class Third(First, Second):
    
    def __init__(self):
        super(Third, self).__init__()
        print("third")

th = Third()

#ex: 1
class Student:
    
    def __init__(self, name, school):
        self.name = name
        self.school = school
        self.marks = []
    
    @property
    def average(self):
        return sum(self.marks) / len(self.marks)

    @classmethod
    def go_to_school(cls):
        print('M going to school')
        print(cls)
        
    @staticmethod
    def go_to_work():
        print('Will go to work')
    
    @classmethod
    def friend(cls, origin, frnd, *args, **kwargs):
        return cls(frnd, origin.school, args, kwargs)

anna = Student('Anna', 'MIT')
anna.marks.append(31)
anna.marks.append(53)
anna.marks
anna.average
anna.go_to_school()
sree = Student('Sree', 'Harvard')
sree.marks.append(4)
sree.marks.append(2)
sree.marks.append(8)
sree.marks
sree.average
sree.go_to_school() # or
Student.go_to_school()
Student.go_to_work()
frnds = sree.friend('Fred')
frnds.name, frnds.school
frnds.marks.append(53)
frnds.marks.append(48)
frnds.marks
frnds.average()
new_frnd = frnds.friend('Greg')
new_frnd.name

class WorkingStudent(Student):
    
    def __init__(self, name, school, salary, job):
        super().__init__(name, school) # or use Student.__init__(self, name, school)
        self.salary = salary
        self.job = job

anna = WorkingStudent('Anna', 'MIT', 23500.24, 'lawyer')
anna.name, anna.school, anna.salary, anna.job
prend = WorkingStudent.friend(anna, 'bablu', 16500.99, 'developer', 24, lover = 'yes')
prend.name, prend.school, prend.salary, prend.job

'''dunder methods (double underscore)/special methods/magic methods'''
# built in methods that can be used on objects created through a created class: init, enter, exit, repr, str, delete, len. __init__ method initializes attributes of objects in a class, self in init definition references instance object of a class
class Book(object):

    def __init__(self, title, author, pages):
        print('Book is created')
        self.title = title
        self.author =  author
        self.pages = pages

    def __str__(self): # user oriented description
        return 'Title with %s is written by %s that has %i pages (str return)' %(self.title, self.author, self.pages)
    
    def __repr__(self): # code oriented description, gives string representation of an object. prefer this to __str__
        return 'Title with %s is written by %s that has %i pages (repr return)' %(self.title, self.author, self.pages)

    def __len__(self):
        return self.pages

    def __del__(self):
        print('Book is deleted')

book = Book('Python', 'Sreekanth', 532)
print(book)
book.title
book.author
len(book)
del(book)
book.title

'''''''''''''''''''''''''''Decorators'''''''''''''''''''''''''''

#ex: 0
def method_one(method_two):
    return method_two()

def add():
    return 1 + 2

method_one(add)

#ex: 1
def start_and_end(func):

    def wrapper():
        print('start')
        func()
        print('end')
    return wrapper

# above can be used in one of below ways
def name_print():
    print('hey there')

name_print = start_and_end(name_print)
name_print()

# or
@start_and_end
def name_prints():
    print('hey there')

name_prints()

#ex: 2
def start_and_end(func):
    
    def wrapper(*args, **kwargs):
        print('start')
        result = func(*args, **kwargs)
        print('end')
        return result
    return wrapper

@start_and_end
def add(x):
    return x + 5

print(add(5))

#ex: 3
import functools

def my_decorator(func):
    
    @functools.wraps(func)
    def func_running_func():
        print('In d func')
        func()
        print('Out of func')
    return func_running_func

@my_decorator
def my_func():
    print('m d function!')

my_func()

#ex: 4
def decorator_with_args(number):

    def my_decorator(func):
        @functools.wraps(func)
        def function_running_func():
            print('In d decorator')
            if number % 2 == 0:
                print('Even no')
            else:
                func()
            print('Out of decorator')
        return function_running_func
    return my_decorator

@decorator_with_args(35)
def my_func2():
    print('hello')

my_func2()

#ex: 5
def decorator_with_args(number):
    def my_decorator(func):
        @functools.wraps(func)
        def function_running_func(*args, **kwargs):
            print('In d decorator')
            if number % 2 == 0:
                print('Even no')
            else:
                func(*args, **kwargs)
            print('Out of decorator')
        return function_running_func
    return my_decorator

@decorator_with_args(35)
def my_func3(x, y):
    print(x ** y)

my_func3(2, 3)

'''''''''''''''''''''''''''Generators'''''''''''''''''''''''''''
# for better memory management for bigger lists. raises StopIteration error if it doesnt reach next yield statement
def generate():
    for i in range(5):
        yield i * 2

for i in generate():
    print(i)

g = generate()
next(g)

s = 'hello'
next(s)
s = iter(s)
next(s)

'''Error & exception handling'''

try:
    2 + 'T'
except:
    print('Exception caught!')
finally:
    print('Finally block here')

try:
    file = open('test_1', 'w') # replace with 'r' and check
    file.write('Typing a test file')
except:
    print('Sorry, unable to write to the file!')
else:
    print('File written successfully')

def ask_int():
    try:
        val = int(input('Please input an integer '))
    except:
        print('Please enter only an integer and not a string!')
        val = int(input('Try to input an integer again '))
    finally:
        print('Thanks for trying')
    print('The entered input is {}'.format(val))

ask_int()

# Improving the above code
def ask_for_int():
    while True:
        try:
            val = int(input('Please input an integer: '))
        except:
            print('Your input is not an integer, please try again!')
            continue
        else:
            print('Thanks for the integer input')
            break
        finally:
            print('Thanks for trying!')
    print('The entered input is {}'.format(val))

ask_for_int()

'''Assessment'''

# Prob 1
try:
    for i in ['a', 'b', 'c', 'd']:
        print(i ** 2)
except:
    print('Given element is not an integer to perform the operation')

# Prob 2
try:
    x = 5
    y = 0
    z = x / y
except:
    print('Division error!, check and try again')
finally:
    print('All done')

# Prob 3
def ask_input():
    while True:
        try:
            val = int(input('Please enter an integer to find square of it '))
        except:
            print('Try again, please enter an integer ')
            continue
        else:
            break
        finally:
            print('Thanks for trying!')
    print('The square for input entered is {}'.format(val ** 2))

ask_input()

# Prob 4
class Line(object):

    def __init__(self, coor1, coor2):
        self.coor1 = coor1
        self.coor2 = coor2

    def distance(self):
        x1, y1 = self.coor1
        x2, y2 = self.coor2
        return round(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 2)

    def slope(self):
        x1, y1 = self.coor1
        x2, y2 = self.coor2
        return round((y2 - y1) / (x2 - x1), 2)

# Example output
coordinate1 = (3, 2)
coordinate2 = (8, 10)
line = Line(coordinate1, coordinate2)
line.distance()
line.slope()

# Prob 5
class Cyclinder(object):

    pi = 3.14

    def __init__(self, height = 1, radius = 1):
        self.height = height
        self.radius = radius

    def volume(self):
        return Cyclinder.pi * (self.radius **2) * self.height
    # v = pi * (r **2) * h

    def surface_area(self):
        return (2 * Cyclinder.pi * self.radius * self.height) + (2 * Cyclinder.pi * self.radius ** 2)
    # A = (2 * pi * r * h) + (2 * pi * r ** 2)

cyc = Cyclinder(2, 3)
cyc.volume() # 56.52
cyc.surface_area() # 94.2

'''Using string class'''
# string methods: find() vs index() - find returns -1 if element is not found, index gives ValueError
import string
string.digits
string.ascii_letters
string.punctuation

txt = 'Hello, welcome to my world'

txt.find(value, start, end) # Syntax
txt.find('to')
txt.find('to', 0, len(txt))
txt.find('s')

txt.index(value, start, end) # Syntax
txt.index('to')
txt.index('to', 0, len(txt))
txt.index('s')

def into_two(x):
    return x * 2
df1['data1'].apply(into_two)
df1['data1'].apply(lambda x : x * 2)
df1.drop('data1', axis = 1)
df1.columns
df1.index
df1.sort_index(ascending = False)
df1.sort_values('keys')
df1.isnull().sum()

'''''''''''''''''''''''''''OS Operations'''''''''''''''''''''''''''
import os

dir(os) # gives all functions in os module as list
# returns current working directory
os.getcwd()
# changing/navigating to specified directory
os.chdir(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS')
os.getcwd() # verifying if it changed to specified working directory
# shows contents of specified directory
os.listdir(r'D:\Programming Tutorials')
# creating new directory
os.mkdir(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\test folder')
# creates top level directory if not present for creating specified directory
os.makedirs(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\test folder\1 step\new folder')
# deleting specified directory
os.rmdir(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\test folder')
# deletes top level directory and its sub directories if any
os.removedirs(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\test folder\1 step\new folder')
# rename single file
os.rename(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\URL.txt', 'url.txt') #file to be named, new name
os.listdir()
# attributes about specific directory or file, like size, last modified etc
os.stat(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS')
# top down tree view of directories & files from specific directory. Outputs tuple of directory path, sub folders, files
directory_tree = os.walk(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS')
for dirpath, dirnames, filenames in directory_tree:
    print(dirpath)
    print(dirnames)
    print(filenames)
# environment variables
os.environ
os.environ.get('HOMEPATH') # or os.environ['HOMEPATH']
# using specific path and creating files in that directory
file_path = os.environ['HOMEPATH'] + 'new_file.txt' # misses '/' and concatenates directly to directory. either use / with filename string
# OR use join()
file_path = os.path.join(os.environ['HOMEPATH'], 'new_file.txt')
# functions in os.path module
os.path.basename(r'D:\Programming Tutorials\Machine Learning') #gives base directory name
os.path.dirname(r'D:\Programming Tutorials\Machine Learning') #gives path to base directory
os.path.split(r'D:\Programming Tutorials\Machine Learning') #gives path to base directory & base directory name
file = os.path.split(r'D:\Programming Tutorials\Machine Learning') #gives path to base directory & base directory name
file[0], file[1]
# splitting file extension & base path of the file
os.path.splitext('test_folder/new.txt')
os.path.exists(r'D:\Programming Tutorials\Machine Learning') #checking if file/directory exists, returns boolean
os.path.isdir(r'D:\Programming Tutorials') #checking if its a directory, returns boolean True if directory
os.path.isdir(r'D:\Programming Tutorials\Python\Context Managers.mp4') #returns boolean False if file
os.stat(r'C:\Users\Srees\Desktop\URLs.txt').st_size == 0 #check if file is empty


'''''''''''''''''''''''''''Context Manager to read & write file objects'''''''''''''''''''''''''''

# FileIO aka context manager: r to read text, w to write text to file, r+ to read & write text, a to append text as w overwrites existing file contents
# rb to read bytes, wb to write bytes. use for multi media objects

# reading & writing to files without context manager
f = open(r'C:\Users\Srees\Desktop\URLs.txt', 'r')
f # contains attributes name, mode, encoding
f.name # file name
f.close() # explicitly close file after opening n performing operations else it causes leaks. So use context manager

f = open(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\URL.txt', 'w')
f.write('hey')
f = open(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\URL.txt', 'r')
f.read()
f.closed # verifying if file is closed
f.close()

# opening files using context manager. closes file automatically
with open(r'C:\Users\Srees\Desktop\URLs.txt', 'r') as f: #'r' doesnt need to be mentioned as it is attribute by default
    f_contents = f.read() # returns entire content
    print(f_contents)
with open(r'C:\Users\Srees\Desktop\URLs.txt', 'r') as f:
    f_contents = f.read(100) # returns specified no of characters
    print(f_contents)
with open(r'C:\Users\Srees\Desktop\URLs.txt', 'r') as f:
    f_contents = f.readline()
    print(f_contents) # f_contents[4] returns specific character
with open(r'C:\Users\Srees\Desktop\URLs.txt', 'r') as f:
    f_contents = f.readline(5) # returns characters until specified index
    print(f_contents) # f_contents[4] returns specific character
with open(r'C:\Users\Srees\Desktop\URLs.txt', 'r') as f:
    f_contents = f.readlines() # returns each line as element in list, hence list slicing & indexing can be used
    print(f_contents) # f_contents[5] returns specified line in file
    ### same as f.read but readlines gives lines as list
with open(r'C:\Users\Srees\Desktop\URLs.txt', 'r') as f:
    for line in f:
        print(line, end = '') # if end is not specified, it returns each line with blank space after it

print(f.closed)

# reading only specified no of lines at a time
with open(r'C:\Users\Srees\Desktop\URLs.txt', 'r') as f:
    size_to_read = 10
    f_contents = f.read(size_to_read)
    while len(f_contents) > 0:
        print(f_contents)
        f_contents = f.read(size_to_read)
        f.tell() # shows current position of cursor
        f.seek(4) # reads file from specified index position

# writing to a file, overwrites existing file contents
with open(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\URL.txt', 'w') as f:
    f.write('test')

# appending to a file, adds to existing file contents
with open(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\URL.txt', 'a') as f:
    f.write('\ntest2')

# reading from 1 file and writing same content to another file
with open(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\URL.txt', 'r') as rf:
    with open(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\URL2.txt', 'w') as wf:
        for line in rf:
            wf.write(line)

# reading, copying multi media files. works in binary mode i.e., it reads & writes in bytes not text
with open(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\fine.png', 'rb') as rf:
    with open(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\fine2.png', 'wb') as wf:
        for line in rf:
            wf.write(line)

# using class for context manager for reading & writing objects. https://www.youtube.com/watch?v=Lv1treHIckI
class File:
    
    def __init__(self, filename, method):
        self.file = open(filename, method)
        
    def __enter__(self):
        print('entering file..')
        return self.file
    
    def __exit__(self, type, value, traceback): # can be used to handle exceptions inside dunder exit
        print('{}, {}, {}'.format(type, value, traceback))
        print('exiting file..')
        self.file.close()
        if type == FileNotFoundError:
            return True #use this if above exception is expected and can be handled which does not need traceback. do not use if exceptions need to be handled

with File(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\URL2.txt', 'w') as f:
    print('performing operation..')
    f.write('heylo')
    raise Exception()

# using decorator on a generator to make it as context manager
from contextlib import contextmanager

@contextmanager
def file(filename, method):
    print('enter..')
    file = open(filename, method)
    yield file
    file.close()
    print('exit..')

with file(r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\URL2.txt', 'w') as f:
    print('operation in progress..')
    f.write('hey!')

'''''''''''''''''''''''''''Read & Write Excel, CSV File Objects'''''''''''''''''''''''''''

import csv

# reading csv files using context manager
with open(r'C:\Users\Srees\Desktop\names.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    # print(csv_reader)
    next(reader) # skipping 1st line by stepping over iterables using next
    for line in reader:
        # print(line)
        print(line[2]) # indexing for specific values

# writing csv files using context manager
with open(r'C:\Users\Srees\Desktop\names.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    with open(r'C:\Users\Srees\Desktop\names2.csv', 'w') as new_file:
        writer = csv.writer(new_file, delimiter = ';')
        for line in reader:
            writer.writerow(line)

# reading csv files using context manager & DictReader
with open(r'C:\Users\Srees\Desktop\names.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for line in csv_reader:
        # print(line)
        print(line['email'])

# writing csv files using context manager & DictReader
with open(r'C:\Users\Srees\Desktop\names.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    with open(r'C:\Users\Srees\Desktop\names2.csv', 'w') as new_file:
        field_names = csv_reader.fieldnames # select only required fields by indexing like csv_reader.fieldnames[:2]
        csv_writer = csv.DictWriter(f = new_file, fieldnames = field_names, delimiter = ';')
        csv_writer.writeheader() # writes header else it omits while writing to csv
        for line in csv_reader:
            # del line['email'] # remove any column if not required to write
            csv_writer.writerow(line)

# reading using xlrd
import xlrd

file = r'D:\Programming Tutorials\Machine Learning\Workspace MLDLDS\Book1.xlsx'
workbook = xlrd.open_workbook(file)
sheet = workbook.sheet_by_name('position')
# sheet.cell_value(0, 1)
for row in range(sheet.nrows):
    print(sheet.cell_value(row, 0))
    for col in range(sheet.ncols):
        print(sheet.cell_value(row, col))

'''''''''''''''''''''''''''Regex'''''''''''''''''''''''''''
# https://w3resource.com/python-exercises/pandas/string/index.php

'''
Snippets:
\d+     - \d a digit (0-9)
\D+     - \D a non digit
\s+     - \s whitespace(tab, space, newline etc)
\S+     - \S non whitespace
\w+     - \w alpha-numeric (a-z, A-Z, 0-9, _)
\W+     - \W non-alpha-numeric

[a-z]+        - sequences of lower case letters
[A-Z]+        - sequences of upper case letters
[a-zA-Z]+     - sequences of lower or upper case letters
[A-Z][a-z]+   - 1 upper case letter followed by lower case letters

Meta characters that need to be escaped using back slash '\' are: . [ { ( ) \ ^ $ | ? * +

Anchors: in visible positions before or after characters. can be used in conjunction with other patterns for searching
\b      - Word Boundary: indicated by a whitespace or a non alpha-numeric character
\B      - Not a Word Boundary
^       - Beginning of a String
$       - End of a String

[]      - Matches Characters in brackets
[^ ]    - Matches Characters NOT in brackets
|       - Either Or
( )     - Group

Quantifiers: can match more than 1 character at once
?       - 0 or One
*       - 0 or More
+       - 1 or More
{3}     - Exact no of digits
{3, 4}   - Range of digits in a number (Minimum, Maximum)
'''

text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890

Ha HaHa

Meta characters that need to be escaped using back slash '\' are: . [ { ( ) \ ^ $ | ? * +
So Meta needs to be very costly $Meta

coreyms.com

321-555-4321
123.555.1234
123*555*1234
123_555_1234
800-555-4321
900-555-4321

Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T

Can I call you as Miss..?
under_score

cat
    bat
pat
rat
'''
sentence = 'Start a sentence and then bring it to an end'

x = re.search(pattern = 'art', string = sentence)
x
x[0] # or
sentence[2:5]

patterns = ['words', 'the', 'vast']
text = 'This is just a sample check for finding the word/words using the regex function'

re.findall('the', text) # returns list of all matches, but without their span (index positions)
re.search('the', text) # returns only first occurance of the match
re.match('the', text)

for pattern in patterns:
    if re.search(pattern, text):
        print('Pattern found for "{}"'.format(pattern))
    else:
        print('Pattern not found for "{}"!'.format(pattern))

match = re.search(patterns[0], text)
type(match)
match[0]
match.start()
match.end()

# compile: captures a pattern into a variable which can be reused for multiple searches
pattern = re.compile(r'abc')
matches = pattern.finditer(text_to_search)
[match for match in matches] # gives span (beginning & end index) of matches
print(text_to_search[1:4])

pattern = re.compile(r'.') # returns all characters, spl characters, digits in text except new line
pattern = re.compile(r'\.') # use backslash '\' to escape & search for actual '.' as its a spl character in regex, not just '.'
pattern = re.compile(r'coreyms\.com')
pattern = re.compile('Miss\.\.\?')
pattern = re.compile(r'\d') # matches all digits
pattern = re.compile(r'\D') # matches anything that is NOT a digit
pattern = re.compile(r'\w') # matches all word characters, digits, cap or small letters, or _
pattern = re.compile(r'\W') # matches all characters that are not words, digits, cap or small letters, or _
pattern = re.compile(r'\s') # matches all white spaces, tabs & new line
pattern = re.compile(r'\S') # matches all characters that are not white spaces, tabs & new line
pattern = re.compile(r'\bHa') # word bounds are indicated by white space or non-numeric character
pattern = re.compile(r'\BHa') # not a word boundary
pattern = re.compile(r'\d\d\d.\d\d\d.\d\d\d\d')
# to match only specific special characters, use charset using []. Also, no need to use escape character for . in charset []
pattern = re.compile(r'\d\d\d[*._-]\d\d\d[*._-]\d\d\d\d')
pattern = re.compile(r'[89]00[-.]\d\d\d[-.]\d\d\d\d') # matching 800, 900 nos
'''
- is a special character within charset. If mentioned at beginning or end within [], denotes - as literal, else it
denotes range if mentioned between values
^ is a special character within charset, denotes negation. mentioned at the beginning within charset []
'''
pattern = re.compile(r'[1-5]') # can be used for letter also, as below
pattern = re.compile(r'[a-zA-Z0-9]')
pattern = re.compile(r'[^a-zA-Z0-9]') # matches everything thats not any alpha numeric as is specified
pattern = re.compile(r'[^p]at') # or it can be written as below
pattern = re.compile(r'[cbhr]at')

# Quantifiers
pattern = re.compile(r'\d\d\d.\d\d\d.\d\d\d\d') # this can be written using quantifiers as below
pattern = re.compile(r'\d{3}.\d{3}.\d{4}')
pattern = re.compile(r'\d{3}[._]\d{3}[._]\d{4}')

pattern = re.compile(r'Mr\.')
pattern = re.compile(r'Mr\.?')
pattern = re.compile(r'Mr\.?\s[A-Z]')
pattern = re.compile(r'Mr\.?\s[A-Z]\w+') # matches names beginning with Mr with/out . followed by a cap letter & small letters
pattern = re.compile(r'Mr\.?\s[A-Z]\w*')
pattern = re.compile(r'M(r|s|rs)\.?\s[A-Z]\w*') # () indicates groups
pattern = re.compile(r'(Mr|Ms|Mrs)\.?\s[A-Z]\w*') # same as above

matches = pattern.finditer(text_to_search)
[match for match in matches]

pattern = re.compile(r'^Start')
pattern = re.compile(r'^and')
pattern = re.compile(r'end$')
pattern = re.compile(r'an$')

matches = pattern.finditer(sentence)
[match for match in matches]

emails = '''
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
'''
pattern = re.compile(r'[a-zA-Z]+@[a-zA-Z]+\.com')
pattern = re.compile(r'[a-zA-Z0-9.-]+@[a-zA-Z-]+\.(com|edu|net)')
pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+') # matches all kinds of email IDs

matches = pattern.finditer(emails)
[match[0] for match in matches]

urls = '''
https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov
'''
pattern_ = re.compile('https?://(www\.)?\w+\.\w+') # matches all urls
matches = pattern.finditer(urls)
for match in matches:
    # group(0) means the entire captured/matched text, and can be indexed similarly as per need
    print(match.group(0))
    print((match.group(2), match.group(3)))
[(match.group(0), match.group(2), match.group(3)) for match in matches] # same as above
[(match.group(2), match.group(3)) for match in matches] # subset of groups however, doesnt work if pattern is not groupwise
[match.group(2) + match.group(3) for match in matches] # fetching only domain names
# back references for the sub groups
pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)') # same as above but using groups
sub_urls = pattern.sub(r'\2\3', urls) # works if pattern is groupwise compiled
print(sub_urls)

'''fetching all the phone nos from external file'''
path = r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\regex_data.txt'
pattern = re.compile(r'[89]00[-.]\d\d\d[-.]\d\d\d\d') # matching 800, 900 nos
with open(path, 'r', encoding = 'utf-8') as f:
    contents = f.read()
    matches = pattern.finditer(contents)
    [match for match in matches]

# findall: returns only matches as a list of strings. if it matches groups, it wil return only groups
pattern = re.compile(r'(Mr|Ms|Mrs)\.?\s[A-Z]\w*') # same as above
matches = pattern.findall(text_to_search)
[match for match in matches]

pattern = re.compile(r'\d{3}.\d{3}.\d{4}')
matches = pattern.findall(text_to_search)
[match for match in matches]

pattern = re.compile(r'Start') # returns if matches at beginning, else it gives None if no match found
matches = pattern.match(sentence) # also, match object is not iterable
print(matches)

pattern = re.compile(r'sentence')
matches = pattern.match(sentence) # returns None since string doesnt match at beginning of string
print(matches)

pattern = re.compile(r'sentence') # searches entire string for specified pattern
matches = pattern.search(sentence) # prints only 1st match if it finds, else returns None
print(matches)

# flags: case insensitive flag, verbose flag, multi line flag
pattern = re.compile(r'Start', re.IGNORECASE) # same as re.I
matches = pattern.search(sentence)
print(matches)

#text ending with ing, tion & sion
pattern = re.compile('\w+(tion|sion|ing)\w*')
#text before fullstop & comma
pattern = re.compile('\w+\.')
#text within brackets (multiple bracket types)
pattern = re.compile('(\(|\{|\[)[a-zA-Z0-9\s]*(\)|\]|\})')
#text within double quotes
pattern = re.compile('"[a-zA-Z0-9,.:\s]*"')
#text having apostrophes
pattern = re.compile("\w+['']\w+")

matches = pattern.finditer(text)
[x[0] for x in matches]

# other experiments
split_term = '@'
text = 'the mail, is your mail id the s.srees@live.com?'
re.split(pattern = split_term, string = text)
re.findall('the', text)

phrase = 'sdsd..sssddd..sdddsddd..dsds..dsssss..sdddd'
patterns = ['sd*', 'sd+', 'sd?', 'sd{3}', 'sd{2, 3}', '[sd]', 's[sd]+']

def multi_re_find(patterns, phrase):
    for pattern in patterns:
        print('Searching the phrase using the recheck: {}'.format(pattern))
        print(re.findall(pattern, phrase))
        print('\n')
multi_re_find(patterns, phrase)

m = re.search(r'(ab[cd]?)', """acdeabdabcde""")
m.groups()

# punctuation removal
phrase1 = 'I wonder, "how can i remove punctuation marks!"'
re.findall('[^!.?", ]+', phrase1)

# character sets/ranges
phrase2 = 'This is just a sample check for finding word/words using the regex function'
patterns = ['[a-z]+', '[A-Z]+', '[a-zA-Z]+', '[A-Z][a-z]+']
multi_re_find(patterns, phrase2)

# escape codes
phrase4 = 'This is just a sample check for finding word and numbers 1985 and # tags with dog tab srees16'
patterns = ['\d+', '\D+', '\s+', '\S+', '\w+', '\W+']
multi_re_find(patterns, phrase4)

# filter valid emails from a series
emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
mailers = emails.map(lambda x: bool(re.match(pattern, x))) # as series of strings
emails[mailers]
emails.str.findall(pattern, flags = re.IGNORECASE) # as series of list
[x[0] for x in [re.findall(pattern, email) for email in emails] if len(x) > 0] # as list

#extract only text from whatsapp chat
mys = '''
[3/21, 8:01 AM] Ghd Anusha 9. 12: whatttt
[3/21, 8:01 AM] Ghd Anusha 9. 12: so are you engaged?
[3/21, 8:01 AM] Ghd Anusha 9. 12: did you proposal her😀
'''
pattern = re.compile(': [a-zA-Z0-9\s]*')
match = pattern.finditer(mys)
[x[0].strip(' :\n') for x in match]

sr = 'Python exercises, PHP exercises, C# exercises'
pattern = re.compile('\w+[#]?\sexercises')
matches = pattern.finditer(sr)
[x[0].split(' ')[0] for x in matches]

'''''''''''''''''''''''''''Unit Test''''''''''''''''''''''''

For designing test cases, there are 3 sets of methods:
setUp()
teardown()

skipTest(aMesg:string)
fail(aMesg:string)

id():string
shortDescription():string

    First set are the pre and post test hooks. setUp() begins before each test routine, teardown() after the routine.
    
    Second set controls test execution. Both methods take a message string as input, and both cancel an ongoing test. But skiptest() aborts the current test while fail() fails it completely.
    
    Third set helps determining the test. id() returns a string consisting of the name of the testcase object and of the test routine. And shortDescription() returns docstr comment at the initiation of each test routine.

Types of Assertions:
    assertEqual(a, b) – used to check if the result obtained is equal to the expected result
    assertNotEqual(a, b) - a != b
    assertAlmostEqual(a, b)
    assertNotAlmostEqual(a, b)
    assertTrue() / assertFalse() – used to verify if a given statement is true or false
    assertIs(a, b) - a is b
    assertIsNone(x) - x is None
    assertIn(a, b) - a in b
    assertIsInstance(a, b) - isinstance(a, b)
    Note: assertIs(), assertIsNone(), assertIn(), assertIsInstance() has opposite methods: assertIsNot() et al
    assertGreater(a, b) - a > b
    assertGreaterEqual(a, b) - a >= b
    assertLess(a, b) - a < b
    assertLessEqual(a, b) - a <= b
    assertRegex(s, r) - r.search(s)
    assertNotRegex(s, r) - not r.search(s)
    assertCountEqual(a, b)
    assertRaises() – used to raise a specific exception

3 test outcomes:
OK – This means that all the tests are passed
FAIL – This means that the test did not pass and an AssertionError exception is raised
ERROR – This means that the test raises an exception other than AssertionError

'''
# ex 1: https://www.geeksforgeeks.org/unit-testing-python-unittest
def fun(x):
    return x + 1

class MyTest(unittest.TestCase):

    def test(self):
        self.assertEqual(fun(3), 4)

if __name__ == '__main__':
    unittest.main()

# ex 2: https://realpython.com/python-testing
class TestStringMethods(unittest.TestCase):

    def setUp(self):
        print('Setup called..')
        self.a = 10
        self.b = 12
        
    def test_adds(self):
        print('Test adds called..')
        sums = self.a + self.b
        self.assertEqual(sums, 22)

    # Returns True if the string contains 4 A's
    def test_strings_a(self):
        self.assertEqual( 'a'*4, 'aaaa', msg = 'That is correct!') # msg is for our custom error msg

    # Returns True if the string is in upper case
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO', msg = 'That is correct!')

    # Returns TRUE if the string is in uppercase else returns False
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    # Returns true if the string is stripped and matches the given output
    def test_strip(self):
        s = 'geeksforgeeks'
        self.assertEqual(s.strip('geek'), 'sforgeeks', msg = 'That is correct!')

    # Returns true if the string splits and matches the given output
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()

# ex 3: https://www.journaldev.com/15899/python-unittest-unit-test-example

class Person:

    name = []

    def set_name(self, user_name):
        Person.name.append(user_name)
        return len(self.name) - 1

    def get_name(self, user_id):
        if user_id >= len(self.name):
            return 'There is no such user'
        else:
            return self.name[user_id]

if __name__ == '__main__':
    person = Person()
    print('User Abbas has been added with id ', person.set_name('Abbas'))
    print('User associated with id 0 is', person.get_name(0))

person = Person()
person.set_name('jack ass')
person.set_name('dumb ass')
person.get_name(1)

# unit test for above code
import Person # import the class we want to test

class Test(unittest.TestCase):
    """
    Basic class inherits unittest.TestCase
    """
    person = Person() # instantiate the Person Class
    user_id = [] # variable that stores obtained user_id
    user_name = [] # variable that stores person name

    '''methods starting with 'test' are considered as a test case'''
    def test_set_name(self): # test case function to check the Person.set_name function
        print("Start set_name test\n")
        for i in range(4):
            name = 'name_' + str(i) # initialize a name
            self.user_name.append(name) # store the name into the list variable
            user_id = self.person.set_name(name) # get the user id obtained from the function
            self.assertIsNotNone(user_id) # check if obtained user id is null or not, null user id will fail test
            self.user_id.append(user_id) # store the user id to the list
        print("user_id length = ", len(self.user_id))
        print(self.user_id)
        print("user_name length = ", len(self.user_name))
        print(self.user_name)
        print("\nFinish set_name test")

    def test_get_name(self): # test case function to check the Person.get_name function
        print("\nStart get_name test\n")
        length = len(self.user_id)  # total number of stored user information
        print("user_id length = ", length)
        print("user_name length = ", len(self.user_name))
        for i in range(6):
            if i < length: # if i not exceed total length then verify the returned name
                # if two names doesn't match, test case would fail
                self.assertEqual(self.user_name[i], self.person.get_name(self.user_id[i]))
            else:
                print("Testing for get_name no user test")
                self.assertEqual('There is no such user', self.person.get_name(i)) # if length exceeds, check 'no such user' type message
        print("\nFinish get_name test\n")

if __name__ == '__main__':
    unittest.main()


'''''''''''''''''''''''''''Speech Recognition'''''''''''''''''''''''''''

import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print('speak..')
    audio = r.listen(source)
print(r.recognize_google(audio))


'''''''''''''''''''''''''''Logging'''''''''''''''''''''''''''
#https://www.youtube.com/watch?v=-ARI4Cz-awo
#https://www.youtube.com/watch?v=jxmzY9soFXg
import logging

'''Levels of loggin:
    Debug
    Info
    Warning: default level
    Error: default level
    Critical: default level
'''

logging.basicConfig(level = logging.DEBUG) #, filename = 'test.log', format = '%(asctime)s:')

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y

a = 32
b = 10

add_result = add(a, b)
print('Add: {} + {} = {}'.format(a, b, add_result))
logging.warning('Add: {} + {} = {}'.format(a, b, add_result))
logging.error('Add: {} + {} = {}'.format(a, b, add_result))
logging.critical('Add: {} + {} = {}'.format(a, b, add_result))

sutract_result = subtract(a, b)
print('Subtract: {} + {} = {}'.format(a, b, sutract_result))
logging.warning('Subtract: {} + {} = {}'.format(a, b, sutract_result))
logging.error('Subtract: {} + {} = {}'.format(a, b, sutract_result))
logging.critical('Subtract: {} + {} = {}'.format(a, b, sutract_result))

multiply_result = multiply(a, b)
print('Multiply: {} + {} = {}'.format(a, b, multiply_result))
logging.warning('Multiply: {} + {} = {}'.format(a, b, multiply_result))
logging.error('Multiply: {} + {} = {}'.format(a, b, multiply_result))
logging.critical('Multiply: {} + {} = {}'.format(a, b, multiply_result))

divide_result = divide(a, b)
print('Divide: {} + {} = {}'.format(a, b, divide_result))
logging.warning('Divide: {} + {} = {}'.format(a, b, divide_result))
logging.error('Divide: {} + {} = {}'.format(a, b, divide_result))
logging.critical('Divide: {} + {} = {}'.format(a, b, divide_result))

'''''''''''''''''''''''''''Memory Management & Garbage Collection'''''''''''''''''''''''''''

x = 10
y = x
id(x) == id(y)

# https://stackabuse.com/basics-of-memory-management-in-python
import gc, sys
print(gc.get_threshold())
gc.collect()

def create_cycle():
    list = [8, 9, 10]
    list.append(list)
    return list
create_cycle()
# Manual GC
def create_cycle():
    list = [8, 9, 10]
    list.append(list)

def main():
    print("Creating garbage...")
    for i in range(8):
        create_cycle()
    print("Collecting...")
    n = gc.collect()
    print("Number of unreachable objects collected by GC:", n)
    print("Uncollectable garbage:", gc.garbage)

if __name__ == "__main__":
    main()
    sys.exit

'''''''''''''''''''''''''''Numpy'''''''''''''''''''''''''''

np.__version__
np.show_config()

my_list = [1, 6, 3 ,8]
np.array(my_list)

my_matrix = [35, 74, 22, 78, 62, 21, 83, 72]
np.array(my_matrix).reshape(2, 4)
my_matrix1 = [[35, 74, 22, 78],[62, 21, 83, 72]]
np.array(my_matrix1).reshape(4, 2)
np.array(my_matrix1).reshape(2, 4)
np.arange(0, 10, 3) # array range between lower n upper bounds with step size
np.zeros(4)
np.zeros(4).size
np.zeros(4).itemsize
memory_size = np.zeros(10).size * np.zeros(10).itemsize
np.zeros((3, 4))
np.ones(5)
np.ones([4, 5])
np.eye(4) # identity matrix

x = np.ones((10, 10))
x[1: -1, 1: -1] = 0
np.pad(x, constant_values = 0, pad_width = 1, mode = 'constant') # add a border or pad

# np.full(shape, fill_value) returns new array of a specified shape, fills with specified fill_value
# 2x3 array with all values 5
fours = np.full((3, 3), 4)

# resize(arr, new_shape) returns new array with specified shape, if new array is larger than original array, then new
# array is filled with repeated copies of arr
fours_resize = np.resize(fours, (3, 2))

# inverse of a matrix
matrix = np.array([[1, 2], [3, 4]])
np.linalg.inv(matrix)

# diagonal matrix
diag = np.diag(np.arange(1, 6))
np.diag(1 + np.arange(4), k = -1)
np.diag(np.arange(0, 6), k = -2)

# trace is the sum of all the diagonal elements of a square matrix
np.trace(fours)

# determinant of a matrix is calculated from a square matrix
np.linalg.det(matrix)

# rank of a matrix is the estimate of the number of linearly independent rows or columns in a matrix
np.linalg.matrix_rank(fours_resize)

# Find Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(diag)
eigenvalues
eigenvectors

'''Difference between arange vs linspace is 3rd arguement, which is step size vs no of elements.
linspace gives no of points i.e., no of linearly spaced points between specified lower n upper bounds
Range/arange gives step size i.e., evenly spaced nos until upper bound'''
np.linspace(2, 8, 12).reshape(3, 4) # linearly spaced elements between specified lower n upper bounds

# logspace returns nos spaced evenly on a logscale (base default 10) after raising each no by specificed base
np.logspace(start = 2, stop = 3, num = 5, base = 10)

'''
rand is for uniform distribution between 0 to 1
randn is for normal distribution centered around 0
randint is for random integers between specified low & high
'''

np.random.rand()
np.random.rand(6)
np.random.rand(6).reshape(3, 2) # output has points between 0 & 1 in uniform distribution
np.random.rand(3, 5)
np.random.rand(3, 5, 4) # output gives 3D matrix centered around 0

np.random.randn()
np.random.randn(6)
np.random.randn(6).reshape(3, 2) # output is centered around normal distribution
np.random.randn(4, 4)
np.random.randn(4, 4, 5) # output is 3D matrix centered around 0

np.random.randint() # gives error
np.random.randint(100)
np.random.randint(10, 100)
np.random.randint(1, 100, 9).reshape(3, 3)
np.random.randint(1, 100, 9).reshape(3, 3).max() # or .min()

np.random.randn(6).argmax()
np.random.randn(6).argmin()

np.argmax(my_matrix)
np.argmin(my_matrix)

np.random.random((3, 3 ,3))
x = np.random.random((10, 10))
x.argmax(), x.argmin()
x.max(), x.min()
np.random.random(30).mean()
np.random.random(3)

# permutation() randomly permutes a iterable or a range
np.random.permutation(45)
np.random.permutation([1, 4, 9, 12, 15])
x = np.arange(9).reshape((3, 3))
np.random.permutation(x)

# calculating time taken for operations by element wise addition & by list comprehension
x = np.random.randint(1, 650, 1000000).reshape(-1, 2)
x = pd.DataFrame(x)
x.columns = ['a', 'b']

# element wise addition
start = time.time()
x['c'] = x['a'] * x['b']
end = time.time()
ops = end - start
print(round(ops, 3))

# list comprehension
start = time.time()
x['c'] = [a * b for a, b in zip(x['a'], x['b'])]
end = time.time()
ops = end - start
print(round(ops, 3))

x = np.arange(10)
y = np.arange(11, 20)
np.savez('temp_arra.npz', x = x, y = y)
with np.load('temp_arra.npz') as data: # load arrays from the temp_arra.npz file
    x1 = data['x']
    y1 = data['y']
    print(x1, y1)

np.random.randn(4).shape
arr = np.random.rand(3)
arr.dtype
lister = np.array([543, 45.62, 'fast'])
type(lister)
lister.dtype

array = np.arange(3, 13)
array[::] # or array or array[:]
array[4]
array[3:]
array[:5]
array[2:4] = 71
array
slice = array[2:4]
slice
slice[:] = slice - 68
array[:]

matrix1 = np.array([64,83,89,23,94,56,41,73,36,79,14,68]) # 1D array is called vector
matrix2 = np.array([[64,83,89],[23,94,56],[41,73,36],[79,14,68]]) # 2D array is called matrix
matrix3 = np.arange(16).reshape(4, 4)
matrix1.reshape(3, 4)
matrix2.reshape(4, 3)
matrix1[3]
matrix2[1][1] # or matrix[1, 1]
# grab 94,56-73,36 slice
matrix2[1:3, 1:]
# grab 23,94-41,73-79,14
matrix2[1:, 0:2]
# grab 83,89-94,56
matrix2[0:2, 1:]

array1 = np.arange(1, 11)
bool_slice = array1 > 4
bool_slice
array1[bool_slice]
# or
array1[array1 > 4]
array1 * 3
array1 * array1
array1 / 0
1 / array1
array1 ** 2
# Universal functions of numpy can be used for operations like sqrt, exp, log, max, min etc.
np.sqrt(array1)
np.exp(array1)
np.max(array1)

# coerce an array having even objects
x = np.array(['1', '2', 'a'])
df_x = pd.to_numeric(x, errors = 'coerce')

# Numerical operations on array & list
ms = [323,743,654,224,737]
a = np.array(ms)
ms * 2 # doubles the same set of elements by the multiplier
a * 2 # performs element wise operation for an array

'''
a.sort()
Sorts the array in-place
Return type is None
Occupies less space. No copy created as it directly sorts the original array

sorted(a)
Creates a new list from the old & returns the new one, sorted
Return type is list
Occupies more space as copy of original array is created and then sorting is done
Slower than a.sort()

np.argsort(a)
Returns the indices that would sort an array
Return type is numpy array

np.argmax(), np.argmin(), np.argsort() these return indices respectively for those functions

'''
ms = [323,743,654,224,737]
sorted(ms)

ms = [323,743,654,224,737]
ms.sort()

ms = np.array([323,743,654,224,737])
in_arr = np.argsort(ms)
ms[in_arr]

sorted(ms, key = lambda x: x[1]) # Or
ms.sort(key = lambda x: x[1])

'''
Newaxis increases dimension of existing array by one more dimension and also to explicitly convert a 1D array
to either a row vector or a column vector.
It can be used more than once to increase dimensions of array (Ex: higher order arrays i.e. Tensors)
1D array becomes 2D array
2D array becomes 3D array

Np.newaxis vs np.reshape
Np.newaxis uses slicing operator to recreate array, while np.reshape reshapes array to desired layout
'''
# 1D array
arr = np.arange(4)
# arr = np.arange(4).reshape(2, 2) # check with 2D array
arr.shape
# make it as row vector by inserting an axis along first dimension
arr[1, 0] # accessing a single value
arr[1:, :1] # accessing a slice of multi dimentional array
row_vec = arr[np.newaxis, :]
row_vec.shape
# make it as column vector by inserting an axis along second dimension
col_vec = arr[:, np.newaxis]
col_vec.shape

# using for numpy broadcasting
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([2, 5, 6])
x1 = x1[:, np.newaxis] # use np.newaxis to increase the dimension of one of the arrays so that NumPy can broadcast
x1 +  x2
x1 = x1[np.newaxis, :] # gives error bcoz x1 and x2 are both along the same axis
x1 +  x2
x2 = x2[:, np.newaxis] # use np.newaxis to increase the dimension of one of the arrays so that NumPy can broadcast
x1 + x2
x2 = x2[np.newaxis, :] # gives error bcoz x1 and x2 are both along the same axis
x1 + x2

x1 = np.arange(1,10).reshape(3,3)
print(x1)
x1_new = x1[:,np.newaxis]
print(x1_new)

a = np.ones((3,4,5,6))
b = np.ones((4,6))
(a + b[:, np.newaxis, :]).shape

'''Exercises'''

np.zeros(10)
np.ones(10)
np.ones(10) * 5
np.arange(10, 51)
arrays = np.arange(10, 51)
arrays[arrays % 2 == 0] # or np.arange(10, 51, 2)
np.arange(9).reshape(3, 3)
np.eye(3)

np.random.rand(1)
np.random.randn(25)
np.arange(0.01, 1.01, 0.01)
np.linspace(0.01, 1, 100).reshape(10, 10)
np.linspace(2, 8, 12).reshape(3, 4)
np.linspace(0, 1, 20)

mat = np.arange(1, 26).reshape(5, 5)
mat[2:, 1:]
mat[3, 4]
mat[0:3, 1:2]
mat[4, :]
mat[3:, ]
np.sum(mat) # mat.sum()
np.std(mat)
mat.sum(axis = 1)
mat.sum(axis = 0)

my_list = [1, 2, 3]
my_list
array = np.array(my_list)
array

my_list2 = [[5,7,3,7], [2,8,4,2]]
my_list2
array_mat = np.array(my_list2)
array_mat.reshape(4, 2)

np.arange(2, 10)
np.arange(10)
np.arange(2, 9, 3)
np.linspace(2, 4, 5)

np.zeros((3, 3))
np.ones((3, 3))
np.eye(3, 4)

np.random.rand(4) # gives uniformly distributed nos between 0 to 1
np.random.rand(4, 4) # gives 2D matrix
np.random.randn(2) # gives normally distributed nos of the input from 0 to 1
np.random.randn(4, 3)
np.random.randint(2, 10)
my_list3 = np.random.randint(2, 7, 6)
my_list3
my_list3.reshape(2, 3)
my_list3.max()
my_list3.min()
my_list3.argmax() # index of the max element
my_list3.argmin() # index of the min element
my_list3.shape
my_list3.dtype

array1 = np.random.randint(1, 100, 10)
array2 = np.arange(11, 21)
array2[3]
array2[1:5]
array2[2:]
array2[:5]
array_slice = array2[2:5]
array_slice[:3] = 9
array2

my_array = np.arange(12, 31, 2)
my_array
slice_array = my_array[2:6]
slice_array
slice_array_copy = slice_array.copy()
slice_array_copy = 34
slice_array_copy


'''''''''''''''''''''''''''Pandas'''''''''''''''''''''''''''

pd.show_versions(as_json = True)
'''Practice:
https://github.com/danielmercy/Python_for_Data_Science_and_Machine_Learning-Bootcamp/tree/master
/04-Pandas-Exercises
https://www.machinelearningplus.com/python/101-pandas-exercises-python/
'''
# The single bracket outputs a Pandas Series, while a double bracket outputs a Pandas DataFrame.
labels = ['a','b','c','d','e']
my_dict = {5 : 'five', 2: 'two', 9: 'nine', 1: 'one', 4: 'four'}
num_labels = [4, 5, 6, 7, 8]
array = np.array(num_labels)

pd.Series(labels)
pd.Series(my_dict)
pd.Series(num_labels)
pd.Series(data = array, index = labels)
pd.Series(data = my_dict, index = labels) # gives nan, since k-v of dict does not match with those in labels
pd.Series(data = my_dict, index = num_labels)
pd.Series(array)
pd.Series(index = labels, data = array)
pd.Series(data = [num_labels] * len(num_labels), index = labels)
pd.Series(np.random.rand(6))
ser1 = pd.Series(data = [1,2,3], index = ['India','Germany','France'])
ser2 = pd.Series(data = [5,6,7], index = ['Japan','India','France'])
ser1['India']
ser1 * ser2
serx = pd.Series([1, 2, 3, 4, 5])
sery = pd.Series([4, 5, 6, 7, 8])
# get the items of series A not present in series B
serx[~serx.isin(sery)]
# get the items of series A present in series B
serx[serx.isin(sery)]
# get the items not common to both series A and series B
seru = pd.Series(np.union1d(ser1, ser2))
seri = pd.Series(np.intersect1d(ser1, ser2))
seru[~seru.isin(seri)]

# convert the first character of each element in a series to uppercase
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
for i in ser:
    print(i[0].upper() + i[1:]) # or
pd.Series([i[0].upper() + i[1:] for i in ser]) # or
ser.map(lambda x: x[0].upper() + x[1:])
ser.map(lambda x: x.title())
pd.Series([i.title() for i in ser])

# creating a dataframe quickly
# method 1
df = pd.DataFrame(columns=['Item', 'Qty1', 'Qty2'])
for i in range(5):
    df.loc[i] = ['name_' + str(i)] + list(randint(1, 9, 2))
df

# method 2
dataset = pd.DataFrame(data = randn(5, 5), index = ['ABC','DEF','GHI','JKL','MNO'], columns =['India','US','Nepal','Bhutan','Japan'])
dataset['US']

# method 3
fruits = pd.Series(index = np.random.choice(['apple', 'banana', 'carrot'], 10), data = np.linspace(1, 10, 10))
fruits.columns = ['Qt']
fruits.index.names = ['fruits']

# method 4
df1 = pd.DataFrame({'keys': ['b', 'e', 'a', 'b', 'i', 'a', 'u'], 'data1': range(7), 'data2': range(21, 28)})

# creating classification model quickly
from sklearn.datasets import load_sample_images, load_sample_image
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=1000, n_features=6, n_informative=2, n_clusters_per_class=1, random_state=4)
x = pd.DataFrame(data = x, columns = ['A', 'B', 'C', 'D', 'E', 'F']).round(2)
y = pd.DataFrame(data = y, columns = ['output']).round(2)
df = pd.concat([x, y], axis = 1)
df.sample(6)

'''accesing values in a data frame'''

df = pd.DataFrame(data = np.random.randn(5, 4).round(2), index = ['a','b','c','d','e'], columns = ['w','x','y','z'])

# 2 ways to access columns
df['w'] # 1st way
df.w # 2nd way
type(df['w'])
type(df)
df[['x','z']] # accessing multiple columns

# 2 ways to access rows
df.loc['a'] # based on row name
df.loc['b','y'] # format is in row, column
df.loc[['a', 'b'], ['w', 'z']]
df.iloc[1] # based on row index

# accessing a value in a specific row/s n column/s or accessing a subset of dataframe
df.loc['a':'c','w':'y']
df[['x', 'z', 'w']].loc[['a', 'c', 'd']] # either cols n rows
df.loc[['a', 'c', 'd'],['x', 'z', 'w']] # or rows n cols as loc[['a', 'c', 'd']][['x', 'z', 'w']]
df[-1:]

# Conditional selection. Always use: & for And and | for Or
df[df['w'] > 0][['x', 'y', 'z']] # selecting the columns we need after the condition
df[(df['w'] > 0) & (df['z'] < 0)][['x', 'y', 'z']] # passing multiple conditions
df[(df['w'] > 0) | (df['z'] < 0)][['x', 'y', 'z']] # passing multiple conditions

# difference between .loc and without .loc
# Conditional updation of values in specific rows n columns
df[df['w'] > 0] = 5 # changes values in all columns which meet specified condition of the column
df.loc[df['w'] > 0, 'z'] = 15 # changes values only in a specified column which meets condition of the column
df.loc[df['w'] > 0, ['z', 'w']] = 15 # changes values only in specified columns which meets condition of the column

df['z+'] = df['x'] + df['z']
df.drop('z+', axis = 1) # inplace = True makes the change permanent
df.drop('e', axis = 0)
df

# reset and set index
df.reset_index() # includes an existing index as a column in data frame
new_col = 'CA ID NY WY NJ'.split()
df['states'] = new_col
df.set_index('states')
df.index.names = ['s.no']
df.keys() # same as df.columns
# syntax to drop rows that have a specific value in a column
df.drop(df[df[col_name] == value].index, inplace = True)

''''''
# iat and iloc accept row and column numbers, at and loc accept index and column names
cars.at[row[0], 'Price']
cars.loc[row[0], 'Price']
cars.iat[row[0], col[0]]
cars.iloc[row[0], col[0]]

'''Let there be 2 columns. Values in 2nd column would be 0 if those in 1st are odd, and the same as 1st
if even. Hint: use slicing/pandas & conditional selection'''
# method 1
dfs = pd.DataFrame(np.arange(10, 19), columns = ['x']) # , index = list('qwertyuio'), dtype = 'object'
dfs['y'] = np.where(dfs['x'] % 2 == 0, dfs['x'], 0)
# method 2
dfs = pd.DataFrame(np.arange(10, 19), columns = ['x'])
dfs['y'] = dfs.copy()
dfs.loc[dfs['x'] % 2 != 0, 'y'] = 0
dfs

'''Rename is for df column renaming, replace & map are for renaming labels of specific columns'''
# renaming rows/columns
df.rename(columns = {old_name: new_name}, inplace = True)

'''replace an iterable or a single element in a column. Can be done in 3 ways'''
# using map(), maps certain values to new attributes/values
dicts = {val1 : new_val1, val2 : new_val2}
df[col_name] = dataset[col_name].map(dicts).astype(int)
# using replace(). Replace can be used with or without dict with iterable/old value vs new but map() needs dict
df[col_name].replace(dicts) # Or below syntax
# df[col_name].replace(iterable/single element, new_value, inplace = False)
# using apply() function with lambda. given below is rough syntax
df.apply(lambda x: x for x in df[x])

# Missing/null/Nan data - isna(), isnull(), dropna(), fillna()
'''
MEAN: Suitable for continuous data without outliers
MEDIAN : Suitable for continuous data with outliers
Mode: Suitable for categorical feature
'''
new_data = pd.DataFrame(np.random.rand(6, 5).round(2), columns = ['a','b','c','d','e'])
new_data.loc[1:3, 0:2] = np.nan
new_data.loc[5][2] = np.nan
new_data.isna().values.any() # boolean for entire dataset
new_data['c'].isnull().values.any() # boolean for column
new_data.loc[5].isnull().values.any() # boolean for row
new_data.loc[5].isna().sum() # gives count of all null values in a row
new_data['c'].isna().sum() # gives count of all null values in a column
new_data.isnull().sum().sort_values(ascending = False) # same as above gives count of all null values in a column
new_data.isna().sum().idxmax() # shows col which has most null values
new_data.isna().sum().idxmin() # shows col which has least null values
new_data.isna().sum() # gives count of null values in each of the column/column wise
new_data.isnull().sum() # or emer.isnull().sum()
new_data.isna().sum().sum() # gives count of all null values across all the columns

# imputing missing vals in multiple cols
cars = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Cars93_miss.csv')
# probing for missing values
cars.isna().sum().sort_values(ascending = False)
# finding & filtering out columns that have values less than 30%
cars = cars[[cols for cols in cars if cars[cols].count() / len(cars.index) >= .3]]
# imputing multiple cols using apply and single method
cars[['Horsepower','Length']] = cars[['Horsepower','Length']].apply(lambda col: col.fillna(value = col.mean())) # OR
# imputing multiple cols using fillna and value parameter
cars[['Horsepower','Length']] = cars[['Horsepower','Length']].fillna(value = cars['Horsepower'].mean())
# imputing by mode method
cars['Length'] = cars['Length'].fillna(cars['Length'].value_counts().index[0]) # individually like this, OR
cars[['Horsepower','Length']] = cars[['Horsepower','Length']].fillna(value = cars['Horsepower'].mode()[0])
# imputing multiple cols using multiple methods
methods = {'Min.Price': np.nanmean, 'Max.Price': np.nanmedian}
cars[['Min.Price', 'Max.Price']].apply(lambda x, methods: x.fillna(methods[x.name](x)), args = (methods,))
# accessing values in a column
cars['Manufacturer'].unique()
cars[cars['Manufacturer'] == 'Audi'] # accessing single row value
cars[cars['Manufacturer'].isin(['Audi', 'BMW', 'Mercedes-Benz'])] # accessing multiple row values of column

'''
If there are illegible characters like ? or - in tables, they can be detected and replaced with
np.nan which can be further replaced using replace() and/or a function/lambda using apply() method
'''
# gives count of each of the values in a specific column
df[col_name].value_counts()
df[col_name].value_counts(normalize = True) # to calculate fractions

d = {'a':[2,6,np.nan], 'b':[20,np.nan,24], 'c':[62,np.nan,66]}
df = pd.DataFrame(d)
df.dropna(axis = 0)
df['a'].dropna(axis = 0)
df.dropna(thresh = 2) # min no of non-nan that are permitted in dataframe, no of NaN values beyond which gets dropped
df['b'].fillna(value = df['b'].mean())
df['c'].fillna(value = 'N/A', inplace = True)
df['a'].fillna(value = '$', inplace = True) # use any value to impute, with integer or string

new_data1 = pd.DataFrame(np.random.rand(10, 5), columns = ['a','b','c','d','e'], index = np.arange(10, 20))
new_data1.iloc[1:3, 0:2] = np.nan # index based, or
new_data1.loc[11:12, ['a', 'b']] = np.nan # label based
new_data1['b'].isna().sum()
new_data1.isna().sum().sum()

dicts = {'a':[6, 7, np.nan],'b':[3, np.nan, np.nan],'c':[7, 2, 5]}
df = pd.DataFrame(dicts)
df
df.dropna() # inplace = True for permanent replace
df.dropna(thresh = 2, axis = 1)
df.fillna(value = 'xx')
df.columns # or df.keys()
df.fillna(value = df.mean()) # imputing missing values by mean
df[::].fillna(value = df[::].mean()) # OR same as above

'''Group by syntax: df.groupby(by = grouping_columns)[columns_to_show].function()'''

data = {'company':['GOOG', 'MSFT', 'FCBK', 'MSFT', 'GOOG', 'FCBK'],
        'person':['sam', 'manq', 'chary', 'carl', 'sara', 'dora'],
        'sales': [654, 567, 345, 876, 246, 765], 'cap': [3543, 5654, 7654, 8764, 2463, 9874]}
df = pd.DataFrame(data)
df
df.groupby('company').count()
df.groupby('company').sum()
df.groupby(['company', 'sales']).sum()
df.groupby('company').sum().loc['GOOG']
df.groupby(['company','sales']).mean()
df[['company', 'sales']].groupby(['company']).mean()
df.groupby('company').describe().transpose()['MSFT']
df.quantile()

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}, index = [1, 2, 3, 4])
df1
# Creating second dataframe
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'], 'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'], 'D': ['D4', 'D5', 'D6', 'D7']}, index = [5, 6, 7, 8])
df2
# Creating third dataframe
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'], 'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'], 'D': ['D8', 'D9', 'D10', 'D11']}, index = [9, 10, 11, 12])
df3

'''
Concat is for row wise unification. Attributes are objs, join, axis. Objs/cols passed as list
Merge is for column wise unification. Attributes are left, right and how, where join types can be specified
'''
new_df = pd.concat([df1, df2, df3], axis = 0)
new_df
new_df = pd.concat([df1, df2, df3], axis = 1)
new_df
'''Merging is for column wise unification'''
left = pd.DataFrame({'Key': ['K0', 'K1', 'K2', 'K3'], 'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'Key': ['K0', 'K1', 'K2', 'K3'], 'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

pd.merge(left, right, how = 'inner', on = 'Key')


df1 = pd.DataFrame({'keys': ['b', 'e', 'a', 'b', 'i', 'a', 'u'], 'data1': range(7), 'data2': range(21, 28)})
df1['keys'].unique() # array of unique values same as set(df1['keys'])
df1['keys'].nunique() # no of unique values
df1['keys'].value_counts()
df1['keys'].value_counts().index # returns list of elements in index
df1[(df1['data2'] > 24) | (df1['data1'] > 5)]
df1[(df1['data2'] > 24) & (df1['data1'] > 5)]

fruits = ['apple', 'banana', 'cherry']
x = fruits.index('cherry')

'''Exercises'''
# https://pynative.com/python-pandas-exercise
url = 'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Automobile_data.csv'
autos = pd.read_csv(url)
autos.columns
autos.head()
autos.tail()
autos.info()
autos[['company', 'price']][autos['price'] == autos['price'].max()] # or
autos[['company', 'price']][autos['price'] == np.max(autos['price'])]
autos[autos['company'] == 'toyota']
autos['company'].groupby(autos['company']).count()
autos[['company', 'price']].groupby(autos['company']).max()
autos[['company', 'price']].groupby(autos['company']).max().sort_values(ascending = False, by = 'price')
autos[['company', 'price']].groupby(autos['company']).max().sort_values(ascending = False, by = 'price').head(1)
autos[['company', 'average-mileage']].groupby(autos['company']).mean()
autos.sort_values(by = 'price', ascending = False)
autos.sort_values(by = ['price', 'company'], ascending = [False, True]).head() # sort by multiple cols
# Find brands that start with 'M'
autos['company'].unique()
autos['company'].str.startswith('m').value_counts() # to find specific values in a column using str.startwith
m_company = autos[autos['company'].str.startswith('m')] # df of companies starting with 'm'

german_cars = {'Company': ['Ford', 'Mercedes', 'BMV', 'Audi'], 'Price': [23845, 171995, 135925, 71400]}
japanese_cars = {'Company': ['Toyota', 'Honda', 'Nissan', 'Mitsubishi '], 'Price': [29995, 23600, 61500, 58900]}
germ = pd.DataFrame.from_dict(german_cars)
jap = pd.DataFrame.from_dict(japanese_cars)
tab = pd.concat([germ, jap], axis = 0, keys = ['German', 'Japs']) # keys is names of columns
tab.index.names = ['nation', 'sno']

# Hierarchical & multi indexing, XS cross section
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
dfs = pd.DataFrame(data = np.random.randn(6, 2), index = hier_index, columns = ['A', 'B'])
dfs.loc['G2'].loc[2]['A']
dfs.index.names = ['Group', 'Floor'] # giving names to indexes
dfs
dfs.xs(2, level = 'Floor') # accessing sub index values from multi level index

# Hierarchical dataframe
car_price = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'Price': [23845, 17995, 135925, 71400]}
car_horsepower = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'horsepower': [141, 80, 182, 160]}
car = pd.DataFrame.from_dict(car_price)
hp = pd.DataFrame.from_dict(car_horsepower)
pd.merge(car, hp, how = 'inner', on = 'Company')

# N largest values in a given column
df = pd.DataFrame({'population': [59000000, 65000000, 434000,434000, 434000, 337000, 11300,11300, 11300],
                   'GDP': [1937894, 2583560 , 12011, 4520, 12128,17036, 182, 38, 311],
                   'alpha-2': ["IT", "FR", "MT", "MV", "BN","IS", "NR", "TV", "AI"]},
                  index=["Italy", "France", "Malta","Maldives", "Brunei", "Iceland","Nauru", "Tuvalu", "Anguilla"])
df['GDP'].sort_values(ascending = False).head(3)
df.nlargest(3, 'population')
df.nlargest(3, 'population')[['GDP', 'alpha-2']]
df.nlargest(3, 'population').index

cor_matrix.nlargest(5, 'SalePrice')['SalePrice'] # replace cor_matrix wit a dataframe as needed
cor_matrix.nlargest(5, 'SalePrice')[['MSSubClass', 'SalePrice','GarageCars']] # gives top 5 values in SalePrice
# filtering out others n keeping SalePrice w.r.t features
cor_matrix.nlargest(5, 'SalePrice')[['MSSubClass', 'SalePrice','GarageCars']].index

'''
From - D:\Programming Tutorials\Machine Learning\Data Science & ML - Jose Portilla 1\
Python-Data-Science-and-Machine-Learning-Bootcamp\Data-Capstone-Projects
'''
emer = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\911.csv')
emer.sample(4)
emer.columns
# method 1: renaming columns
emer.columns = ['latitude', 'longitude', 'description', 'zip code', 'salutation', 'time', 'TWP', 'address', 'E']
# method 2: renaming columns
old = ['lat', 'lng', 'desc', 'zip', 'title', 'timeStamp', 'twp', 'addr', 'e']
new = ['latitude','longitude','description','zip code','salutation','time','TWP','address', 'E']
cols_dict = dict(zip(old, new))
emer.rename(columns = cols_dict, inplace = True)

emer.info()
emer.select_dtypes(include = ['float64', 'int64']) # selecting columns of only specific data types
stats = emer.describe()
emer['E'].unique()
emer['E'].astype('int64')
emer.head(5)
emer.sample(6)
emer.shape
emer.isna().sum()
emer['zip code'].value_counts().head(5)
emer['TWP'].value_counts().head(5)
emer['salutation'].nunique()
type(emer['time'].iloc[0])
emer['Reason'] = emer['salutation'].apply(lambda salute: salute.split(':')[0])
emer['Reason'].value_counts().head()

'''Date time'''
# single entry
time = emer['time'].iloc[0] # just 1 time value
tym = pd.to_datetime(time)
tym.date()
tym.time()
[tym.day, tym.month, tym.year, tym.hour, tym.minute, tym.second]
# entire series
time = emer['time']
tym = pd.to_datetime(time)
emer['tym'] = tym
emer['Hour'] = emer['tym'].apply(lambda tym: tym.hour)
emer['Month'] = emer['tym'].apply(lambda tym: tym.month)
emer['Day of Week'] = emer['tym'].apply(lambda tym: tym.dayofweek)
dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
emer['Day of Week'] = emer['Day of Week'].map(dmap)

by_month = emer.groupby(emer['Month']).count().head()
emer['Date'] = emer['tym'].apply(lambda tym: tym.date())

emer[['Hour', 'Day of Week']]
hour_day = pd.DataFrame(data = , columns = emer['Hour'], index = emer['Day of Week'])

# get the day of month, week number, day of year and day of week from a series of date strings
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
time = pd.to_datetime(ser)
dics = {'dates': [], 'week_of_year': [], 'day_of_year': [], 'weekday_name': []}
for i in time:
    dics['dates'].append(i.day)
    dics['week_of_year'].append(i.weekofyear)
    dics['day_of_year'].append(i.dayofyear)
    dics['weekday_name'].append(i.weekday_name)
print('Date: ', dics['dates'],'\nWeek number:', dics['week_of_year'],
      '\nDay num of year:', dics['day_of_year'],'\nDay of week:', dics['weekday_name'])

import datetime
time = datetime.time(16, 45, 54)
print(time)
time.hour
time.minute
today = datetime.date.today()
today.timetuple()

import calendar
for i, day in enumerate(calendar.day_name):
    print(i, day)

# CSV, Excel, HTML, SQL
df1.to_csv('outputs.csv', index = False)
df1 = pd.read_csv('outputs.csv')
df1
df1.drop('Unnamed: 0', axis = 1, inplace = True)

html_df = pd.read_html('https://www.fdic.gov/bank/individual/failed/banklist.html') # or
banks = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\banklist.csv')
type(html_df)
type(banks)
html_df[0].head(3)

'''Transformation of raw data to pandas, numpy '''

dataset_raw = {"A": ['r', 's'], "B": ['f', 't']}
type(dataset_raw)

dataset_np = pd.DataFrame(dataset_raw).to_numpy()
type(dataset_np)

dataset_pd = pd.DataFrame(dataset_raw)
type(dataset_pd)

dataset_np_pd = pd.DataFrame(dataset_np)
type(dataset_np_pd)

dataset_pd_np = pd.DataFrame(dataset_pd).to_numpy()
type(dataset_pd_np)


'''''''''''''''''''''''''''Visualization'''''''''''''''''''''''''''

'''''''''Matplotlib'''''''''
# barplot, boxplot, pieplot, histplot, scatterplot, stackplot

https://www.kaggle.com/sujan97/data-visualization-quick-guide To do

https://www.analyticsvidhya.com/blog/2020/02/beginner-guide-matplotlib-data-visualization-exploration-python/
https://www.youtube.com/watch?v=Fkdd9cQd3bo
https://www.youtube.com/watch?v=_YWwU-gJI5U

x = np.linspace(1, 50, 15)
y = x ** 4
a = np.linspace(1,30,15)
b = np.sin(a)

### line plot & bar plot. replace line with bar to see bar plot else line plot
# functional method
plt.plot(x, y)
plt.xlabel('X Labels')
plt.ylabel('Y Labels')
plt.title('Titles')
plt.show()

plt.subplot(1, 2, 1)
plt.plot(x, y, 'r')
plt.subplot(1, 2, 2)
plt.plot(y, x, 'b')

# Object Oriented method
fig = plt.figure()
axes = fig.add_axes([.1,.1,.8,.8]) # left, bottom, height, width
axes.plot(x, y)
axes.set_xlabel('X Labels')
axes.set_ylabel('Y Labels')
axes.set_title('Titles')

fig = plt.figure(figsize = (6, 4))
axes = fig.add_axes([.1,.1,.8,.8])
axes.bar(x, y)
axes.set_xlabel('X Labels')
axes.set_ylabel('Y Labels')
axes.set_title('Titles')

# loc positions: 0 is best, 1 is top right, 2 is top left, 3 is bottom left, 4 is bottom right, 10 is center
fig = plt.figure()
axes = fig.add_axes([.1,.2,.7,.6])
axes.plot(x, y, label = 'X vs Y')
axes.plot(a, b, label = 'A vs B')
axes.set_xlabel('X Axis')
axes.set_ylabel('Y Axis')
axes.set_title('X vs Y & A vs B')
axes.legend(loc = 4) # specify tuple with x & y points for custom location of legend

### plot within plot on same canvas
fig = plt.figure()
axes1 = fig.add_axes([.1,.2,.7,.6])
axes2 = fig.add_axes([.2,.4,.3,.3]) # % or ratio w.r.t left, bottom, width, height of above axes
axes1.plot(x, y, label = 'x vs y')
axes2.plot(b, a, label = 'b vs a')
axes1.set_xlabel('X Axis')
axes1.set_ylabel('Y Axis')
axes1.set_title('X-Y Axes')
axes2.set_xlabel('B Axis')
axes2.set_ylabel('A Axis')
axes2.set_title('B-A Axes')
axes1.legend(loc = 4)
axes2.legend(loc = 4)

fig = plt.figure(figsize = (8, 8)) # with figsize
axes1 = fig.add_axes([.1,.2,.7,.6])
axes2 = fig.add_axes([.2,.45,.3,.3])
axes1.plot(x, y, label = 'X vs Y')
axes2.plot(b, a, label = 'B vs A')
axes1.set_xlabel('X Axis')
axes1.set_ylabel('Y Axis')
axes1.set_title('X Y Axes')
axes2.set_xlabel('B Axis')
axes2.set_ylabel('A Axis')
axes2.set_title('B A Axes')
axes1.legend(loc = 4)
axes2.legend(loc = 4)

### plot on sub plots on different canvas. fig.add_axes() doesnt apply for sub plots
fig, axes = plt.subplots(nrows = 1, ncols = 2)
fig
axes

fig, axes = plt.subplots(nrows = 1, ncols = 2)
plt.tight_layout()
for axis in axes:
    axis.plot(x, y)

fig, axes = plt.subplots(nrows = 1, ncols = 2)
axes[0].plot(x, y)
axes[1].plot(y, x)
axes[0].set_title('Axes 0 Title')
axes[0].set_xlabel('axes 0 X')
axes[0].set_ylabel('axes 0 Y')
axes[1].set_title('Axes 1 Title')
axes[1].set_xlabel('axes 1 X')
axes[1].set_ylabel('axes 1 Y')
plt.tight_layout()

fig, ax = plt.subplots(figsize = (6, 3), nrows = 1, ncols = 2)
ax[0].plot(x, y)
ax[1].plot(y, x)
plt.tight_layout()
# fig.savefig('fine.png')

# Figure size & DPI
fig = plt.figure(figsize = (5, 3))
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y)

fig, ax = plt.subplots(figsize = (3, 7), nrows = 2, ncols = 1)
ax[0].plot(x, y, label = 'X vs Y')
ax[1].plot(b, a, label = 'B vs A')
ax[0].set_title('Axes 0 Title')
ax[0].set_xlabel('axes 0 X')
ax[0].set_ylabel('axes 0 Y')
ax[1].set_title('Axes 1 Title')
ax[1].set_xlabel('axes 1 X')
ax[1].set_ylabel('axes 1 Y')
plt.tight_layout()
ax[0].legend(loc = 4)
ax[1].legend(loc = 4)
fig.savefig('my_pic.jpeg', dpi = 200)

# Legends
fig = plt.figure(figsize = (4, 3))
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, x ** 2, label = 'X square')
ax.plot(x, x ** 3, label = 'X cube')
ax.plot(x ** 2, x, label = 'X square reverse')
ax.legend(loc = 0)

# Colors
fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
ax.plot(x, y, color = 'violet', lw = 2, alpha = 0.7, ls = '-.', marker = 'o', markersize = 4, markerfacecolor = 'y',
        markeredgewidth = 4, markeredgecolor = 'g') # alpha is for transparency

fig = plt.figure()
ax = fig.add_axes([1, 1, 1, 1])
ax.plot(x, y, color = 'violet', lw = 2, ls = '-.')
ax.set_xlim([0, 1]) # show plot only between upper n lower bounds on a unit of X axis
ax.set_ylim([0, 2]) # show plot only between upper n lower bounds on a unit of Y axis

### Histograms are like barplot but groups data within boundaries aka bins instead of plotting individual data points
ages = [11, 15, 53, 35, 73, 75, 45, 21, 50, 75, 25, 99, 86, 13, 34, 88, 63]
plt.hist(ages, bins = 5, edgecolor = 'r')
# or
bins = [10, 20, 30, 40, 80, 100]
plt.hist(ages, bins = bins, edgecolor = 'r', log = True) # use log = True to see values too small to show on hist plot
# or in OO method
fig = plt.figure(figsize = (5, 5))
ax = fig.add_axes([1, 1, 1, 1])
ax.hist(ages, bins = 5, color = 'r', lw = 2, alpha = 0.7)

### Scatter plots show relationship of 2 features/sets of values to find how they are correlated
ages = [11,68, 15, 53, 35, 73, 75, 45, 21, 50, 75, 25, 99, 86, 13, 34, 88, 63]
scores = [50,75,25,100,86,13,34,88,51,10,11, 15, 53, 35, 73, 75, 45, 21]
plt.scatter(ages, scores, s = 50, c = 'r', marker = 'o', edgecolor = 'b', lw = 2, alpha = .75) # s = size, c = color

url = r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\views_data.csv'
data = pd.read_csv(url)
views = data['view_count']
likes = data['likes']
ratio = data['ratio']
plt.scatter(views, likes, s = 50, c = 'r', marker = 'o', edgecolor = 'b', lw = 2, alpha = .75)
# Or improvement over above
plt.scatter(views, likes, s = 50, c = ratio, cmap = 'Greens', marker = 'o', edgecolor = 'b', lw = 2, alpha = .75)
cbar = plt.colorbar()
cbar.set_label('Like vs Dislike ratio')
plt.xscale('log') # log scaling of data on x axis
plt.yscale('log') # log scaling of data on y axis
# or in OO method
fig = plt.figure(figsize = (5, 5))
ax = fig.add_axes([1, 1, 1, 1])
ax.scatter(views, likes, s = 50, c = ratio, cmap = 'Greens', marker = 'o', edgecolor = 'b', lw = 2, alpha = .75)
ax.set_xscale('log') # log scaling of data on x axis
ax.set_yscale('log') # log scaling of data on y axis

### Project
# Visualization of Trigonometric functions wen applied to numbers from 1 to 10
a = np.arange(1,10)
b = np.sin(a)
c = np.cos(a)
d = np.tan(a)
g = np.arctan(a)

fig = plt.figure(figsize = (12,12))
ax = fig.add_axes([.2,.2,.6,.6])
ax.plot(a, b, label = 'Sin function')
ax.plot(a, c, label = 'Cos function')
ax.plot(a, d, label = 'Tan function')
ax.plot(a, g, label = 'Co-tangent function')
ax.set_title('Visualization of Sin, Cos, Tan & Co-tangent functions of integers in the range of 1 to 10')
ax.set_xlabel('A')
ax.set_ylabel('B C D')
ax.legend(loc = 0)

'''''''''Seaborn''''''''' # for statistical plotting library on top of matplotlib

# =============================================================================
# :::::Seaborn plot styles:::::
#
# Histogram shows numerical data, bars touch. Used to view numerical data/frequency distribution
# Bar chart shows categorical data, bars dont touch. Used to view categorical/frequency distribution
# Pie charts shows relative size of each value w.r.t whole data

# Distribution plots: histplot, distplot, kdeplot
# Categorical plots: boxplot, catplot, stripplot, swarmplot, violinplot, countplot
# Relational plots: lineplot, scatterplot
# Matrix plots: heatmap, clustermap
# Multi-plot grids: pairplot, facetgrid
# Joint grids: jointplot, jointgrid
# Regression plots: lmplot
# =============================================================================

tips = sns.load_dataset('tips')
tips.sample(5)

'''Numerical plots to view their distribution'''
# distplot is for univariate samples, for distribution of single feature
sns.distplot(tips['total_bill'])
# kernel density estimation, curve of graph
sns.distplot(tips['total_bill'], kde = False, bins = 40)
# jointplot is to match 2 dist plots to match bivariate data, combine 2 distribution plots
sns.jointplot(x = 'total_bill', y = 'tip', data = tips, kind = 'scatter') # kind: reg, hex , kde
# pairplot builds plots across all numerical features/columns in a dataset
sns.pairplot(tips, hue = 'sex', palette = 'coolwarm') # pass categorical col for hue, colors plot based on it
# kde plot for univariate sample data
sns.kdeplot(tips['total_bill'])
# rug plot, like distplot, draws dash for every data point for univariate samples
sns.rugplot(tips['total_bill'])

'''Categorical plots to view distribution of categorical cols w.r.t a categorical/numerical feature'''
# barplot aggregates categorical data based on a function, like visualization of 'group by'
sns.barplot(x = 'sex', y = 'total_bill', data = tips, estimator = np.std, hue = 'smoker') # std deviation, mean, median etc
# counplot gives a count of each class in a feature/column
sns.countplot(data = tips, x = 'sex')
# box plot shows distribution across categorical columns. Better to detect outliers in data set
'''Box plot has Minimum, First Quartile or 25%, Second Quartile or 50% or Median, Third Quartile or 75%, Maximum'''
sns.boxplot(x = 'day', y = 'total_bill', data = tips, hue = 'smoker') # hue splits data as per 3rd category
sns.boxplot(x = 'day', y = 'total_bill', data = tips, hue = 'sex')
# Violin plot
sns.violinplot(x = 'day', y = 'total_bill', data = tips, hue = 'smoker', split = True)
# Strip plot to compare a numerical feature with categorical
sns.stripplot(x = 'day', y = 'total_bill', data = tips, jitter = True, hue = 'sex', dodge = True)
# Swarm plot combines strip plot and violin plot loosely. Not ideal for large datasets
sns.swarmplot(x = 'day', y = 'total_bill', data = tips, hue = 'smoker', dodge = True)
# combining swarm plot and violin plot on top of each other
sns.violinplot(x = 'day', y = 'total_bill', data = tips)
sns.swarmplot(x = 'day', y = 'total_bill', data = tips, color = 'green')
# cat plot is a general form of all the above plots, having parameter 'kind' to mention as an arguement
sns.catplot(x = 'day', y = 'total_bill', data = tips, kind = 'swarm') # default is strip, others are bar, box, swarm, violin
# Heatmap. transform numerical data to have rows & columns by either using corr() or pivot_table()
tc = tips.corr()
# plt.figure(figsize = (20, 1))
sns.heatmap(data = tc, annot = True, cmap = 'magma')
# flights data
flights = sns.load_dataset('flights')
flights.head()

fp = flights.pivot_table(index = 'month', columns = 'year', values = 'passengers')
sns.heatmap(data = fp, annot = False, cmap = 'magma', linecolor = 'white') # , lw = .2
# Clustermap
sns.clustermap(data = fp, standard_scale = 1)
# regression Plots
sns.lmplot(x = 'total_bill', y = 'tip', data = tips, hue = 'sex', markers = ['o', 'v'], scatter_kws = {'s': 40},
           aspect = .6, height = 6)
sns.lmplot(x = 'total_bill', y = 'tip', data = tips, col = 'sex', row = 'time', aspect = .6, height = 8)

# Pair grid gives control over pair plot with plot styles
iris = sns.load_dataset(name = 'iris')
iris.head()

sns.pairplot(data = iris)
### Pair grid
# single plot style
ire = sns.PairGrid(data = iris)
ire.map(plt.scatter)
# multiple plot styles
ire = sns.PairGrid(data = iris)
ire.map_diag(sns.distplot)
ire.map_upper(plt.scatter)
ire.map_lower(sns.kdeplot)

### Facet grid
irs = sns.FacetGrid(data = tips, col= 'time', row = 'smoker')
irs.map(sns.distplot, 'total_bill')
irs.map(plt.scatter, 'total_bill', 'tip')

# style & color
tips.sample(5)
sns.set_style(style = 'ticks') # sets specified style for plots. whitegrid, darkgrid, dark, white, ticks
sns.countplot(x = 'sex', data = tips)
sns.despine() # removes top & right spines by default, other axis can also be removed..works only for ticks plot style

# applying matplotlib function to control figure size
plt.figure(figsize = (8, 8))
sns.set_style(style = 'ticks') # sets specified style for plots. whitegrid, darkgrid, dark, white, ticks
sns.countplot(x = 'sex', data = tips)
sns.despine() # removes top & right spines by default, other axis can also be removed..works only for ticks plot style

'''''''''Pandas built-in'''''''''

df1 = pd.DataFrame(columns = 'A B C D E'.split(), data = np.arange(1, 61).reshape(-1, 5))
df2 = pd.DataFrame(columns = 'P Q R S T'.split(), data = np.random.randn(1, 10, 8).reshape(-1, 5), index = np.arange(100, 116))
df1.head()
df2.head()
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
tips.sample(5)
flights.sample(5)

# for single feature aka univariate
tips['total_bill'].hist(bins = 15, color = 'C')
tips['tip'].plot(kind = 'box')
flights['passengers'].plot.kde()

# for 2 features aka bivariate
flights.plot(kind = 'line', x = 'year', y = 'passengers')
tips.plot.scatter(x = 'size', y = 'tip', cmap = 'viridis')
tips[['tip','total_bill']].plot.kde()
df = pd.DataFrame(data = np.random.randn(500, 2), columns = ['A', 'B'])
df.plot.hexbin(x = 'A', y = 'B', gridsize = 50, figsize = (25, 25), cmap = 'viridis')

'''Each pie is a category of a pandas series & its spread is based on its corresponding numeric value,
cant be used for big sample data unless aggregated as per mean/count etc'''
x = tips.groupby('sex').mean().T
x.plot.pie(subplots = True, labels = tips['sex'].values.tolist(), radius = 1, autopct = '%0.2f%%', explode = [0,.3,0], startangle = 45, shadow = True)
plt.tight_layout()
plt.show()

# for multiple features/entire dataframe aka multivariate
tips.plot.hist(bins = 15)
flights[['year', 'passengers']].plot.bar()
tips.plot(kind = 'box')
tips.count().plot.bar(stacked = True)
tips[['total_bill', 'tip', 'size']].plot.area(alpha = .6)
df.plot.kde()

'''''''''Plotly & Cufflinks'''''''''

import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True) # specific to Jupyter
cf.go_offline() # allows to use cufflinks offline
print(plotly.__version__)

df = pd.DataFrame(data = np.random.randn(200, 4), columns = ['A', 'B', 'C', 'D'])
df.sample(4)

df.iplot()
df.iplot(kind = 'bar', x = 'A', y = 'D')
tips.count().iplot(kind = 'bar')
tips.iplot(kind = 'box')
tips.iplot(kind = 'hist', bins = 30)
tips['size'].iplot(kind = 'hist', bins = 30)

df2 = pd.DataFrame({'x': [x for x in np.arange(23, 34)], 'y': [x for x in np.arange(163, 174)], 'z':
                    [x for x in np.arange(1114, 1103, -1)]})
df2.iplot(kind = 'surface')


'''''''''''''Categorical Variables Encoding''''''''''''''''

High dimensionality: bigger feature space, too many columns. Leads to Curse of Dimensionality
High cardinality: lot of unique values (a large k). Ex: A column with hundreds of distinct categories

Binarization: transforms categorical variables into binary values: LabelBinarizer() and MultiLabelBinarizer()
Nominal categorical encoders: OneHotEncoder, Hashing, LeaveOneOut, and Target encoding. Mode is sensible
Ordinal categorical encoders: Ordinal, Binary, LabelEncoder. Median is sensible

OneHotEncoder needs data in integer encoded form 1st to be encoded, it's not required for LabelBinarizer
OneHotEncoder can be avoided for high cardinality columns and decision tree-based algorithms.
OneHotEncoder can be used for multi column data, while LabelEncoder and LabelBinarizer are not helpful

https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159
https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
'''
# Ref: https://pbpython.com/categorical-encoding.html
headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration","num_doors", "body_style",
            "drive_wheels", "engine_location", "wheel_base", "length", "width", "height", "curb_weight", "engine_type",
            "num_cylinders", "engine_size","fuel_system", "bore", "stroke", "compression_ratio", "horsepower",
            "peak_rpm", "city_mpg", "highway_mpg", "price"]

# Read in the CSV file and convert "?" to NaN
cars = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\cars_price.data', header = None,
                   names = headers, na_values = '?' )
cars.head()
cars.dtypes
cars.info()
# probing for missing values to impute before encoding categorical variables
cars.isna().sum().sort_values(ascending = False)
# probing for missing values in features that may have values less than 15%
cars = cars[[cols for cols in cars if cars[cols].count() / len(cars.index) >= .15]]
cars.isna().sum().sort_values(ascending = False)
# creating separate data frame for categorical variables for easy of encoding operations
cat_vars = cars.select_dtypes(include = ['object'])
cat_vars.columns

### method 1: find & replace using dict for multiple columns in a single shot or individual columns
cat_vars.isna().any()
cat_vars[cat_vars.isna().any(axis = 1)]
cat_vars['num_doors'].value_counts()
cat_vars.fillna({'num_doors': 'four'}, inplace = True)
cat_vars['num_cylinders'].value_counts()
cat_vars['num_doors'].value_counts()
cleanup_nums = {'num_doors': {'four': 4, 'two': 2}, 'num_cylinders': {'four': 4, 'six': 6, 'five': 5, 'eight': 8,
                'two': 2, 'three': 3, 'twelve': 12}}
cat_vars.replace(cleanup_nums, inplace = True)
cat_vars.dtypes
# find & replace using dict seperately for individual categories
cat_vars['num_cylinders'].value_counts()
cylindr_map = {'four': 4, 'six': 6, 'five': 5, 'eight': 8, 'two': 2, 'three': 3, 'twelve': 12}
cat_vars.replace(cylindr_map, inplace = True)
cat_vars.dtypes

cat_vars['engine_location'].value_counts()
engine_maps = {'front': 0, 'rear': 1}
cat_vars.replace(engine_maps, inplace = True)
cat_vars.dtypes

### method 2: using map() function for individual columns. cant be used for multiple columns or entire data frame
cat_vars['num_cylinders'].value_counts()
cyclinder_dict = {'four': 4, 'six': 6, 'five': 5, 'eight': 8, 'two': 2, 'three': 3, 'twelve': 12}
cat_vars['num_cylinders'] = cat_vars['num_cylinders'].map(cyclinder_dict).astype(int)
cat_vars.dtypes

### method 3:factorize using pandas factorize function. Used to encode all categorical variables by .apply() method
cat_vars['aspiration'].unique()
cat_vars.loc[:, 'aspiration_encode'] = pd.factorize(cat_vars['aspiration'])[0].reshape(-1, 1)
# for entire data frame
cat_vars = cat_vars.apply(lambda x : pd.factorize(x)[0])
pd.factorize(cat_vars)[0].reshape(-1, 1) # to do

### method 4: using dict comprehension
cat_vars['fuel_system'].unique()
labels = cat_vars['fuel_system'].astype('category').cat.categories.tolist()
new_replacement = {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}
cat_vars.replace(new_replacement, inplace = True)
cat_vars.dtypes

### method 5: using label encoding with cat.codes accessor, it alphabetically labels from 0 to 10. Cant use for dataframe
'''
Categorical variables can be label encoded using pandas by first changing the object dtype of categorical variables
to category dtype, and then using .cat accessor. Better to typecast categorical features to a category dtype as the
operations are faster than object dtype
'''
cat_vars['body_style'] = cat_vars['body_style'].astype('category')
cat_vars['body_style'] = cat_vars['body_style'].cat.codes
# cat_vars['body_style_cat'] = cat_vars['body_style'].cat.codes # to create a new column with the encoding
cat_vars['body_style'] = cat_vars['body_style'].astype('category').cat.codes # Or same as above but in single line
cat_vars.dtypes

### method 6: one hot encoding using pandas get_dummies function
cat_vars['drive_wheels'].unique()
# for single column
cat_vars = pd.get_dummies(cat_vars, columns = ['drive_wheels'], prefix = 'body') # drop_first = True
# for multiple columns
cat_vars = pd.get_dummies(cat_vars, columns = ['drive_wheels', 'body_style'], prefix = ['body', 'drive'], drop_first = True)
# for entire data frame. creates a large sparse matrix, not condusive for high cardinal and high dimensional data
cat_vars = pd.get_dummies(cat_vars, columns = cat_vars.columns.tolist()) # use drop_first = True to avoid dummy variable trap
cat_vars.shape

### method 7: Custom Binary Encoding, encodes a single category within a feature to a numeric value and rest all to default value
cat_vars['engine_type'].value_counts()
cat_vars['engine_type'] = np.where(cat_vars['engine_type'].str.contains('ohc'), 1, 0)
cat_vars[['make', 'engine_type']]
# if a new column is required
cat_vars['OHC_code'] = np.where(cat_vars['engine_type'].str.contains('ohc'), 1, 0)
cat_vars[['make', 'engine_type', 'OHC_code']]

### method 8: using enumeration and For loop
cat_vars['drive_wheels']
for i, wheels in enumerate(cat_vars['drive_wheels']):
     if wheels == 'rwd':
         cat_vars['drive_wheels'][i] = 1
     elif wheels == 'fwd':
         cat_vars['drive_wheels'][i] = 2
     else:
         cat_vars['drive_wheels'][i] = 0

### method 9: using scikit-learn's LabelEncoder
lb = LabelEncoder()
cat_vars['make'].unique()
cat_vars['make_codes'] = lb.fit_transform(cat_vars['make']) # OR cat_vars['make'] = lb.fit_transform(cat_vars['make'])
cat_vars[['make_codes', 'make']]
lb.classes_

### method 10: using scikit-learn's One-Hot encoding
one_hot = OneHotEncoder()
cat_vars['fuel_type'].unique()
hot_encodes = one_hot.fit_transform(cat_vars['fuel_type'].values.reshape(-1, 1)).toarray()
cars_onehot = pd.DataFrame(hot_encodes,
                           columns = ['fuel_' + str(one_hot.categories_[0][i]) for i in range(len(one_hot.categories_[0]))])
cars_onehot_encoded = pd.concat([cat_vars, cars_onehot], axis = 1)

### method 11: Scikit-learn supports binary encoding by using LabelBinarizer
'''LabelBinarizer results in a new DataFrame with only the one hot encodings for the feature, it needs to be
concatenated back with the original DataFrame, which can be done via pandas .concat() method'''
cat_vars['body_style'].unique()
lbinary = LabelBinarizer()
binarized = lbinary.fit_transform(cat_vars['body_style']) # can pass numpy array or pandas series
lbinary_results = pd.DataFrame(binarized, columns = lbinary.classes_) # OR
cat_vars = pd.concat([cat_vars, lbinary_results], axis = 1)

### method 12:  functions from category_encoders: BinaryEncoder, PolynomialEncoder, BackwardDifferenceEncoder, HelmertEncoder
# BinaryEncoder, specify the columns to encode then fit and transform
encoder = ce.BinaryEncoder(cols = ['fuel_type'])
cat_vars = encoder.fit_transform(cat_vars)
cat_vars
# BinaryEncoder to encode all categorical columns. leads to sparse matrix just like pd.dummies
encoder = ce.BinaryEncoder(cols = cat_vars.columns.tolist())
cat_vars = encoder.fit_transform(cat_vars)
cat_vars.shape

# PolynomialEncoder, specify the columns to encode then fit and transform
encoder = ce.polynomial.PolynomialEncoder(cols = ['engine_type'])
encoder.fit_transform(cat_vars, verbose = 1).iloc[:, 0:7].head() # Only display the first 8 columns for brevity
# to encode all categorical columns
encoder = ce.polynomial.PolynomialEncoder(cols = cat_vars.columns.tolist())
cat_vars = encoder.fit_transform(cat_vars, verbose = 1)
cat_vars.shape

# BackwardDifferenceEncoder, specify the columns to encode then fit and transform
encoder = ce.BackwardDifferenceEncoder(cols = ['engine_type'])
encoder.fit_transform(cat_vars, verbose = 1).iloc[:, 0:7] # Only display the first 8 columns for brevity
cat_vars.head()
# to encode all categorical columns
encoder = ce.BackwardDifferenceEncoder(cols = cat_vars.columns.tolist())
encoder.fit_transform(cat_vars, verbose = 1)

# HelmertEncoder
encoder = ce.HelmertEncoder(cols = ['engine_type'], drop_invariant = True)
helmert_binarized = encoder.fit_transform(cat_vars)
cat_vars = pd.concat([cat_vars, helmert_binarized], axis = 1)
cat_vars

''' LabelEncoder vs OneHotEncoder vs LabelBinarizer vs MultiLabelBinarizer'''
# Use MultiLabelBinarizer for multiple labels per instance
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = np.array(data)
values1 = pd.Series(data)
# LabelEncoder & OneHotEncoder
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse = False)
# LabelEncoder changes to integer matrix (with numpy array)
integer_encoded = label_encoder.fit_transform(values) # can pass numpy array or pandas series
onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))
type(onehot_encoded)
print ("OneHot Encoder:", onehot_encoded)
# LabelEncoder changes to integer matrix (with pandas series)
integer_encoded1 = label_encoder.fit_transform(values1) # can pass numpy array or pandas series
onehot_encoded1 = onehot_encoder.fit_transform(integer_encoded1.reshape(-1, 1))
type(onehot_encoded1)
print ("OneHot Encoder:", onehot_encoded1)
# Binary encode: LabelBinarizer
# ex: 1
lb = LabelBinarizer()
labels = lb.fit_transform(values)
labels1 = lb.fit_transform(values1)
print ("Label Binarizer:", labels)
print ("Label Binarizer:", labels1)
# ex: 2
lb = LabelBinarizer()
lb.fit([1, 2, 6, 4, 2, 1, 4]) # or lb.fit_transform([1, 2, 6, 4, 2, 1, 4])
lb.transform([1, 6]) # binary values of only 2 inputs out of the entire list/classes
lb.classes_
# Binary encode: MultiLabelBinarizer
# ex 1:
mlb = MultiLabelBinarizer()
mlb.fit_transform([(1, 2), (3,)])
mlb.classes_
# ex 2:
data = np.array([[2.2, 5.9, -1.8], [5.4, -3.2, -4.4], [-1.9, 4.2, 3.8]])
mlb.fit(data)
mlb.classes_
# ex 3:
mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
list(mlb.classes_)
# ex 4:
city_tups = [('Texas', 'Florida'), ('California', 'Alabama'), ('Texas', 'Florida'), ('Delware',
             'Florida'), ('Texas', 'Alabama')]
mlb.fit_transform(city_tups)
mlb.classes_

# Values having frequencies
dummy_df_age = pd.DataFrame({'age': ['0-20', '20-40', '40-60','60-80']})
dummy_df_age['start'], dummy_df_age['end'] = zip(*dummy_df_age['age'].map(lambda x: x.split('-')))
dummy_df_age
def split_mean(x):
    split_list = x.split('-')
    mean = (float(split_list[0])+float(split_list[1]))/2
    return mean
dummy_df_age['age_mean'] = dummy_df_age['age'].apply(lambda x: split_mean(x))
dummy_df_age

'''Exercise'''
# Ref: https://www.datacamp.com/community/tutorials/categorical-data
airline = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\flights.csv')
# Ref: https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
data = {'Temperature': ['hot','cold','very hot','warm','hot','warm','warm','hot','hot','cold'],
        'Color': ['red','yellow','blue','blue','red','yellow','red','yellow','yellow','yellow'],
        'Target':[1,1,1,0,1,0,1,0,1,1]}
df = pd.DataFrame(data)


''''''''''''''''Categorical Variables Correlation/Feature Selection/Dimensionality Reduction''''''''''''''''

# https://datascienceplus.com/selecting-categorical-features-in-customer-attrition-prediction-using-python
# https://www.kaggle.com/prashant111/a-reference-guide-to-feature-selection-methods
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# https://www.kaggle.com/shakedzy/alone-in-the-woods-using-theil-s-u-for-survival
'''
Feature selection techniques:
    Supervised:
        filter - uses statistical measures to score correlation or dependence between features that can be filtered in/out
        wrapper
        embedded/intrinsic
    Unsupervised:
        PCA

Filter methods:
    Basic methods: remove constant and quasi-constant features
    Univariate feature selection: SelectKBest, SelectPercentile
    Information gain measures how much information the presence/absence of a feature contributes to make correct prediction
    Fischer score
    ANOVA F-Value for Feature Selection
    Correlation Matrix with Heatmap

Wrapper Methods:
    Forward Selection
    Backward Elimination
    Exhaustive Feature Selection
    Recursive Feature Elimination
    Recursive Feature Elimination with Cross-Validation

Embedded/Intrinsic Methods:
    LASSO Regression
    Random Forest Importance

SelectKBest, SelectPercentile works with numpy array

Prefer using wrapper methods to filter methods for selecting best features
Fisher Score computes chi-squared stats between each non-negative feature and class.

5 steps for feature selection methods:
Remove constant n quasi constant features using VarianceThreshold
SelectKBest
Recursive Feature Elimination
Correlation-matrix with heatmap
Random Forest Importance
'''

churn = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Churn_Customer.csv')

senior = {0 : 'No', 1 : 'Yes'}
'''difference between replace() & map(): replace has inplace parameter but map doesn't, so replace doesnt need to be assigned
to its column if inplace=True is used, but it must happen for map as inplace parameter is not available'''
churn['SeniorCitizen'].replace(senior), inplace = True) # same can be done as below using map
churn['SeniorCitizen'] = churn['SeniorCitizen'].map(senior)
# categorizing tenure col
def tenure(data):
    if 0 < data  <= 24:
        return 'Short'
    else:
        return 'Long'
churn['tenure'] = churn['tenure'].apply(tenure)
# categorizing MonthlyCharges col
def charges(data):
    if 0 < data  <= 70:
        return 'LowCharge'
    else:
        return 'HighCharge'
churn['MonthlyCharges'] = churn['MonthlyCharges'].apply(charges)
# replacing categorical values
recode = {'No phone service' : 'No', 'No internet service' : 'No', 'Fiber optic' : 'Fberoptic',
          'Month-to-month' : 'MtM', 'Two year' : 'TwoYr', 'One year' : 'OneYr', 'Electronic check' : 'check',
          'Mailed check' : 'check', 'Bank transfer (automatic)' : 'automatic', 'Credit card (automatic)' : 'automatic'}
churn.replace(recode, inplace = True)
# dropping irrelevant columns
churn.drop(['customerID', 'TotalCharges'], axis = 1, inplace = True)

# dataset after desired changes
churn.shape
churn.columns
churn.select_dtypes(include = ['object']).columns
churn.select_dtypes(include = ['int64', 'float64']).columns
churn.nunique()

# method 1: scikit-learn OrdinalEncoder()/LabelEncoder() and SelectKBest
''' Multiple Correspondence Analysis constructs low-dimensional visual representation of cat variable associations'''
mca = prince.MCA(n_components = 2, n_iter = 3, copy = True, check_input = True, engine = 'auto', random_state = 42)
churn_mca = mca.fit(churn)
ax = churn_mca.plot_coordinates(X = churn, ax = None, figsize = (8, 10), show_row_points = False, row_points_size = 0,
                                show_row_labels = False, show_column_points = True, column_points_size = 30,
                                show_column_labels = True, legend_n_cols = 1).legend(loc = 'center left',
                                                                            bbox_to_anchor = (1, 0.5))

# method 2: remove constant features. Works for numerical variables. Encode categorical variables to remove constant features
sat_train = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\satander train.csv') # nrows=35000
# drop TARGET label from X_train
sat_train.drop(labels = ['TARGET'], axis = 1, inplace = True)
sat_train_x = sat_train
sat_train_x.shape
sat_train_x.dtypes.unique()
# removing constant and quasi-constant features using sklearn variancethreshold to find constant features
thresh_variance = VarianceThreshold(threshold = 0)
thresh_variance.fit(sat_train_x)  # fit finds the features with zero variance
# get_support is boolean vector indicating which features are retained.
sum(thresh_variance.get_support()) # gives the no of features that are not constant
# alternate way of finding non-constant features
len(sat_train_x.columns[thresh_variance.get_support()])
# print the constant features
[x for x in sat_train_x.columns if x not in sat_train_x.columns[thresh_variance.get_support()]]
len([x for x in sat_train_x.columns if x not in sat_train_x.columns[thresh_variance.get_support()]])
# we can then drop these columns from the train and test sets
sat_train_x = thresh_variance.transform(sat_train_x)
# check the shape of training and test set
sat_train_x.shape

# method 3: Remove quasi-constant features
X_train = sat_train.copy()
thresh_variance = VarianceThreshold(threshold = 0.01)  # 0.1 indicates 99% of observations approximately
thresh_variance.fit(X_train)  # fit finds the features with low variance
# get_support is a boolean vector that indicates which features are retained
sum(thresh_variance.get_support()) # gives the no of features that are not constant
# alternative way of doing the above operation
len(X_train.columns[thresh_variance.get_support()])
# finally we can print the quasi-constant features
[x for x in X_train.columns if x not in X_train.columns[thresh_variance.get_support()]]
len([x for x in X_train.columns if x not in X_train.columns[thresh_variance.get_support()]])
# percentage of observations showing each of the different values
X_train['ind_var31'].value_counts() / np.float(len(X_train))
# we can then remove the features from training and test set
X_train = thresh_variance.transform(X_train)
# check the shape of training
X_train.shape

# method 4: pandas get_dummies and SelectKBest
churns = churn.copy()
churn1 = pd.get_dummies(churns, drop_first = True)
churn1.columns
x = churn1.drop('Churn_Yes', axis = 1) # input categorical features
y = churn1['Churn_Yes'] # categorical target variable
# selectKBest step has options to request scores for ‘all’ or ‘k’ no of categorical features
score_function = SelectKBest(score_func = chi2, k = 'all') # try score_func with mutual_info_classif or k = 2
sf_fit = score_function.fit(x, y)
# categorical features with the highest values for the chi-squared stat indicate higher relevance and importance
sf_fit.scores_
# scores as a pandas dataframe
datset = pd.DataFrame()
datset['feature'] = x.columns[range(len(sf_fit.scores_))]
datset['scores'] = sf_fit.scores_
datset = datset.sort_values(by = 'scores', ascending = False)
sns.barplot(datset['scores'], datset['feature'], color = 'blue')
sns.set_style('whitegrid')
plt.ylabel('Categorical Feature', fontsize = 18)
plt.xlabel('Score', fontsize = 18)
plt.show()
'''From chi2 scores & figure, top 10 categorical features to select are Contract_TwoYr, InternetService_Fiberoptic,
Tenure, InternetService_No, Contract_oneYr, MonthlyCharges, OnlineSecurity, TechSupport, PaymentMethod and SeniorCitizen'''

# method 5: pandas get_dummies and SelectPercentile
churns = churn.copy()
churn1 = pd.get_dummies(churns, drop_first = True)
churn1.columns
x = churn1.drop('Churn_Yes', axis = 1) # input categorical features
y = churn1['Churn_Yes'] # target variable
score_function = SelectPercentile(score_func = chi2, percentile = 10) # now select features based on top 10 percentile
sf_fit = score_function.fit_transform(x, y)
sf_fit.shape
# scores as a pandas dataframe
datset = pd.DataFrame()
datset['feature'] = x.columns[range(len(sf_fit.scores_))]
datset['scores'] = sf_fit.scores_
datset = datset.sort_values(by = 'scores', ascending = False)
# scores as a plain text
for i in range(len(sf_fit.scores_)):
    print(' %s: %.3f' % (x.columns[i], sf_fit.scores_[i]))
# scores as bar plot method 1
plt.bar([i for i in range(len(sf_fit.scores_))], sf_fit.scores_)
# scores as bar plot method 2
sns.barplot(datset['scores'], datset['feature'], color = 'blue')
sns.set_style('whitegrid')
plt.ylabel('Categorical Feature', fontsize = 18)
plt.xlabel('Score', fontsize = 18)
plt.show()

# method 6: scikit-learn OrdinalEncoder()/LabelEncoder() and SelectKBest
'''selecting feature subset using the chi2 test stat for categorical features.
Encode categorical features using pandas get_dummies or encoder libraries OrdinalEncoder()/LabelEncoder()
using SelectKBest. It's good to select important feature-levels when features comprises of more than two levels'''

churns = churn.copy()
x = churns.drop('Churn', axis = 1) # input features
y = churns['Churn'] # target variable
# encode the categorical features
oe = OrdinalEncoder()
X_enc = oe.fit_transform(x)
# prepare target variable
le = LabelEncoder()
y_enc = le.fit_transform(y)
# feature selection, compare Chi-Squared Statistics. select two features with highest chi-squared statistics
score_func = SelectKBest(score_func = chi2, k = 'all') # try score_func with mutual_info_classif
sf_fit = score_func.fit(X_enc, y_enc)
# scores as a pandas dataframe
datset = pd.DataFrame()
datset['feature'] = x.columns[range(len(sf_fit.scores_))]
datset['scores'] = sf_fit.scores_
datset = datset.sort_values(by = 'scores', ascending = False)
# scores as plain text
for i in range(len(sf_fit.scores_)):
    print(' %s: %.3f' % (x.columns[i], sf_fit.scores_[i]))
# scores as bar plot method 1
plt.bar([i for i in range(len(sf_fit.scores_))], sf_fit.scores_)
# scores as bar plot method 2
sns.barplot(datset['scores'], datset['feature'], color = 'green')
sns.set_style('whitegrid')
plt.ylabel('Categorical feature', fontsize = 18)
plt.xlabel('Score', fontsize = 18)
plt.show()
'''From chi2 scores & figure, top 10 categorical features to select are Contract, Tenure, MonthlyCharges,
OnlineSecurity, TechSupport, PaymentMethod, SeniorCitizen, Dependents, PaperlessBilling and Partner'''

# method 7: ANOVA F-value: works for numpy array
churns = churn.copy()
x = churns.drop('Churn', axis = 1) # input features
y = churns['Churn'] # target variable
# encode the categorical features
oe = OrdinalEncoder()
X_enc = oe.fit_transform(x)
# prepare target variable
le = LabelEncoder()
y_enc = le.fit_transform(y)
# Select features with ANOVA F-Values, create a SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(score_func = f_classif, k = 2)
# Apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(X_enc, y_enc)
# View results
x.shape[1]
X_kbest.shape[1]
# scores as plain text
for i in range(len(fvalue_selector.scores_)):
    print(' %s: %.3f' % (x.columns[i], fvalue_selector.scores_[i]))

# method 8: Random Forest Importance
df = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\mushrooms.csv')
# Declare feature vector and target variable
X = df.drop(['class'], axis = 1)
y = df['class']
# Encode categorical variables
X = pd.get_dummies(X, prefix_sep = '_')
y = LabelEncoder().fit_transform(y)
# Normalize feature vector
X2 = StandardScaler().fit_transform(X)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.30, random_state = 0)
# instantiate the classifier with n_estimators = 100
classifier = RandomForestClassifier(n_estimators = 100, random_state = 3)
# fit the classifier to the training set
classifier.fit(X_train, y_train)
# predict on the test set
y_pred = classifier.predict(X_test)
classifier.feature_importances_
# visualize feature importance
plt.figure(num = None, figsize = (10, 8), dpi = 80, facecolor = 'w', edgecolor = 'k')
feat_importances = pd.Series(classifier.feature_importances_, index = X.columns)
feat_importances.nlargest(7).plot(kind = 'barh')

# method 9: Forward Selection & Backward Elimination. works with numerical data. Try encoding categorical variables
url = 'D:\Programming Tutorials\Machine Learning\Projects\Datasets\House Price Adv Regression train.csv'
data = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\House Price Adv Regression train.csv')
data.shape
numerical_vars = list(data.select_dtypes(include = ['int64', 'float64']).columns)
data = data[numerical_vars]
data.shape
# separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels = ['SalePrice'], axis = 1), data['SalePrice'],
                                                    test_size = 0.3, random_state = 0)
X_train.shape
''' commented bcoz correlated features can be found using careful EDA techniques.
# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                col_name = corr_matrix.columns[i] # getting the name of column
                col_corr.add(col_name)
    return col_corr
corr_features = correlation(X_train, 0.8)
print('correlated features: ', len(set(corr_features)),'\n', set(corr_features))
# removed correlated  features
X_train.drop(labels = corr_features, axis = 1, inplace = True)
X_train.shape
'''
# filling missing values with numerical value since SFS takes in numerical values for feature selection
X_train.fillna(0, inplace = True)

# forward selection
sfs1 = sfs(RandomForestRegressor(), k_features = 10, forward = True, floating = False, verbose = 2, scoring = 'r2', cv = 3)
sfs1 = sfs1.fit(np.array(X_train), y_train) # time taking for performing feature selection
sfs1.k_feature_idx_
X_train.columns[list(sfs1.k_feature_idx_)]
# backward elimination
sfs1 = sfs(RandomForestRegressor(), k_features = 10, forward = False, floating = False, verbose = 2, scoring = 'r2', cv = 3)
sfs1 = sfs1.fit(np.array(X_train), y_train)
sfs1.k_feature_idx_
X_train.columns[list(sfs1.k_feature_idx_)]

# method 10: Recursive feature elimination
digits = load_digits()
x = digits.images.reshape((len(digits.images), -1))
y = digits.target
# create the RFE object and rank each pixel
svc = SVC(kernel = 'linear', C = 1)
rfe = RFE(estimator = svc, n_features_to_select = 1, step = 1)
rfe.fit(x, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)
rfe.support_ # mask of selected features
rfe.ranking_ # feature ranking, estimated best features are assigned rank 1.
# plot pixel ranking
# plt.plot(ranking)
plt.matshow(ranking, cmap = plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()

# method 11: Recursive feature elimination with cross-validation
# A recursive feature elimination gives automatic tuning of the no of features selected with cross-validation
digits = load_digits()
x = digits.images.reshape((len(digits.images), -1))
y = digits.target
# Create the RFE object and rank each pixel
svc = SVC(kernel = 'linear', C = 1)
rfecv = RFECV(estimator = svc, step = 1, cv = StratifiedKFold(2), scoring = 'accuracy')
rfecv.fit(x, y)
rankings = rfecv.ranking_.reshape(digits.images[0].shape)
rfecv.support_ # The mask of selected features.
rfecv.ranking_ # The feature ranking, estimated best features are assigned rank 1.
# Plot pixel ranking
plt.matshow(rankings, cmap = plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()

# Method 12
data = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\mushrooms.csv')
# unique values of each feature
for feature in data.columns:
    uniq = data[feature].unique() # np.unique(data[feature])
    print('{}: {} distinct values - {}'.format(feature, len(uniq), uniq))

# probing for duplicates with same features and class
print('Known mushrooms: {}\nUnique mushrooms: {}'.format(len(data.index), len(data.drop_duplicates().index)))

# probing for mushrooms with the same features but different classes
print('Known mushrooms: {}\nMushrooms with same features: {}'.format(len(data.index),
      len(data.drop_duplicates(subset = data.drop(['class'], axis = 1).columns).index)))

corrs = data.apply(lambda x : pd.factorize(x)[0]).corr(method = 'pearson', min_periods = 1)

# Method 13
ass = associations(data, theil_u = True, figsize = (15, 15))

'''DEBUG'''
x = [3,8,7,9,1]
y = 2
z = 3
result = y + z
print(result)
result2 = x + y
pdb.set_trace()
print(result2)

'''TensorFlow'''

y_true = np.array([[2], [1], [0], [3], [0]]).astype(np.int64)
y_true = tf.identity(y_true)

y_pred = np.array([[0.1, 0.2, 0.6, 0.1],
                   [0.8, 0.05, 0.1, 0.05],
                   [0.3, 0.4, 0.1, 0.2],
                   [0.6, 0.25, 0.1, 0.05],
                   [0.1, 0.2, 0.6, 0.1]
                   ]).astype(np.float32)
y_pred = tf.identity(y_pred)

_, m_ap = tf.metrics.average_precision_at_k(y_true, y_pred, 3)

sess = tf.Session()
sess.run(tf.local_variables_initializer())

stream_vars = [i for i in tf.local_variables()]

tf_map = sess.run(m_ap)
print(tf_map)

print((sess.run(stream_vars)))

tmp_rank = tf.nn.top_k(y_pred,3)

print(sess.run(tmp_rank))
''''''

lists = [True, True, True, False, True]
any(lists)
all(lists)

arrays = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']),
          np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]

s = pd.Series(np.random.randn(8), index = arrays)
s
df = pd.DataFrame(np.random.randn(8, 4), index = arrays)
df

a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
for a in a_dict:
    print(a, '->', a_dict[a])

piss = [1, 5, 8, 4, 6, 7]
df = pd.DataFrame(piss).to_numpy().reshape(3,2)
df.reshape(2,3)

a = [[1.2,'abc',3],[1.2,'werew',4],[1.4,'qew',2]]
my_df = pd.DataFrame(a)
my_df.to_csv('my_csv.csv', index=False, header=False)

ab = np.asarray([[1,2,3], [4,5,6], [7,8,9]])
np.savetxt("foo.csv", ab, delimiter=",")
np.savetxt()
ab = np.asarray([[1,2,3], [4,5,6], [7,8,9]])
pd.DataFrame(ab).to_csv(columns = [], index = [], header = True)

bc = {100 : ['abc', 'def'], 200 : ['ghi', 'lkj']}
lists = []
for i in bc:
    for j in bc[i]:
        print(j)
        print(j[0])
        lists.append(j)

# reading csv file from url
data = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/nba.csv")
'''OR'''
data = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\nba.csv')
data.dropna(inplace = True)
data_dict = data.to_dict()
data_dict

df = pd.DataFrame({'a': ['red', 'yellow', 'blue'], 'b': [0.5, 0.25, 0.125]})
lists = df.to_dict('list')

'''Multiple column breaks'''
d_frame = pd.DataFrame({'EmployeeId': ['001', '002', '003', '004', '005'],
                   'City': ['Mumbai|Bangalore', 'Pune|Mumbai|Delhi', 'Mumbai|Bangalore', 'Mumbai|Pune',
                            'Bangalore']})
# Step 1
# We start with creating a new dataframe from the series with EmployeeId as the index
new_df = pd.DataFrame(d_frame.City.str.split('|').tolist(), index = d_frame.EmployeeId).stack()
# Step 2
# We now want to get rid of the secondary index
# To do this, we will make EmployeeId as a column (it can't be an index since the values will be
# duplicate)
new_df = new_df.reset_index([0, 'EmployeeId'])
# Step 3
# The final step is to set the column names as we want them
new_df.columns = ['EmployeeId', 'City']


'''Multiple values in a cell split'''
a = pd.DataFrame([{'var1': 'a', 'var2': 1}, {'var1': 'b', 'var2': 1}, {'var1': 'c', 'var2': 1},
                  {'var1': 'd', 'var2': 2}, {'var1': 'e', 'var2': 2}, {'var1': 'f', 'var2': 2}])
b = pd.DataFrame([{'var1': 'a,b,c', 'var2': 1}, {'var1': 'd,e,f', 'var2': 2}])
c = pd.concat([pd.Series(row['var2'], row['var1'].split(',')) for _, row in a.iterrows()]).reset_index()

'''Convert a dense matrix to sparse matrix & vice versa'''

array = np.array([[1, 0, 0, 1, 0, 0], [0, 0, 2, 0, 0, 1], [0, 0, 0, 2, 0, 0]])
print(array)
# convert to compressed sparse row (CSR) matrix
csr_mat = csr_matrix(array)
print(csr_mat)
# reconstruct dense matrix
dense_matrix = csr_mat.todense()
print(dense_matrix)
# calculate sparsity
sparsity = 1.0 - np.count_nonzero(array) / array.size
print(sparsity)

a = 14.54
print(a.as_integer_ratio())

'''''''''''''''''''''''''''
use custom functions as utility in another file
1. define a function/functions
2. save file as .py in same location where it is to be used in another file
3. use import function to import that .py file
4. access functions in .py file using imported file.function

use variables in another file (esp jupyter notebook)
1. declare variables that need to be accessed in another file using %store variable_name
2. access them using %store -r variable_name
'''''''''''''''''''''''''''
import case # case.py file in same workspace
case.upper_case('hey') # accessing function from it

'''''''''''''''''''''''''''Web Scraping'''''''''''''''''''''''''''
from bs4 import BeautifulSoup
import requests

page = 'D:\Programming Tutorials\Java & J2EE\HTML Practicals\HTML Practical\PrakOriginal.html'

with open(page) as html_file:
    soup = BeautifulSoup(html_file, 'lxml')

print(soup)
print(soup.prettify())

match = soup.title.text
print(match)
match = soup.h2
print(match)
match = soup.find('h2')

'''
page = requests.get('https://www.youtube.com/c/EconomicsExplained/videos').text
with open(page) as file:
    content = file.read()
    soup = BeautifulSoup(page, 'lxml')
    soup.prettify()
'''

page = requests.get('https://www.youtube.com/c/EconomicsExplained/videos').text
soup = BeautifulSoup(page, 'lxml')
#or
page = requests.get('https://www.youtube.com/c/EconomicsExplained/videos')
soup = BeautifulSoup(page.content)

soup.prettify()
title = soup.find('div', class_ = 'yt-simple-endpoint style-scope ytd-grid-video-renderer')
type(title)

anchors = soup.find_all('a', class_ = 'yt-simple-endpoint style-scope ytd-grid-video-renderer')
type(anchors)
[a for a in anchors]

headers = soup.find(['h1', 'h2'])
headers

para = soup.find_all('p1')
para

soup.body.prettify()
soup.select('a.class')
soup.find('')

https://www.youtube.com/watch?v=GjKQ6V_ViQE
https://www.youtube.com/watch?v=Ewgy-G9cmbg
https://www.youtube.com/watch?v=sAuGH1Kto2I
https://www.youtube.com/playlist?list=PLGKQkV4guDKEKZXAyeLQZjE6fulXHW11y

=======================================================================================================

from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
df1.to_sql('my_table', engine)
sql_data = pd.read_sql('my_table', con = engine)
sql_data

'''Python – MongoDB integration'''

import pymongo as pym
# creating client for mongo DB
client = pym.MongoClient('mongodb://127.0.0.1:27017/') #'''protocol/ip address/port no'''
# creating/accessing DB
my_db = client['Ex_Employees'] # same as client1.Ex_Students
# creating collection
information = my_db.ex_emp_collection
# creating records
docs = [{'username' : 'Srees', 'alias' : 'v-sresur', 'age' : 34, 'guage' : 78},
        {'username' : 'Bees', 'alias' : 'v-brebur', 'age' : 43, 'guage' : 87}]
# persisting to the DB
information.insert_one(records) '''use insert_many to insert multiple records as a list'''

'''Python – PostgreSQL integration'''

# Ref: https://www.youtube.com/watch?v=2PDkXviEMD0&t=1s
import psycopg2 as pgs
# connect to the DB
connect = pgs.connect(host = 'localhost', database = 'My DB', user = 'postgres', password = 'Srees1!', port = 5432)
# cursor to interact with DB. client side n server side cursors
cursor = connect.cursor()
# execute query for a DB
query = 'select * from actor limit 5'
cursor.execute(query)
data = cursor.fetchall()
for item in data:
    #print(f'Actor ID {item[0]}' 'First Name {item[1]}' 'Last Name {item[2]}')
    print('Actor name is {} {}, with ID {}'.format(item[1], item[2], item[0]))
# execute data insertion into the table
values_list = [(203, 'Jack', 'Ass'), (204, 'Dumb', 'Ass'), (205, 'Moron', 'Ass')]
query = "insert into actor (actor_id, first_name, last_name) values (values_list)"
query = "insert into actor (actor_id, first_name, last_name) values (203, 'Jack', 'S')"
cursor.execute(query)
connect.commit() # commit the trasaction
cursor.close() # close cursor
connect.close() # closing connection

'''Python – SQL integration'''
import sqlite3
connect = sqlite3.connect('data.db')
cursor = connect.cursor()

create_query = 'create table game (game_id integer not null primary key, game_name varchar (20), video longblob)'
cursor.execute(create_query)

insert_query = 'INSERT INTO game values (3, "Terminator", "C:\\Users\\Srees\\Desktop\\Invictus.srt")'
cursor.execute(insert_query)
connect.commit()

search_query = 'select * from game'
result = cursor.execute(search_query)

for res in result:
    print(res)
connect.close()

'''''''''''''Foot Notes''''''''''''''

Coding:
    https://www.geeksforgeeks.org/python-programming-examples
    https://pynative.com/python-exercises-with-solutions
    https://www.w3resource.com/python-exercises

Python OOP practice:
    https://pynative.com/python-object-oriented-programming-oop-exercise/ DONE
    https://www.w3resource.com/python-exercises/class-exercises/
    https://www.pythonprogramming.in/object-oriented-programming.html
    https://medium.com/@gurupratap.matharu/object-oriented-programming-project-in-python-for-your-github-portfolio-d34feaf1332c
    https://www.my-courses.net/2020/02/exercises-with-solutions-on-oop-object.html
    https://hub-courses.pages.pasteur.fr/python-solutions/Object_Oriented_Programming.html

Python DS:
    https://www.w3schools.com/python/python_tuples.asp

RegEx:
    https://www.w3resource.com/python-exercises/re

PostgreSql:
    https://www.w3resource.com/sql/group-by.php
    https://www.w3resource.com/sql-exercises/sql-aggregate-functions.php

Numpy:
    https://pynative.com/python-numpy-exercise
    https://www.kaggle.com/utsav15/100-numpy-exercises
    https://www.machinelearningplus.com/python/101-numpy-exercises-python/

Pandas:
    https://pynative.com/python-pandas-exercise
    https://www.machinelearningplus.com/python/101-pandas-exercises-python
    https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset
    https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset-solution
    https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
    https://thispointer.com/python-pandas-how-to-add-rows-in-a-dataframe-using-dataframe-append-loc-iloc

Matplotlib:
    https://pynative.com/python-matplotlib-exercise
    https://elitedatascience.com/python-seaborn-tutorial
    https://www.w3resource.com/graphics/matplotlib/basic/index.php

List:
    https://www.codesdope.com/practice/python-make-a-list

Map, Filter, Reduce:
    https://www.learnpython.org/en/Map,_Filter,_Reduce

Hypothesis Test:
    T Test
    Z Test
    ANOVA Test
    Chi-Square Test
    https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f
    https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce
    https://github.com/yug95/MachineLearning/tree/master/Hypothesis%20testing
    https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
    https://towardsdatascience.com/1-way-anova-from-scratch-dissecting-the-anova-table-with-a-worked-example-170f4f2e58ad
    https://pythonfordatascience.org/anova-python/
    https://codingdisciple.com/hypothesis-testing-ANOVA-python.html
