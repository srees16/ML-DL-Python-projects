# Convolutional Neural Networks & Computer Vision

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

'''''''''''''''''''''''''''''''''''''''''''''Convolutional Neural Network'''''''''''''''''''''''''''''''''''''''''''''
# Choose neurons & hidden layers arbitrarily at first, Weights can be negative or positive

# Initializing CNN
classifier = Sequential()

# Steps 1: Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2: Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding 2nd layer & Max pooling
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN model to images
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
trainingSet = train_datagen.flow_from_directory('D:/Programming Tutorials/Machine Learning/Projects/Datasets/CNN/cnn_training_set/', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
testSet = test_datagen.flow_from_directory('D:/Programming Tutorials/Machine Learning/Projects/Datasets/CNN/cnn_test_set/', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
classifier.fit_generator(trainingSet, steps_per_epoch = 8000, epochs = 25, validation_data = testSet, validation_steps = 2000)

# Making new predictions
testImage = image.load_img('', target_size = (64, 64))
testImage = image.img_to_array(testImage)
testImage = np.expand_dims(testImage, axis = 0)
classifier.predict(testImage)
result = classifier.predict(testImage)
trainingSet.class_indices
if result[0][0] == 1:
    prediction = 'Doggy'
else:
    prediction = 'Catty'

'''''''''''''''''''''''''''''''''''''''''''''Computer Vision'''''''''''''''''''''''''''''''''''''''''''''

np.random.seed(101)
arr = np.random.randint(0, 100, 10)
arr2 = np.random.randint(0, 100, 10)
arr.max()
arr.min()
arr.argmax()
arr.mean()
ars = np.random.randint(0, 100, 15).reshape(5, 3)

x = 'D:/Programming Tutorials/Machine Learning/Computer Vision/1 Computer Vision with OpenCV - Jose Portilla'
y = '/1. Course Overview and Introduction/3.1 Computer-Vision-with-Python/DATA/'
data_folder = x + y
# Each of the 3 channels (RGB) has a range of 0 to 255, 0 is no color, 255 is full color

url = '00-puppy.jpg'
pic = Image.open(data_folder + url)
pic
type(pic)

from skimage import io
pic = io.imread(data_folder + url)
plt.imshow(pic)

np_array_pic = np.asarray(pic)
np_array_pic.shape
mat_img = plt.imshow(np_array_pic)
type(mat_img)

array_pic_dupe = np_array_pic.copy()
plt.imshow(array_pic_dupe)
array_pic_dupe.shape

reds = array_pic_dupe[:, :, 0] # red channel
plt.imshow(reds) # this can be shown below in gray scale
plt.imshow(reds, cmap = 'gray') # red channel values range 0 - 255. 0 is no red or pure black, 255 is full red or white

greens = array_pic_dupe[:, :, 1] # green channel
plt.imshow(greens)
plt.imshow(greens, cmap = 'gray')

blues = greens = array_pic_dupe[:, :, 2] # blue channel
plt.imshow(blues)
plt.imshow(blues, cmap = 'gray')

# making the red channel 0
array_pic_dupe[:,:,0] = 0
plt.imshow(array_pic_dupe)

# making the green channel 0
array_pic_dupe[:,:,1] = 0
plt.imshow(array_pic_dupe)

# making the blue channel 0
array_pic_dupe[:,:,2] = 0
plt.imshow(array_pic_dupe) # shows black as values for all color channels RGB are 0

array_pic_dupe.shape # shows shape of 3 color channels since we view RGB together
array_pic_dupe[:, :, 2].shape # shows shape of 1 color channel since we view only 1 specfied channel of RGB

# section 2 learning
'''
for numpy array, order of tuple for shape is (height, width, channels)
for OpenCV, order of tuple for shape is (width, height, channels)
'''

img = cv2.imread(url)
type(img)
plt.imshow(img) # openCV reads the colors as BGR, so to fix it, image needs to be encoded from BGR to RGB
img.shape
fixed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fixed_img)

# reading the image to be in gray scale
img_gray = cv2.imread(url, cv2.IMREAD_GRAYSCALE)
img_gray.shape # gets rid of the 3 RGB channels due to greying of color image
img_gray.max(), img_gray.min()
plt.imshow(img_gray) # does not show in B&W due to default color mapping in opencv. use below to fix it to gray scale
plt.imshow(img_gray, cmap = 'gray')
plt.imshow(img_gray, cmap = 'magma') # in magma scale

# resizing images
plt.imshow(fixed_img)
fixed_img.shape
new_img = cv2.resize(fixed_img, (1000, 400))
plt.imshow(new_img)

# resizing images by ratio
help(cv2.resize)
img_height = .5 # 50% of the original
img_width = .5 # 50% of the original
resize_img = cv2.resize(fixed_img, (0,0), dst = fixed_img, fx = img_height, fy = img_width)
plt.imshow(resize_img)
resize_img.shape

# flipping images
help(cv2.flip)
flip_img = cv2.flip(src = fixed_img, flipCode = 0) # vertical flip
plt.imshow(flip_img)
flip_img = cv2.flip(src = fixed_img, flipCode = 1) # horizontal flip
plt.imshow(flip_img)
flip_img = cv2.flip(src = fixed_img, flipCode = -1) # flip horizontal and vertical
plt.imshow(flip_img)
type(flip_img)
type(fixed_img)

# saving image
cv2.imwrite('flipped.jpg', flip_img)

# increasing the display size of image in console/jupyter canvas
fig = plt.figure(figsize = (4, 4))
ax = fig.add_subplot(111)
ax.imshow(fixed_img)

# dealing with error when opening images outside console. Python script can be run to quit by Esc key etc
img = cv2.imread(url)
cv2.imshow('puppy', img)
cv2.waitKey(3)

# drawing shapes on images on a blank image
blank_img = np.zeros(shape = (512, 512, 3), dtype = np.int16)
plt.imshow(blank_img)

# rectangle 1
help(cv2.rectangle)
cv2.rectangle(img = blank_img, pt1 = (384, 0), pt2 = (490, 150), color = (255, 0, 0), thickness = 5) # pt1 is top-left, pt2 is bottom-right
plt.imshow(blank_img)

# rectangle 2
cv2.rectangle(img = blank_img, pt1 = (200, 200), pt2 = (300, 300), color = (0, 255, 0), thickness = 5) # green
plt.imshow(blank_img)

# circle 1
cv2.circle(img = blank_img, center = (100, 100), radius = 50, color = (0, 0, 255), thickness = 4) # blue
plt.imshow(blank_img)

# circle 2
cv2.circle(img = blank_img, center = (400, 400), radius = 50, color = (255, 0, 0), thickness = -1)
plt.imshow(blank_img)

# line
cv2.line(blank_img, pt1 = (0, 0), pt2 = (500, 500), color = (0, 0, 255), thickness = 4)
plt.imshow(blank_img)

# put text on image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img = blank_img, text = 'hello', org = (10, 500), fontFace = font, fontScale = 2, color = (205, 135, 0), thickness = 3)
plt.imshow(blank_img)

# custom shapes
blank_img1 = np.zeros(shape = (512, 512, 3), dtype = np.int32)
plt.imshow(blank_img1)
vertices = np.array([[100, 400], [200, 250], [400, 200], [450, 110]], dtype = np.int32)
vertices.shape
pts = vertices.reshape((-1, 1, 2))
pts.shape
cv2.polylines(img = blank_img1, pts = [pts], isClosed = True, color = (55, 205, 155), thickness = 3)
plt.imshow(blank_img1)

# adding events 1
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img = img, center = (x, y), radius = 50, color = (255, 0, 0), thickness = -1)

cv2.namedWindow(winname = 'drawing')
cv2.setMouseCallback('drawing', draw_circle)

img = np.zeros((512, 512, 3), np.int8)
while True:
    cv2.imshow('drawing', img)
    if cv2.waitKey(2) & 0xFF == 27:
        break
cv2.destroyAllWindows()

# adding events 2
img = np.zeros((512, 512, 3), np.int8)
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img = img, center = (x, y), radius = 50, color = (255, 0, 0), thickness = -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img = img, center = (x, y), radius = 50, color = (0, 255, 0), thickness = -1)
    elif event == cv2.EVENT_MOUSEWHEEL:
        cv2.circle(img = img, center = (x, y), radius = 50, color = (0, 0, 255), thickness = -1)
cv2.namedWindow(winname = 'drawing')
cv2.setMouseCallback('drawing', draw_circle)
while True:
    cv2.imshow('drawing', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()

# adding events 3
drawing = False # True while mouse button is down, False wen mouse button is up
ix, iy = -1, -1
def draw_rect(event, x, y, flags, params):
    global ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img = img, pt1 = (ix, iy), pt2 = (x, y), color = (0, 255, 0), thickness = -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img = img, pt1 = (ix, iy), pt2 = (x, y), color = (0, 255, 0), thickness = -1)
img = np.zeros((512, 512, 3))
cv2.namedWindow(winname = 'drawn')
cv2.setMouseCallback('drawn', draw_rect)
while True:
    cv2.imshow('drawn', img)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cv2.destroyAllWindows()

'''assessment'''

url = 'dog_backpack.jpg'
dog = cv2.imread(data_folder + url)
plt.imshow(dog)
fix_dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)
plt.imshow(fix_dog)
vertical_flip = cv2.flip(fix_dog, 0)
plt.imshow(vertical_flip)


'''
3 Color models:
RGB
HSL (Hue, Saturation, Lightness)
HSV (Hue, Saturation, Value)
'''

# converting images from RGB/BGR to HSL/HSV
img = cv2.imread(url)
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # converting to HSL or HSV
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # converting to HSL or HSV
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) # converting to HSL or HSV
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) # converting to HSL or HSV
plt.imshow(img)

'''Blending and Pasting Images'''

# loading & converting images
url1 = 'watermark_no_copy.png'
img1 = cv2.imread(data_folder + url)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1)
img1.shape
img2 = cv2.imread(data_folder + url1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
img2.shape

# 1. blending by resizing to be of same dimensions. addWeighted works only for images having same size
img1 = cv2.resize(src = img1, dsize = (1200, 1200))
img2 = cv2.resize(src = img2, dsize = (1200, 1200))
plt.imshow(img1)
plt.imshow(img2)
blender = cv2.addWeighted(src1 = img1, alpha = .9, src2 = img2, beta = .1, gamma = 0)
plt.imshow(blender)

# 2. overlay small image on top of large image (without blending)
img1 = cv2.imread(url)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread(url1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, dsize = (600, 600))

plt.imshow(img1)
plt.imshow(img2)

large_image = img1
small_image = img2
large_image.shape
small_image.shape

x_offset = 0
y_offset = 0
x_end = x_offset + small_image.shape[1]
y_end = y_offset + small_image.shape[0]
large_image[y_offset: y_end, x_offset: x_end] = small_image
plt.imshow(large_image)

# 3. blend images of different sizes
img1 = cv2.imread(url)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread(url1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, dsize = (600, 600))

img1.shape
img2.shape
x_offset = 934 - 600
y_offset = 1401 - 600
rows, cols, channels = img2.shape
roi = img1[y_offset: 1401, x_offset: 934]
plt.imshow(roi)

img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
plt.imshow(img2gray, cmap = 'gray')

mask_inv = cv2.bitwise_not(img2gray)
plt.imshow(mask_inv, cmap = 'gray')
mask_inv.shape

# white background
white_bg = np.full(shape = img2.shape, fill_value = 255, dtype = np.uint8)
plt.imshow(white_bg)
# plt.grid(None)

'''
# black background
black_bg = np.full(shape = img2.shape, fill_value = 0, dtype = np.uint8)
plt.imshow(black_bg)
plt.grid(None)
'''

bk = cv2.bitwise_or(white_bg, white_bg, mask = mask_inv)
bk.shape
plt.imshow(bk)

fore_ground = cv2.bitwise_or(img2, img2, mask = mask_inv)
plt.imshow(fore_ground)

final_roi = cv2.bitwise_or(src1 = roi, src2 = fore_ground)
plt.imshow(final_roi)

large_img = img1
small_img = final_roi

# putting it back to original image
large_img[y_offset: y_offset + small_img.shape[0], x_offset: x_offset + small_img.shape[1]] = small_img
plt.imshow(large_img)

'''Thresholding'''
'''
It is a method to segment different parts of an image. Binary threshold converts image to 2 values: white or black
gray threshold: gray
'''
img1 = cv2.imread(data_folder + 'rainbow.jpg', 0) # 0 shows image in gray scale
plt.imshow(img1)
plt.imshow(img1, cmap = 'gray')
ret, thresh1 = cv2.threshold(src = img1, thresh = 128, maxval = 255, type = cv2.THRESH_BINARY)
ret
thresh1
plt.imshow(thresh1, cmap = 'gray')

# same threshold in inverse
ret, thresh1 = cv2.threshold(src = img1, thresh = 127, maxval = 255, type = cv2.THRESH_BINARY_INV)
plt.imshow(thresh1, cmap = 'gray')

# THRESH_TRUNC truncation: if image is above threshold, rounds off back to threshold, less than threshold keeps image
ret, thresh1 = cv2.threshold(src = img1, thresh = 127, maxval = 255, type = cv2.THRESH_TRUNC)
plt.imshow(thresh1, cmap = 'gray')

# THRESH_TOZERO
ret, thresh1 = cv2.threshold(src = img1, thresh = 127, maxval = 255, type = cv2.THRESH_TOZERO)
plt.imshow(thresh1, cmap = 'gray')

# new image Crossword
url = 'crossword.jpg'
img = cv2.imread(data_folder + url)
plt.imshow(img)
img1 = cv2.imread(url, 0) # in viridis since its default
plt.imshow(img1)
plt.imshow(img1, cmap = 'gray') # gray scale

# function to display image
def show_pic(img):
    fig = plt.figure(figsize = (15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = 'gray')
show_pic(img1)

# applying binary threshold to image
ret, th1 = cv2.threshold(src = img1, thresh = 200, maxval = 255, type = cv2.THRESH_BINARY) # threshold value can be experimented for better clarity
show_pic(th1)

# adapt automatically the threshold based on pixel and where gray is around pixels
th2 = cv2.adaptiveThreshold(src = img1, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 8)
show_pic(th2)

# blending above 2 images as experiment
blended = cv2.addWeighted(src1 = th1, alpha = 0.6, src2 = th2, beta = 0.3, gamma = 0)
show_pic(blended)

'''Blurring & Smoothing'''

'''blurring and smoothing gets rid of noise and helps CV app to focus on imp details. edge detection detects edges in
hi-res image. edge detection method usually applied after blurring & smoothing for better edge detection'''

# 2 methods: 1. Gamma correction: makes an image bright or dark depending on gamma value 2. Kernel based filters

def load_img():
    url = 'bricks.jpg'
    img1 = cv2.imread(data_folder + url).astype(np.float32) / 255
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    return img1
load_img()

def display(img):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    return ax.imshow(img)
i = load_img()
display(i)

# applying gamma correction
gamma = 1/10 # gamma for increasing/decreasing brightness
result = np.power(i, gamma)
display(result)

# blurring by low pass filter of 2D convolution
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text = 'bricks', org = (10, 600), fontFace = font, fontScale = 6, color = (255, 0, 0), thickness = 4)
display(img)

# setting up kernel for low pass filter
kernel = np.ones(shape = (5, 5), dtype = np.float32) / 25
kernel
# applying 2D filter
dst = cv2.filter2D(img, -1, kernel) # -1 is ddepth parameter
display(dst)

# reloading previously used image and apply blurring
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text = 'bricks', org = (10, 600), fontFace = font, fontScale = 6, color = (255, 0, 0), thickness = 4)
display(img)

blurred = cv2.blur(img, ksize = (10, 10)) # kernel size can be played with
display(blurred)

# reloading previously used image and apply gaussing blurring
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text = 'bricks', org = (10, 600), fontFace = font, fontScale = 6, color = (255, 0, 0), thickness = 4)
display(img)

blurred = cv2.GaussianBlur(img, ksize = (5, 5), sigmaX = 10) # kernel size can be played with
display(blurred)

# reloading previously used image and apply median blurring
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text = 'bricks', org = (10, 600), fontFace = font, fontScale = 6, color = (255, 0, 0), thickness = 4)
display(img)

blurred = cv2.medianBlur(img, 5) # kernel size can be played with
display(blurred)

url = data_folder + 'sammy.jpg'
img = cv2.imread(url)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display(img)

noise_img = cv2.imread(data_folder + 'sammy_noise.jpg')
display(noise_img)
median = cv2.medianBlur(noise_img, 5)
display(median)

'''Morphological operators: helps reduce noise, and various effects on images like erosion & dilation'''

url = 'sammy.jpg'

# function to load image
def load_img():
    blank_img = np.zeros((600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text = 'ABCDE', org = (50, 300), fontFace = font, fontScale = 5, color = (255, 255, 255), thickness = 25)
    return blank_img
load_img()

def display(img):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = 'gray')

# trying with various iterations
i = load_img()
display(i)

kernel = np.ones((5, 5), dtype = np.uint8)
kernel
result = cv2.erode(i, kernel, iterations = 1)
display(result)

# reload n check with other iterations
i = load_img()
display(i)

kernel = np.ones((5, 5), dtype = np.uint8)
kernel
result = cv2.erode(i, kernel, iterations = 3)
display(result)

'''Opening: erosion followed by dilation. removes background noise'''

# creating white noise
img = load_img()
white_noise = np.random.randint(0, 2, (600, 600))
white_noise
display(white_noise)

white_noise = white_noise * 255
display(white_noise)

noise_img = white_noise + img
display(noise_img)

# applying opening
opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
display(opening), display(img)

'''Closing'''

# creating foreground noise
img = load_img()
black_noise = np.random.randint(0, 2, (600, 600))
black_noise = black_noise * -255
black_noise
display(black_noise)
black_noise_img = black_noise + img
black_noise_img
black_noise_img[black_noise_img == -255] = 0
black_noise_img.min()
display(black_noise_img)

closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)
display(closing)

'''Morphological gradient'''

img = load_img()
display(img)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
display(gradient)

'''Gradients: an extension of morphological operators. gradient is a directional change in color or intensity of image'''

# Sobel-Feldman operators: used for edge detection. x directional & y directional gradients or x-gradient, y-gradient
url = 'sudoku.jpg'
img = cv2.imread(data_folder + url, 0) # 0 makes image in red-gray scale
display(img)

# x gradient
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5) # 1 for x gradient, 0 for y gradient
display(sobelx)

# y gradient
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5) # 1 for x gradient, 0 for y gradient
display(sobely)

# Laplace operator
laplacian = cv2.Laplacian(img, cv2.CV_64F)
display(laplacian)

# blending x and y gradients for better quality than laplacian operator
blended = cv2.addWeighted(src1 = sobelx, alpha = .7, src2 = sobely, beta = .7, gamma = 0)
display(blended)

# trying thresholding and morphological operators on the above output image
res, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
display(th1)

# morphological operator
kernel = np.ones((4, 4), np.uint8)
gradient = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
display(gradient)

'''Histograms'''

horse = cv2.imread(data_folder + 'horse.jpg') # read as per OpenCV's BGR
show_horse = cv2.cvtColor(horse, cv2.COLOR_BGR2RGB) # converted to RGB
display(show_horse)

rainbow = cv2.imread(data_folder + 'rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)
display(show_rainbow)

bricks = cv2.imread(data_folder + 'bricks.jpg')
show_bricks = cv2.cvtColor(bricks, cv2.COLOR_BGR2RGB)
display(show_bricks)

# calculating histogram values for horse
hist_vals = cv2.calcHist([horse], channels = [0], mask = None, histSize = [256], ranges = [0, 256])
hist_vals.shape
plt.plot(hist_vals)

# calculating histogram values for bricks
hist_vals = cv2.calcHist([bricks], channels = [0], mask = None, histSize = [256], ranges = [0, 256])
hist_vals.shape
plt.plot(hist_vals)

# calculating histogram values for bricks
img = bricks
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], channels = [i], mask = None, histSize = [256], ranges = [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.title('Bricks color histogram')
display(show_bricks)

# calculating histogram values for horse
img = horse
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], channels = [i], mask = None, histSize = [256], ranges = [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 50])
    plt.ylim([0, 500000])
plt.title('Horse color histogram')
display(show_horse)

'''Histograms on masked portion of image'''

rainbow = cv2.imread(data_folder + 'rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)
display(show_rainbow)

img = rainbow
img.shape

mask = np.zeros(img.shape[:2], np.uint8)
plt.imshow(mask)
mask[300:400, 100:400] = 255
plt.imshow(mask, cmap = 'gray')

masked_img = cv2.bitwise_and(img, img, mask = mask)
show_masked = cv2.bitwise_and(show_rainbow, show_rainbow, mask = mask)
plt.imshow(show_masked)

hist_masked_vals_red = cv2.calcHist([rainbow], channels = [2], mask = mask, histSize = [256], ranges = [0, 256])
hist_vals_red = cv2.calcHist([rainbow], channels = [2], mask = None, histSize = [256], ranges = [0, 256])

plt.plot(hist_masked_vals_red)
plt.title('Red Histogram for masked rainbow')

plt.plot(hist_vals_red)
plt.title('Red Histogram')

'''Histogram equalization'''
url = '/gorilla.jpg'
gorilla = cv2.imread(data_folder + url, 0)
display(gorilla)
gorilla.shape

hist_vals = cv2.calcHist([gorilla], channels = [0], mask = None, histSize = [256], ranges = [0, 256])
plt.plot(hist_vals)

# equalizing the histogram
eq_gorilla = cv2.equalizeHist(gorilla)
display(eq_gorilla)
# calculating hist vals for equalized gorilla
hist_vals = cv2.calcHist([eq_gorilla], channels = [0], mask = None, histSize = [256], ranges = [0, 256])
plt.plot(hist_vals)

# equalizing for color gorilla
color_gorilla = cv2.imread(url)
show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)
display(show_gorilla)

'''equalize histogram of color image, i.e., increase contrast, first translate to HSV color space'''
hsv = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)
hsv[:, :, 2]
hsv[:, :, 2].min(), hsv[:, :, 2].max()

hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
eq_color_gorilla = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
display(eq_color_gorilla)

'''''''''Video Basics'''''''''

'''using laptop's webcam'''
cap = cv2.VideoCapture(0) # 0 means default camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # dimensions of camera to be displayed
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(filename = 'my_vid.mp4', fourcc = cv2.VideoWriter_fourcc(*'DIVX'), fps = 20,
                         frameSize = (width, height), isColor = True) # DIVX codec for windows

# display the image/vid
while True:
    ret, frame = cap.read()
    # cv2.imshow('vid_frame', frame) '''to display direct vid in real colors'''
    writer.write(frame)
    gray = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2GRAY) # converting frame to gray (use COLOR_BGR2RGB for fun)
    cv2.imshow('vid_frame', gray)

    if cv2.waitKey(2) & 0xFF == 27: # 27 for Esc key or use ord('q') for 'q' key
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

'''using external videos'''

cap = cv2.VideoCapture(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\my_vid.mp4')
if cap.isOpened() == False:
    print('Error: File Not Found Or Incorrect Codec Used')

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        time.sleep(1/20) # remove if not for human watching. divide 1 by fps rate to view as per recorded fps
        cv2.imshow('vid', frame)
        if cv2.waitKey(2) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

'''drawing on live camera'''

cap = cv2.VideoCapture(0) # 0 means default camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # dimensions of camera to be displayed
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# TOP RIGHT CORNER
x = width // 2
y = height // 2

# width n height of rectangle
w = width // 4
h = height // 4

# BOTTOM RIGHT CORNER x+w, y+h

while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, pt1 = (x, y), pt2 = (x + w, y + h), color = (123, 101, 45), thickness = 3)
    cv2.imshow(winname = 'frame', mat = frame)

    if cv2.waitKey(2) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()

'''drawing live shapes on live camera'''

# defining a callback function for rectangle
def draw_rectangle(event, x, y, flags, param):
    global pt1, pt2, topLeft_clicked, bottomRight_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # reset rectangle, checks if rectangle is present
        if topLeft_clicked == True & bottomRight_clicked == True:
            pt1 = (0, 0)
            pt2 = (0, 0)
            topLeft_clicked = False
            bottomRight_clicked = False

        if topLeft_clicked == False:
            pt1 = (x, y)
            topLeft_clicked = True

        elif bottomRight_clicked == False:
            pt2 = (x, y)
            bottomRight_clicked = True

# global variables from mouse action
pt1 = (0, 0)
pt2 = (0, 0)
topLeft_clicked = False
bottomRight_clicked = False

# connect to callback
cap = cv2.VideoCapture(0) # 0 means default camera
cv2.namedWindow('frame')
cv2.setMouseCallback(window_name = 'frame', on_mouse = draw_rectangle)

while True:
    ret, frame = cap.read()
    # drawing frame based on global variables
    if topLeft_clicked:
        cv2.circle(img = frame, center = pt1, radius = 5, color = (101, 191, 201), thickness = -1)
    if topLeft_clicked & bottomRight_clicked:
        cv2.rectangle(img = frame, pt1 = pt1, pt2 = pt2, color = (101, 191, 201), thickness = 3)
    cv2.imshow(winname = 'frame', mat = frame)
    if cv2.waitKey(2) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()

'''Template Matching'''

# scans a larger image for a provided template by sliding the template target image across the larger image
url = data_folder + 'sammy.jpg'
full = cv2.imread(url)
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
plt.imshow(full)
full.shape

url1 = data_folder + 'sammy_face.jpg'
face = cv2.imread(url1)
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
plt.imshow(face)
face.shape

# using eval function
string = 'sum'
eval(string)
# ex
myfunc = eval('sum')
myfunc([2,1,5])

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
           'cv2.TM_SQDIFF_NORMED']

full_copy = full.copy()
res = cv2.matchTemplate(image = full_copy, templ = face, method = eval('cv2.TM_CCOEFF'))
plt.imshow(res)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src = res)

for m in methods:
    method = eval(m)
    # template matching
    res = cv2.matchTemplate(image = full_copy, templ = face, method = method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src = res)
    if method in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
        top_left = min_loc
    else:
        top_left = max_loc
    height, width, channels = face.shape
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(img = full_copy, pt1 = top_left, pt2 = bottom_right, color = (255, 0, 0), thickness = 10)
    # plot n show image
    plt.subplot(121)
    plt.imshow(res)
    plt.title('Heatmap of template matching')

    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('Detection of template')
    plt.suptitle(m)
    plt.show()
    print('\n')
    print('\n')

'''Corner detection'''
'''
a point whose local neighborhood stands in 2 dominant and different edge directions OR
a junction of two edges, where edge is a sudden change in image brightness
Algos:
Harris Corner detection
Shi Tomasi corner detection
'''

url = data_folder + 'flat_chessboard.png'
flat_chess = cv2.imread(url)
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
plt.imshow(flat_chess)
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess, cmap = 'gray')

real_chess = cv2.imread(data_folder + 'real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
plt.imshow(real_chess)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_real_chess, cmap = 'gray')

gray = np.float32(gray_flat_chess)
dst = cv2.cornerHarris(src = gray, blockSize = 2, ksize = 3, k = .04)
dst = cv2.dilate(dst, None)

# threshold may vary with image
flat_chess[dst > .01 * dst.max()] = [255, 0, 0] # RGB channel
plt.imshow(flat_chess)

# on real chess board
gray = np.float32(gray_real_chess)
dst = cv2.cornerHarris(src = gray, blockSize = 2, ksize = 3, k = .04)
dst = cv2.dilate(src = dst, kernel = None)
real_chess[dst > .01 * dst.max()] = [255, 0, 0]
plt.imshow(real_chess)

# Shi - Tomasi edge detection
real_chess = cv2.imread(data_folder + 'real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)

flat_chess = cv2.imread(data_folder + 'flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)

gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

# trying on flat chess board
corners = cv2.goodFeaturesToTrack(image = gray_flat_chess, maxCorners = 64, qualityLevel = .01, minDistance = 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img = flat_chess, center = (x, y), radius = 3, color = (255, 0, 0), thickness = -1)
plt.imshow(flat_chess)

# trying the above on real chess board
cornered = cv2.goodFeaturesToTrack(image = gray_real_chess, maxCorners = 100, qualityLevel = .01, minDistance = 10)
cornered = np.int0(cornered)

for i in cornered:
    x, y = i.ravel()
    cv2.circle(img = real_chess, center = (x, y), radius = 3, color = (255, 0, 0), thickness = -1)
plt.imshow(real_chess)

'''Edge detection'''

# Canny edge detector
url1 = data_folder + 'sammy_face.jpg'
img = cv2.imread(url1)
plt.imshow(img)

edges = cv2.Canny(image = img, threshold1 = 127, threshold2 = 127) # trial n error with threshold values
plt.imshow(edges)
edges = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)
plt.imshow(edges)

med_val = np.median(img)
med_val
# lower threshold to either 0 or 70% of median, whichever is greater
lower = int(max(0, .7 * med_val))
# upper threshold to either 130% of median or max val of 255, whichever is smaller
upper = int(min(255, 1.3 * med_val))

edges = cv2.Canny(image = img, threshold1 = lower, threshold2 = upper + 600)
plt.imshow(edges)

# blurring to avoid noise
blurred = cv2.blur(img, ksize = (5, 5))
edges = cv2.Canny(image = blurred, threshold1 = lower, threshold2 = upper + 50)
plt.imshow(edges)

'''Grid Detection'''

url = data_folder + 'flat_chessboard.png'
flat_chess = cv2.imread(url)
plt.imshow(flat_chess)

found, corners = cv2.findChessboardCorners(image = flat_chess, patternSize = (7, 7))
found
corners
cv2.drawChessboardCorners(image = flat_chess, corners = corners, patternWasFound = found, patternSize = (7, 7))
plt.imshow(flat_chess)

dots = cv2.imread(data_folder + 'dot_grid.png')
plt.imshow(dots)
found, corners = cv2.findCirclesGrid(image = dots, patternSize = (10, 10), flags = cv2.CALIB_CB_SYMMETRIC_GRID)
cv2.drawChessboardCorners(image = dots, corners = corners, patternWasFound = found, patternSize = (10, 10))
plt.imshow(dots)

'''Contour detection'''

img = cv2.imread(data_folder + 'internal_external.png', 0)
img.shape
plt.imshow(img, cmap = 'gray')

contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
type(contours)
type(hierarchy)

# external contour
external_countours = np.zeros(img.shape)
external_countours.shape

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image = external_countours, contours = contours, contourIdx = i, color = (255, 1, 1), thickness = -1)

plt.imshow(external_countours, cmap = 'gray')

# internal contours
internal_countours = np.zeros(img.shape)
internal_countours.shape

for i in range(len(contours)):
    # internal contour
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image = internal_countours, contours = contours, contourIdx = i, color = (255, 1, 1), thickness = -1)

plt.imshow(internal_countours, cmap = 'gray')

'''Feature Matching'''

def display(img):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = 'gray')

reeses = cv2.imread(data_folder + 'reeses_puffs.png', 0)
display(reeses)

cereals = cv2.imread(data_folder + 'many_cereals.jpg', 0)
display(cereals)

## brute force detection with ORB detectors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(image = reeses, mask = None) # returns key points & descriptors
kp2, des2 = orb.detectAndCompute(image = cereals, mask = None)

bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(queryDescriptors = des1, trainDescriptors = des2)
matches
single_match = matches[0] # contains 4 attributes of the object, distance being one
matches = sorted(matches, key = lambda x: x.distance)

reeses_matches = cv2.drawMatches(img1 = reeses, keypoints1 = kp1, img2 = cereals, keypoints2 = kp2,
                                 matches1to2 = matches[:25], outImg = None, matchesMask = None, flags = 2)
display(reeses_matches)

## brute force detection with SIFT descriptors & ratio test (Scale Invariant Feature Transform)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(image = reeses, mask = None)
kp2, des2 = sift.detectAndCompute(image = cereals, mask = None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(queryDescriptors = des1, trainDescriptors = des2, k = 2) # finding k best matches
matches
# applying ratio test to check for best pair of match. less distance the better match
good_matches = []
for match1, match2 in matches:
    # if match1 distance is less than 75% of match2 distance, descriptor for that row may be good match
    if match1.distance < .75 * match2.distance:
        good_matches.append([match1])

good_matches
len(good_matches)
len(matches)

# drawing matches
sift_matches = cv2.drawMatchesKnn(img1 = reeses, keypoints1 = kp1, img2 = cereals, keypoints2 = kp2,
                                  matches1to2 = good_matches, outImg = None, matchesMask = None, flags = 2)
display(sift_matches)

## FLANN descriptor: Fast Library for Approximate Nearest Neighbors. faster than brute force, finds good matches
sift= cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(image = reeses, mask = None)
kp2, des2 = sift.detectAndCompute(image = cereals, mask = None)

FLANN_INDEX_KDTREE = 0
index_params = {'algorithm': 'FLANN_INDEX_KDTREE', 'trees': 5}
search_params = {'checks': 50}
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(queryDescriptors = des1, trainDescriptors = des2, k = 2)
# applying ratio test
good_matches = []
for match1, match2 in matches:
    if match1.distance < .75 * match2.distance:
        good_matches.append([match1])

flann_matches = cv2.drawMatchesKnn(img1 = reeses, keypoints1 = kp1, img2, keypoints2 = cereals, matches1to2 = good_matches, outImg = None, matchesMask = None, flags = 0)
display(flann_matches)

'''Watershed algorithm'''

def display(img, cmap = 'gray'):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = 'gray')

coins = cv2.imread(data_folder + 'pennies.jpg')
display(coins)

'''
steps:
    median blur
    grayscale # order of blur n grayscale doesnt matter
    binary threshold
    find contours
'''
# median blur gets rid of features that are not needed, like faces on coins
sep_blur = cv2.medianBlur(src = coins, ksize = 25)
display(sep_blur)
# grayscale
gray_coins = cv2.cvtColor(src = sep_blur, code = cv2.COLOR_BGR2GRAY)
display(gray_coins)
# binary threshold makes image black & white or separates foreground and background
ret, sep_thresh = cv2.threshold(src = gray_coins, thresh = 150, maxval = 255, type = cv2.THRESH_BINARY_INV)
display(sep_thresh)
# find contours
image, contours, hierarchy = cv2.findContours(image = sep_thresh.copy(), mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_SIMPLE)
for x in range(len(contours)):
    if hierarchy[0][x][3] == -1:
        cv2.drawContours(image = coins, contours = contours, contourIdx = x, color = (201, 123, 86), thickness = 10)
display(coins)

# implementing watershed algorithm
coins = cv2.medianBlur(src = coins, ksize = 35)
display(coins)
gray_coins = cv2.cvtColor(src = coins, code = cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(src = gray_coins, thresh = 127, maxval = 255, type = cv2.THRESH_BINARY_INV)
display(thresh)
# using Otsu's method
ret, thresh = cv2.threshold(src = gray_coins, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
display(thresh)
# noise removal
kernel = np.ones(shape = (3, 3), dtype = np.uint8)
kernel
opening = cv2.morphologyEx(src = thresh, op = cv2.MORPH_OPEN, kernel = kernel, iterations = 2)
display(opening)

sure_bg = cv2.dilate(src = opening, kernel = kernel, iterations = 3)
display(sure_bg)

# distance transformation
dist_trans = cv2.distanceTransform(src = opening, distanceType = cv2.DIST_L2, maskSize = 5)
display(dist_trans)
ret, sure_fg = cv2.threshold(src = dist_trans, thresh = 0.7 * dist_trans.max(), maxval = 255, type = 0)
display(sure_fg)

# finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(src1 = sure_bg, src2 = sure_fg)
display(unknown)

# creating label markers to use as seeds for watershed algorithm to find segments
ret, markers = cv2.connectedComponents(image = sure_fg)
markers
markers = markers + 1 # making sure background as 1 and to mark unknown region as 0
markers[unknown == 255] = 0
display(markers)

markers = cv2.watershed(image = coins, markers = markers)
display(markers)

# find contours
image, contours, hierarchy = cv2.findContours(image = markers.copy(), mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_SIMPLE)
for x in range(len(contours)):
    if hierarchy[0][x][3] == -1:
        cv2.drawContours(image = coins, contours = contours, contourIdx = x, color = (201, 123, 86), thickness = 10)
display(coins)

'''Custom seeds with watershed algorithm'''

road = cv2.imread(data_folder + 'road_image.jpg')
road_copy = road.copy() # or np.copy(road)
plt.imshow(road)
road.shape
marker_image = np.zeros(road.shape[:2], dtype = np.int32)
segments = np.zeros(road.shape, dtype = np.uint8)
marker_image.shape
segments.shape

from matplotlib import cm
cm.tab10(0) # gives RGB and alpha value of tab10 color map
tuple(np.array(cm.tab10(0)[:3]) * 255) # scaling values to be between 0 & 255

def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3]) * 255)

colors = []
for i in range(10):
    colors.append(create_rgb(i))
colors

# global variables
n_markers = 10
current_marker = 1 # color choice
marks_updated = False # markers updated by watershed
# setting up call back function
def mouse_callback(event, x, y, flags, param):
    global marks_updated
    if event == cv2.EVENT_LBUTTONDOWN:
        # markers passed to watershed algo
        cv2.circle(img = marker_image, center = (x, y), radius = 10, color = (current_marker), thickness = -1)
        # user sees on road image
        cv2.circle(road_copy, (x,y), 10, colors[current_marker], -1)
        marks_updated = True
# running while true loop for interactive display segment
cv2.namedWindow('road_image')
cv2.setMouseCallback(window_name = 'road_image', on_mouse = mouse_callback)
while True:
    cv2.imshow('Watershed segmens', segments)
    cv2.imshow('road_image', road_copy)
    # close windows
    k = cv2.waitKey(1)
    if k == 27:
        break
    # clearing all colors, press C
    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(shape = road.shape[:2], dtype = np.int32)
        segments = np.zeros(shape = road.shape, dtype = np.uint8)
    # update color choice
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))
    # update markings
    if marks_updated:
        marker_image_copy = marker_image.copy()
        cv2.watershed(road, marker_image_copy)
        segments = np.zeros(road.shape, dtype = np.uint8)
        for color_ind in range(n_markers):
            # coloring segments
            segments[marker_image_copy == (color_ind)] = colors[color_ind]
cv2.destroyAllWindows()
'''''''''''''''''''''''''''
Face Detection:
Main feature types: Edge features, Line features, Four Rectangle features
Viola Jones calculates summed area table for entire image using integral image.
'''
nadia = cv2.imread(data_folder + 'Nadia_Murad.jpg', 0)
denis = cv2.imread(data_folder + 'Denis_Mukwege.jpg', 0)
solvay = cv2.imread(data_folder + 'solvay_conference.jpg', 0)

plt.imshow(nadia, cmap = 'gray')
plt.imshow(denis, cmap = 'gray')
plt.imshow(solvay, cmap = 'gray')

# face cascade
face_cascade = cv2.CascadeClassifier(data_folder + 'haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(image = face_img)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(img = face_img, pt1 = (x, y), pt2 = (x + w, y + h), color = (255, 255, 255), thickness = 8)
    return plt.imshow(face_img, cmap = 'gray')

detect_face(denis)
detect_face(nadia)
detect_face(solvay)

# adjusted for scale factor and min neighbours
def detect_face_adjusted(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(image = face_img, scaleFactor = 1.2, minNeighbors = 5)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(img = face_img, pt1 = (x, y), pt2 = (x + w, y + h), color = (255, 255, 255), thickness = 8)
    return plt.imshow(face_img, cmap = 'gray')

detect_face_adjusted(solvay)

# eye cascade
eye_cascade = cv2.CascadeClassifier(data_folder + 'haarcascades/haarcascade_eye.xml')

def detect_eyes(img):
    face_img = img.copy()
    eye_rects = eye_cascade.detectMultiScale(image = face_img, scaleFactor = 1.2, minNeighbors = 5)
    for (x, y, w, h) in eye_rects:
        cv2.rectangle(img = face_img, pt1 = (x, y), pt2 = (x + w, y + h), color = (255, 255, 255), thickness = 8)
    return plt.imshow(face_img, cmap = 'gray')

detect_eyes(nadia)
detect_eyes(denis) # doesnt detect due to dark background
detect_eyes(solvay)

# detecting eyes n face on video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read(0)
    frame = detect_face(frame)
    cv2.imshow(winname = 'vid_frame', mat = frame)
    if cv2.waitKey(2) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()

'''Object Detection'''

'''
Techniues:
    Basic:
        Optical Flow
        MeanShift & CamShift
    Advanced:
        Built-in tracking APIs from OpenCV
'''

# Lucas Canarde
corner_track_params = dict(maxCorners = 10, qualityLevel = .3, minDistance = 7, blockSize = 7)
lk_params = dict(winSize = (200,200), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, .03))

# capturing vid from webcam
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(src = prev_frame, code = cv2.COLOR_RGB2GRAY)
# points to track
prev_pts = cv2.goodFeaturesToTrack(image = prev_gray, **corner_track_params, mask = None)
mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2GRAY)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prevImg = prev_gray, nextImg = frame_gray, prevPts = prev_pts, nextPts = None, **lk_params)

    good_new = next_pts[status == 1]
    good_prev = prev_pts[status == 1]

    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        mask = cv2.line(img = mask, pt1 = (x_new, y_new), pt2 = (x_prev, y_prev), color = (0,255,0), thickness = 3)
        frame = cv2.circle(img = frame, center = (x_new, y_new), radius = 8, color = (0,0,255), thickness = -1)
    img = cv2.add(src1 = frame, src2 = mask)
    cv2.imshow('tracking', img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    prev_gray = frame_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()

# Dense Optical FLow
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
prevImg = cv2.cvtColor(src = frame1, code = cv2.COLOR_RGB2GRAY)
hsv_mask = np.zeros_like(frame1)
hsv_mask[:,:,1] = 255

while True:
    ret, frame2 = cap.read()
    nextImg = cv2.cvtColor(src = frame2, code = cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev = prevImg, next = nextImg, flow = None, pyr_scale = .5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)
    mag, angle = cv2.cartToPolar(x = flow[:,:,0], y = flow[:,:,1], angleInDegrees = True)
    hsv_mask[:,:,0] = angle/2
    hsv_mask[:,:,2] = cv2.normalize(src = mag, dst = None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(src = hsv_mask, code = cv2.COLOR_HSV2BGR)
    cv2.imshow('frame', bgr)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
    prevImg = nextImg

cap.release()
cv2.destroyAllWindows()

'''MeanShift & CAMShift (Continuously Adaptive Meanshift) tracking algorithms'''

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

face_cascade = cv2.CascadeClassifier(data_folder + 'haarcascades/haarcascade_frontalface_default.xml')
face_rects = face_cascade.detectMultiScale(frame)
(face_x, face_y, w, h) = tuple(face_rects[0])
track_window = (face_x, face_y, w, h)
roi = frame[face_y:face_y + h, face_x:face_x + w]

hsv_roi = cv2.cvtColor(src = roi, code = cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist(images = [hsv_roi], channels = [0], mask = None, histSize = [180], ranges = [0, 180])
cv2.normalize(src = roi_hist, dst = roi_hist, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) # termination criteria

while True:
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject(images = [hsv], channels = [0], hist = roi_hist, ranges = [0, 180], scale = 1)
        '''
        ret, track_window = cv2.meanShift(probImage = dst, window = track_window, criteria = term_crit)
        x, y, w, h = track_window
        img2 = cv2.rectangle(img = frame, pt1 = (x, y), pt2 = (x + w, y + h), color = (0,0,255), thickness = 5)
        '''
        ret, track_window = cv2.CamShift(probImage = dst, window = track_window, criteria = term_crit)
        pts = cv2.boxPoints(box = ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(img = frame, pts = [pts], isClosed = True, color = (0,0,255), thickness = 5)

        cv2.imshow(winname = 'frame', mat = img2)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

'''Object tracking APIs'''

## Boosting tracker: based on Adaboost algorithm (same algo that HAAR cascade based face detector uses)
## MIL (Multiple Instance Learning) tracker: improved upon Boosting tracker
## KCF (Kernelized Correlation Filter) tracker:
## TLD (Tracking, Learning & Detection) tracker:
## MedianFlow Tracker


'''''''''''''''''''''''''''Transfer Learning'''''''''''''''''''''''''''

'''1st approach Classification'''

# https://www.youtube.com/watch?v=m5RjXjvAAhQ

files = glob.glob(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\big_cats\*')

# with active top layer for readily using for classifications
def predict_pics(my_files):
    preds = []
    for file in my_files:
        file_name = os.path.split(file)
        model = ResNet50(include_top = True, weights = 'imagenet')
        model.layers[-1].get_config()
        img_path = file
        img = load_img(img_path, target_size = (244, 244))
        x = img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        pred = model.predict(x)
        pred_probas = decode_predictions(pred) # use this to check prediction probabilities of other classes
        top_pred = decode_predictions(pred)[0][0]
        image_n_pred = file_name[1], top_pred[1]
        preds.append(image_n_pred)
    return preds

filex = np.random.choice(files, 10)
predict_pics(filex)

# without top layer, can connect with classification or clustering algorithms
model = ResNet50(include_top=False, weights='imagenet')
model.layers[-1].get_config()
img_path = files[98]
img = load_img(img_path, target_size = (244, 244))
x = img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)
x.shape

'''2nd approach Clustering'''

https://github.com/naikshubham/Image-Clustering-Using-Convnets-Transfer-Learning-and-K-Means-/blob/master/image_clustering.py
https://medium.com/@franky07724_57962/using-keras-pre-trained-models-for-feature-extraction-in-image-clustering-a142c6cdf5b1

'''
def get_model(layer = 'fc2'):
    base_model = VGG16(weights = 'imagenet', include_top = True)
    model = Model(inputs = base_model.input, outputs = base_model.get_layer(layer).output)
    return model

'''
def get_model(layer = 'avg_pool'):
    base_model = ResNet50(weights = 'imagenet', include_top = True)
    model = Model(inputs = base_model.input, outputs = base_model.get_layer(layer).output)
    return model

'''
def model():
    # Initializing CNN
    classifier = Sequential()
    # Steps 1: Convolution
    classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    # Step 2: Max Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding 2nd layer & Max pooling
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding 3rd layer & Max pooling
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3: Flattening
    classifier.add(Flatten())
    # Step 4: Full Connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    # Compiling CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_cross_entropy', metrics = ['accuracy'])
    return classifier

# Fitting the CNN model to images
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
trainingSet = train_datagen.flow_from_directory('D:/Programming Tutorials/Machine Learning/Projects/Datasets/big_cats/kitties/', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
testSet = test_datagen.flow_from_directory('D:/Programming Tutorials/Machine Learning/Projects/Datasets/big_cats/kitties/', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
classifier.fit_generator(trainingSet, steps_per_epoch = 8000, epochs = 25, validation_data = testSet, validation_steps = 2000)
'''

def get_files(path_to_files, size):
    fn_imgs = []
    files = [file for file in os.listdir(path_to_files)]
    for file in files:
        img = cv2.resize(cv2.imread(path_to_files + file), size)
        fn_imgs.append([file, img])
    return dict(fn_imgs)

def feature_vector(img_arr, model):
    if img_arr.shape[2] == 1:
        img_arr = img_arr.repeat(3, axis = 2) # (1, 224, 224, 3)
        #arr4d = np.expand_dims(img_arr, axis = 0)
        #arr4d_pp = preprocess_input(arr4d)
    return model.predict(preprocess_input(np.expand_dims(img_arr, axis = 0)))[0,:]

def feature_vectors(imgs_dict, model):
    f_vect = {}
    for fn, img in imgs_dict.items():
        f_vect[fn] = feature_vector(img, model)
    return f_vect

imgs_dict = get_files('D:/Programming Tutorials/Machine Learning/Projects/Datasets/big_cats/kitties/', (224, 224))
model = get_model()
img_feature_vector = feature_vectors(imgs_dict, model) # Feed images through the model and extract feature vectors

# Elbow method to find Optimal K
images = list(img_feature_vector.values())
fns = list(img_feature_vector.keys())
sum_of_squared_distances = []
K = range(1, 10)
for k in K:
    km = KMeans(n_clusters = k)
    km = km.fit(images)
    sum_of_squared_distances.append(km.inertia_)
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++')
kmeans.fit(images)
y_kmeans = kmeans.predict(images)
file_names = list(imgs_dict.keys())

n_clusters = 4
cluster_path = 'D:/Programming Tutorials/Machine Learning/Projects/Datasets/big_cats/kitties/'
path_to_files = 'D:/Programming Tutorials/Machine Learning/Projects/Datasets/big_cats/kitties/'

for c in range(0, n_clusters):
    if not os.path.exists(cluster_path + 'cluster_' + str(c)):
        os.mkdir(cluster_path + 'cluster_' + str(c))

for fn, cluster in zip(file_names, y_kmeans):
    image = cv2.imread(path_to_files + fn)
    cv2.imwrite(cluster_path + 'cluster_' + str(cluster) + '/' + fn, image)


fig = plt.figure(figsize = (14, 14))
cluster_path = 'D:/Programming Tutorials/Machine Learning/Projects/Datasets/big_cats/kitties/cluster_2/'
images = [file for file in os.listdir(cluster_path)]
for count, data in enumerate(images[1:11]):
    y = fig.add_subplot(5, 2, count + 1)
    img = mpimg.imread(cluster_path + data)
    y.imshow(img)
    plt.title('cluster_2')
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)


'''3rd approach'''
# https://towardsdatascience.com/image-clustering-using-transfer-learning-df5862779571

### step1: load the pretrained Resnet50 model and remove last softmax layer from the model
resnet_weights = r'D:\Programming Tutorials\Machine Learning\Projects\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))

# Say not to train first layer (ResNet) model its already trained
my_new_model.layers[0].trainable = False

### step 2: function to load all the images, resize into fixed pixel size (224,224), pass it thru model and extract features
def extract_vector(path):
    resnet_feature_list = []
    for im in glob.glob(path):
        im = cv2.imread(im)
        im = cv2.resize(im, (224, 224))
        img = preprocess_input(np.expand_dims(im.copy(), axis = 0))
        resnet_feature = my_new_model.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())
    return np.array(resnet_feature_list)

files = glob.glob(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\big_cats\*')
file = files[33]
extract_vector(file)

file = r'C:\Users\Srees\Desktop\interview puzzles\2.png'
extract_vector(file)

# not working for multiple files
files = r'C:\Users\Srees\Desktop\interview puzzles\*'
def getting_files(files):
    features = []
    for file in files:
        fs = extract_vector(file)
        features.append(fs)
    return features

### step 3: after extracting features, apply KMeans clustering over datset. decide K, or it can be plotted using loss function vs K to derive it
kmeans = KMeans(n_clusters=2, random_state=0).fit(array)
print(kmeans.labels_)

'''4th approach'''
https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models

### using ResNet50 as a classifier
image = load_img(file, target_size=(224, 224)) # load an image from file
image = img_to_array(image) # convert the image pixels to a numpy array
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # reshape data for the model
image = preprocess_input(image) # prepare the image for the VGG model
model = ResNet50() # load the model
y_pred = model.predict(image) # predict the probability across all output classes
label = decode_predictions(y_pred) # convert the probabilities to class labels
label = label[0][0] # retrieve the most likely result, e.g. highest probability
print('%s (%.2f%%)' % (label[1], label[2]*100))

### using ResNet50 as a feature extraction model
image = load_img(file, target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
model = ResNet50()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
features = model.predict(image)
print(features.shape)
# save to file
dump(features, open('pic.pkl', 'wb'))

### using ResNet50 as a feature extraction model
image = load_img(file, target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
model = ResNet50()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
features = model.predict(image)
print(features.shape)
# save to file
dump(features, open('pic.pkl', 'wb'))

## train new layers of  model without updating weights of VGG16 layers
# load model without classifier layers
model = ResNet50(include_top=False, input_shape=(300, 300, 3))
# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False

## choosing layers to retrain some convolutional layers deep in the model, but none of the layers earlier in the model
model = ResNet50(include_top=False, input_shape=(300, 300, 3)) # load model without classifier layers
# choose layers to make untrainable
model.get_layer('block1_conv1').trainable = False
model.get_layer('block2_conv2').trainable = False


'''5th approach'''

pred = model.predict(x)
IMG_WIDTH = 300
IMG_HEIGHT = 300
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size = 30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size = 30)

restnet = ResNet50(include_top = False, weights = 'imagenet', input_shape = input_shape)
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(inputs = restnet.input, outputs = output)
# to avoid training existing weights of previous layers again
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()

# adding our own fully connected layer and final classifier using sigmoid activation function
model = Sequential()
model.add(restnet)
model.add(Dense(512, activation = 'relu', input_dim = input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(lr = 2e-5), metrics = ['accuracy'])
model.summary()

# run the model
history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 100, validation_data = val_generator,
                              validation_steps = 50, verbose = 1)

'''6th approach using Convolutional Autoencoder'''

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential

(X_train,_),(X_test,_) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

autoencoder = Sequential()
autoencoder.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))
autoencoder.add(MaxPooling2D((2, 2), padding = 'same'))
autoencoder.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
autoencoder.add(MaxPooling2D((2, 2), padding = 'same'))
autoencoder.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
autoencoder.add(MaxPooling2D((2, 2), padding = 'same')) #our encoding
autoencoder.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(32, (3, 3), activation = 'relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation = 'relu', padding = 'same'))

autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')

autoencoder.fit(X_train, X_train, epochs = 50, batch_size = 256, shuffle = True, validation_split = 0.2)


''''''''''''''''''

https://www.kaggle.com/s00100624/digit-image-clustering-via-autoencoder-kmeans?select=test.csv

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns

train = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\cvv\train.csv')
test = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\cvv\test.csv')

# Get X and y for training
X = train.drop(['label'], axis=1)
y = train['label']

# Train, Test and Validate Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=123)

# Reshape and Rescale the images
X_train = X_train.values.reshape(-1,28,28,1) / 255
X_test = X_test.values.reshape(-1,28,28,1) / 255
X_validate = X_validate.values.reshape(-1,28,28,1) / 255

# Build the autoencoder
model = Sequential()
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse")
model.summary()

# Train the model
model.fit(X_train, X_train, epochs=3, batch_size=64, validation_data=(X_validate, X_validate), verbose=1)

# Fitting testing dataset
restored_testing_dataset = model.predict(X_test)

# Observe the reconstructed image quality
plt.figure(figsize=(20,5))
for i in range(10):
    index = y_test.tolist().index(i)
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test[index].reshape((28,28)))
    plt.gray()
    plt.subplot(2, 10, i+11)
    plt.imshow(restored_testing_dataset[index].reshape((28,28)))
    plt.gray()

# Extract the encoder
encoder = K.function([model.layers[0].input], [model.layers[4].output])

# Encode the training set
encoded_images = encoder([X_test])[0].reshape(-1,7*7*7)

# Cluster the training set
kmeans = KMeans(n_clusters=10)
clustered_training_set = kmeans.fit_predict(encoded_images)

# Observe and compare clustering result with actual label using confusion matrix
cm = confusion_matrix(y_test, clustered_training_set)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

# Plot the actual pictures grouped by clustering
fig = plt.figure(figsize=(20,20))
for r in range(10):
    cluster = cm[r].argmax()
    for c, val in enumerate(X_test[clustered_training_set == cluster][0:10]):
        fig.add_subplot(10, 10, 10*r+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: '+str(cluster))
        plt.ylabel('digit: '+str(r))


'''
My learning projects
====================
Web-scraping:
	extract pictures from wikipedia
	extract tables
	extract video titles from youtube
CNN:
	classify cats & dogs
	classify naval ships & merchant ships
	cluster wild & pet animals
	detect & track palms
	detect & track nose
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats
https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras
https://www.geeksforgeeks.org/opencv-python-tutorial/?ref=lbp

Deployment:

Pipeline:

'''
