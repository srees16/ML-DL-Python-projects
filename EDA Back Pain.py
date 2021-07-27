# EDA Back Pain

'''Ref: https://towardsdatascience.com/an-exploratory-data-analysis-on-lower-back-pain-6283d0b0123
https://www.kaggle.com/nasirislamsujan/exploratory-data-analysis-lower-back-pain?scriptVersionId=5589885'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sb
sb.set()
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, plot_importance
from sklearn import model_selection

#importing dataset
dataset = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Dataset_spine.csv')
dataset.head(3)

#unnecessary column
dataset.iloc[:, -1].head()

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
sb.heatmap(dataset.corr(), annot = True, cmap = 'viridis')

# 2nd way to find correlation between features as heatmap that gives only 1 diagonal
mask = np.array(dataset.corr())
mask[np.tril_indices_from(mask)] = False
fig, ax = plot.subplots(figsize = (10, 8))
sb.heatmap(dataset.corr(), vmax = .8, square = True, annot = True, cmap = 'viridis', mask = mask)

# 3rd way of custom correlation between each pair of features w.r.t output
sb.pairplot(dataset, hue = 'class')

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
    sb.set_style('whitegrid')
    sb.boxplot(dataset[feature_space[i]], color = 'green', orient = 'v')
    plot.tight_layout()

# To check distribution-Skewness
plot.figure(figsize = (2 * number_of_columns, 5 * number_of_rows))
for k in range(0, len(feature_space)):
    plot.subplot(number_of_rows + 1, number_of_columns, k + 1)
    sb.distplot(dataset[feature_space[k]], kde = True)

# Visualization with barplot and normal distribution plot
for j, features in enumerate(list(dataset.columns)[:-1]):
    fg = sb.FacetGrid(dataset, hue = 'class', height = 5)
    fg.map(sb.distplot, features).add_legend()
dataset.pelvic_slope[dataset.scoliosis_slope == 1].median()
sb.boxplot(data = dataset, x = 'class', y = 'pelvic_slope', color = 'g')

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
    sb.set_style('whitegrid')
    sb.boxplot(dataset[feature_space[i]], color = 'green', orient = 'v')
    plot.tight_layout()

# To check distribution-Skewness
plot.figure(figsize = (2 * number_of_columns, 5 * number_of_rows))
for k in range(0, len(feature_space)):
    plot.subplot(number_of_rows + 1, number_of_columns, k + 1)
    sb.distplot(dataset[feature_space[k]], kde = True)

# Visualization with barplot and normal distribution plot
for j, features in enumerate(list(dataset.columns)[:-1]):
    fg = sb.FacetGrid(dataset, hue = 'class', height = 5)
    fg.map(sb.distplot, features).add_legend()
dataset.pelvic_slope[dataset.scoliosis_slope == 1].median()
sb.boxplot(data = dataset, x = 'class', y = 'pelvic_slope', color = 'g')

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
models = []
names = ['Logistic Regression', 'Random Forest', 'ANN', 'Gaussian NB', 'DecisionTree Classifier',
         'XGBClassifier']
models.append((LogisticRegression(solver = 'liblinear')))
models.append(RandomForestClassifier(n_estimators = 100))
models.append((MLPClassifier()))
models.append((GaussianNB()))
models.append((DecisionTreeClassifier()))
models.append((XGBClassifier()))
models

kfold = model_selection.KFold(n_splits = 5, random_state = 7)

for i in range(0, len(models)):
    cv_result = model_selection.cross_val_score(models[i], X_train, y_train, cv = kfold, scoring = 'accuracy')
    score = models[i].fit(X_train, y_train)
    prediction = models[i].predict(X_test)
    acc_score = accuracy_score(y_test, prediction)
    print ('-' * 40)
    print ('{0}: {1}'.format(names[i], acc_score))

'''Fitting the dataset to the appropriate ML model to predict & compare with test data as per the accuracy above'''

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
sb.set(style = 'white', color_codes = True)
sb.jointplot(x = X['pelvic_slope'], y = y, kind = 'kde', color = 'skyblue')

'''Using Random Forest for important features''' # Taken from EDA Wine, make required changes

# Using in-built feature_importance function.Convert the data into numeric by OneHotEncoding
model = RandomForestRegressor(random_state = 1, max_depth = 10)
dataset = pd.get_dummies(dataset)
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
sb.set(style = "white", color_codes = True) #Not working, need to probe~~~~~
sb.jointplot(x = x['alcohol'], y = y, kind = 'kde', color = 'skyblue')