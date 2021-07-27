# EDA Wine

'''Ref- http://nbviewer.jupyter.org/github/PBPatil/Exploratory_Data_Analysis-Wine_Quality_Dataset/blob/master/winequality_white.ipynb
Ref- https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sb
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier as xgb, plot_importance

# Import dataset
dataset = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\winequality-white.csv', sep = ';')

# Finding the null/Nan values in the columns 1st way
dataset.isnull().sum()#/len(df)*100 #checking for missing values in each feature column, unblock for % count

# Finding the null/Nan values in the columns 2nd way
for i in dataset.columns:
    print("Null values in {} = {}".format(i, dataset[i].isnull().sum()))

# Visualizing the null/Nan values in the columns 1st way. If there were,it'll show different shade on color background
sb.heatmap(dataset.isnull(), cbar = True, yticklabels = False, cmap = 'viridis')

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
sb.heatmap(dataset.corr(method = 'pearson'), cmap = 'viridis', annot = True)

'''2nd way of multi correlation between features as heatmap that gives only 1 diagonal'''
mask = np.array(dataset.corr())
mask[np.tril_indices_from(mask)] = False
fig,ax = plot.subplots(figsize = (10, 6))
sb.heatmap(dataset.corr(), mask = mask, vmax = .8, square = True, annot = True, cmap = 'viridis')

'''3rd way of multi correlation matrix heatmap w.r.t specific feature "Quality"'''
# number of features for heatmap
k = 12
#gives correlation using Pearson by default
cols = dataset.corr(method = 'pearson').nlargest(k, 'quality')['quality'].index
corr_matrix = dataset[cols].corr()
plot.figure(figsize = (10, 6))
sb.heatmap(corr_matrix, cmap = 'viridis',annot = True)

#Custom correlogram between each pair of features w.r.t output
sb.pairplot(dataset, hue = 'quality')

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
number_of_columns = 12
number_of_rows = len(features) - 1 / number_of_columns
plot.figure(figsize = (number_of_columns, 5 * number_of_rows))
for i in range(0, len(features)):
    plot.subplot(number_of_rows + 1, number_of_columns, i + 1)
    sb.set_style('whitegrid')
    sb.boxplot(dataset[features[i]], color = 'green', orient = 'v')
    plot.tight_layout()

# To check distribution-Skewness
plot.figure(figsize = (2 * number_of_columns, 5 * number_of_rows))
for i in range(0, len(features)):
    plot.subplot(number_of_rows + 1, number_of_columns, i + 1)
    sb.distplot(dataset[features[i]], kde = True)

# Visualization with barplot and normal distribution plot
for i,features in enumerate(list(dataset.columns)[:-1]):
    fg = sb.FacetGrid(dataset, hue = 'quality', height = 12)
    fg.map(sb.distplot, features).add_legend()
dataset.alcohol[dataset.quality == 0].median() # to verify what the graphs above showed
sb.boxplot(data = dataset, x = 'quality', y = 'alcohol', color = 'g') #to know the impact of the no of positive alcohol nodes detected and the patient status

# select the feature columns & the target column
x = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values

# Splitting dataset
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size = 0.8, random_state = 10)

'''Using Random Forest for important features'''

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