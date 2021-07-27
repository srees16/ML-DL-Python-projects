# EDA Breast Cancer
'''Ref: https://github.com/bacemtayeb/EDA/blob/master/Haberman.ipynb'''

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plot
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier as xgb,plot_importance

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
sb.heatmap(dataset.corr(method = 'pearson'), annot = True)

#2nd way for multi correlation between features as heatmap that gives only 1 diagonal
mask = np.array(dataset.corr())
mask[np.tril_indices_from(mask)] = False
fig, ax = plot.subplots(figsize = (4, 3))
sb.heatmap(dataset.corr(), mask = mask, vmax = .8, square = True, annot = True, cmap = 'viridis')

# Custom correlation between each pair of features w.r.t output
sb.pairplot(dataset, hue = 'Surv_status')

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
    #sb.set_style('whitegrid')
    sb.boxplot(dataset[features[i]], color = 'blue', orient = 'v')
    plot.tight_layout()

# To check distribution-Skewness
plot.figure(figsize = (2 * number_of_columns, 5 * number_of_rows))
for i in range(0, number_of_columns):
    plot.subplot(number_of_rows + 1, number_of_columns, i + 1)
    sb.distplot(dataset[features[i]], kde = True)

# Visualization with barplot and normal distribution plot
for i, features in enumerate(list(dataset.columns)[:-1]):
    fg = sb.FacetGrid(dataset, hue = 'Surv_status', height = 5)
    fg.map(sb.distplot, features).add_legend()

# to verify what the graphs above showed
dataset.Operation_Year[dataset.Surv_status == 1].median()

# to know the impact of the number of positive axillary nodes detected and the patient status
sb.boxplot(data = dataset, x = 'Surv_status', y = 'Axil_nodes', color = 'g')

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
sb.set(style = "white", color_codes = True) #Not working, need to probe~~~~~
sb.jointplot(x = x["Operation_Year"], y = y, kind = 'kde', color = "skyblue")