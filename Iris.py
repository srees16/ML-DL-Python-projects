# Iris

'''Ref: https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset'''
#Check once https://www.kaggle.com/lalitharajesh/iris-dataset-exploratory-data-analysis

import numpy as np
import pandas as pd
import seaborn as sns
#sns.set_palette('husl')
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Iris.csv')
data.head(3)
data.info()
data.describe()
data['Species'].value_counts()


tmp = data.drop('Id', axis = 1)
plotting = sns.pairplot(tmp, hue = 'Species', markers = '+')
plt.show()


graph = sns.violinplot(y = 'Species', x = 'SepalLengthCm', data = data, inner = 'quartile')
plt.show()
graph = sns.violinplot(y = 'Species', x =' SepalWidthCm', data = data, inner = 'quartile')
plt.show()
graph = sns.violinplot(y = 'Species', x = 'PetalLengthCm', data = data, inner = 'quartile')
plt.show()
graph = sns.violinplot(y = 'Species', x = 'PetalWidthCm', data = data, inner = 'quartile')
plt.show()

X = data.drop(['Id', 'Species'], axis = 1)
y = data['Species']

X.shape
y.shape

#finding optimal n neighbours
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

#Logistic regression classifier
logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)
print(metrics.accuracy_score(y, y_pred))
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.4, random_state = 5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#experimenting with different n values
k_range=list(range(1,26))
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
plt.plot(k_range,scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X, y)

# make a prediction for an example of an out-of-sample observation
knn.predict([[6, 3, 4, 2]])