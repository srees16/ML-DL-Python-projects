#PPP done
#Imbal bank SMOTE AND NEAR MISS for IMBALANCED DATASETS
#Ref: https://github.com/saeed-abdul-rahim/tutorials/blob/master/imblearn/SMOTENearMiss.py
#Ref: https://medium.com/@saeedAR/smote-and-near-miss-in-python-machine-learning-in-imbalanced-datasets-b7976d9a7a79
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plot
bank = pd.read_csv('bank marketing.csv')
bank.head(3)
bank.describe()
bank.info()
bank.shape
bank.columns
#Value counts by visualization
pd.value_counts(bank['deposit']).plot.bar()
plot.title('Deposit class histogram')
plot.xlabel('Class')
plot.ylabel('Frequency')
#mapping the categories to numeric values
bank['default'] = bank['default'].map({'no':0, 'yes':1})
bank['housing'] = bank['housing'].map({'no':0, 'yes':1})
bank['loan'] = bank['loan'].map({'no':0, 'yes':1})
bank['deposit'] = bank['deposit'].map({'no':0, 'yes':1})
bank.education = bank.education.map({'primary':0, 'secondary':1, 'tertiary':2})
bank.month = pd.to_datetime(bank.month, format = '%b').dt.month
bank.head(3)
bank.isnull().sum()
bank.drop(['poutcome', 'contact'],axis = 1, inplace = True)
bank.dropna(inplace = True)
bank = pd.get_dummies(bank, drop_first = True)
bank.deposit.value_counts()#.plot.bar()
X = bank.drop('deposit', axis = 1)
y = bank.deposit
X_train, X_test, y_train, y_test = tts(X, y, random_state = 1, stratify = y)
#Fitting imbalanced dataset to classifier
y_train.value_counts()
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
recall_score(y_test, y_pred)
'''
Using SMOTE
'''
X_train, X_test, y_train, y_test = tts(X, y, random_state = 1, stratify = y)
smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)
np.bincount(y_train)

# using SMOTE data to fit to regressor
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
recall_score(y_test, y_pred)
'''
Using NearMiss
'''
X_train, X_test, y_train, y_test = tts(X, y, random_state = 1, stratify = y)
nr = NearMiss()
X_train, y_train = nr.fit_sample(X_train, y_train)
np.bincount(y_train)

# using NearMiss data to fit to regressor
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
recall_score(y_test, y_pred)