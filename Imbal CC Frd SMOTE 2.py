# Credit Card Fraud
'''Ref: https://www.kaggle.com/gargmanish/how-to-handle-imbalance-data-study-in-detail/notebook'''

import pandas as pd # to import csv and for data manipulation
import numpy as np # for linear algebra
import matplotlib.pyplot as plt
import seaborn as sns # for intractve graphs
import datetime # to dela with date and time
from sklearn.preprocessing import MinMaxScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split as tts, KFold # to split the data
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
from imblearn.over_sampling import SMOTE
%matplotlib inline

data = pd.read_csv('creditcard fraud.csv',header = 0)
data.head(2)
data.describe()
data.info()
sns.countplot("Class", data = data)

# Check the number in Percentage
Count_Normal_transacation = len(data[data["Class"] == 0]) # normal transaction are repersented by 0
Count_Fraud_transacation = len(data[data["Class"] == 1]) # fraud by 1
Percentage_of_Normal_transacation = Count_Normal_transacation/(Count_Normal_transacation + Count_Fraud_transacation)
print("percentage of normal transacation is",Percentage_of_Normal_transacation * 100)
Percentage_of_Fraud_transacation= Count_Fraud_transacation/(Count_Normal_transacation + Count_Fraud_transacation)
print("percentage of fraud transacation",Percentage_of_Fraud_transacation * 100)

# No of valid transcation and fraud transcations
Fraud_transacation = data[data["Class"] == 1]
Normal_transacation = data[data["Class"] == 0]
plt.figure(figsize = (10, 6))
plt.subplot(121)
Fraud_transacation.Amount.plot.hist(title = "Fraud Transacation")
plt.subplot(122)
Normal_transacation.Amount.plot.hist(title = "Normal Transaction")

# distribution for Normal transction is not clear seems all transaction are less than 2.5 K
Fraud_transacation = data[data["Class"] == 1]
Normal_transacation = data[data["Class"] == 0]
plt.figure(figsize = (10, 6))
plt.subplot(121)
Fraud_transacation[Fraud_transacation["Amount"]<= 2500].Amount.plot.hist(title="Fraud Tranascation")
plt.subplot(122)
Normal_transacation[Normal_transacation["Amount"]<=2500].Amount.plot.hist(title="Normal Transaction")

# index of fraud cases
fraud_indices=np.array(data[data.Class==1].index)
normal_indices=np.array(data[data.Class==0].index)

# function to undersample data with different proportion i.e, normal classes of data
def undersample(normal_indices,fraud_indices,times):#times denote the normal data = times*fraud data
    Normal_indices_undersample=np.array(np.random.choice(normal_indices,(times*Count_Fraud_transacation),replace=False))
    undersample_data=np.concatenate([fraud_indices,Normal_indices_undersample])
    undersample_data=data.iloc[undersample_data,:]
    print("the normal transacation proportion is :",len(undersample_data[undersample_data.Class==0])/len(undersample_data[undersample_data.Class]))
    print("the fraud transacation proportion is :",len(undersample_data[undersample_data.Class==1])/len(undersample_data[undersample_data.Class]))
    print("total number of record in resampled data is:",len(undersample_data[undersample_data.Class]))
    return(undersample_data)

# first make a model function for modeling with confusion matrix
def model(model,features_train,features_test,labels_train,labels_test):
    clf=model
    clf.fit(features_train,labels_train.values.ravel())
    pred=clf.predict(features_test)
    cnf_matrix=confusion_matrix(labels_test,pred)
    print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    fig= plt.figure(figsize=(6,3))# to plot the graph
    print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud
    print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal
    print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud
    print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal
    sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report----------")
    print(classification_report(labels_test,pred))

# Preparing data for training and testing as we are going to use different data
def data_preparation(x):
    #again and again so make a function
    x_features=x.iloc[:,x.columns!="Class"]
    x_labels=x.iloc[:,x.columns=="Class"]
    x_features_train,x_features_test,x_labels_train,x_labels_test=tts(x_features,x_labels,test_size=0.3)
    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)

# Standardize amount column
data["Normalized Amount"] = MinMaxScaler().fit_transform(data["Amount"].values.reshape(-1,1))
data.head(2)
'''
data.drop(["Time","Amount"],axis=1,inplace=True)
'''
# using SMOTE for oversampling
os = SMOTE(random_state = 3)

# Splitting data into training and test set and use data preparation method
data_train_X, data_test_X, data_train_y, data_test_y = data_preparation(data)
columns = data_train_X.columns
os_data_X, os_data_y = os.fit_sample(data_train_X, data_train_y)
os_data_X = pd.DataFrame(data = os_data_X, columns = columns )
os_data_y = pd.DataFrame(data = os_data_y, columns=["Class"])

# we can Check the numbers of our data
print("length of oversampled data is ", len(os_data_X))
print("Number of normal transcation in oversampled data", len(os_data_y[os_data_y["Class"] == 0]))
print("No.of fraud transcation", len(os_data_y[os_data_y["Class"] == 1]))
print("Proportion of Normal data in oversampled data is ", len(os_data_y[os_data_y["Class"] == 0])/len(os_data_X))
print("Proportion of fraud data in oversampled data is ", len(os_data_y[os_data_y["Class"] == 1])/len(os_data_X))

os_data_X["Normalized Amount"] = StandardScaler().fit_transform(os_data_X["Amount"].values.reshape(-1, 1))
os_data_X.drop(["Time","Amount"], axis = 1, inplace = True)
data_test_X["Normalized Amount"] = StandardScaler().fit_transform(data_test_X["Amount"].values.reshape(-1, 1))
data_test_X.drop(["Time","Amount"], axis = 1, inplace = True)
clf = RandomForestClassifier(n_estimators = 100)

# train data using oversampled data and predict for the test data
model(clf, os_data_X, data_test_X, os_data_y, data_test_y)