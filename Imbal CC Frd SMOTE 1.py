# Credit Card Fraud
'''Ref: https://www.kaggle.com/qianchao/smote-with-imbalance-data'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
import itertools
%matplotlib inline

data = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\creditcard fraud detect.csv')
data.head(3)

pd.value_counts(data['Class']).plot.bar() #1st way
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')

data['Class'].value_counts().plot.bar() #2nd way

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))

data = data.drop(['Time','Amount'], axis = 1)
data.head(2)

X = np.array(data.iloc[:, data.columns != 'Class'])
y = np.array(data.iloc[:, data.columns == 'Class'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print('Shape of X_train:',X_train.shape)
print('Shape of y_train dataset:',y_train.shape)
print('Shape of X_test dataset:',X_test.shape)
print('Shape of y_test dataset:',y_test.shape)
print('Before OverSampling, count of label "1":{}'.format(sum(y_train == 1)))
print('Before OverSampling, count of label "0":{} \n'.format(sum(y_train == 0)))

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
print('After OverSampling, shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, shape of train_y: {}'.format(y_train_res.shape))
print("After OverSampling, count of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, count of label '0': {}".format(sum(y_train_res == 0)))

parameters = {'C' : np.linspace(1, 10, 10)}
lr = LogisticRegression()

clf = GridSearchCV(lr, parameters, cv = 5, verbose = 1, n_jobs = 3)
clf.fit(X_train_res, y_train_res.ravel())
clf.best_params_
lr1 = LogisticRegression(C = 4, penalty = 'l1', verbose = 5)
lr1.fit(X_train_res, y_train_res.ravel())

#Function to print and plots confusion matrix.Normalization can be applied by setting `normalize=True`
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(cm,interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm=cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis] #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')
    thresh=cm.max()/2. #print(cm)
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_train_pre=lr1.predict(X_train)
cnf_matrix_tra=confusion_matrix(y_train,y_train_pre)
print('Recall metric in the train dataset: {}%'.format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra,classes=class_names,title='Confusion matrix')
plt.show()
y_pre=lr1.predict(X_test)
cnf_matrix=confusion_matrix(y_test,y_pre)
print('Recall metric in the testing dataset: {}%'.format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
# print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))

# Plot non-normalized confusion matrix
class_names=[0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_names,title='Confusion matrix')
plt.show()
tmp=lr1.fit(X_train_res,y_train_res.ravel())
y_pred_sample_score=tmp.decision_function(X_test)
fpr,tpr,thresholds=roc_curve(y_test,y_pred_sample_score)
roc_auc=auc(fpr,tpr)

#Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr,'b',label='AUC= %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print(roc_auc)