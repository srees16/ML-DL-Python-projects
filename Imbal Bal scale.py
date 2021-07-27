#Imbalanced dataset using ML algos
#Ref: https://elitedatascience.com/imbalanced-classes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.utils import resample
df=pd.read_csv('balance_scale_data.csv',names=['balance','var1','var2','var3','var4'])
df.head()
df['balance'].value_counts()
#Value counts by visualization
pd.value_counts(df['balance']).plot.bar()
plt.title('Balance Scale histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
#Transform into binary classification
df['balance']=[1 if b=='B' else 0 for b in df.balance]
df['balance'].value_counts()#.plot.bar()
#training on imbalanced dataset
X=df.drop('balance',axis=1)
y=df.balance
logRegression1=LogisticRegression().fit(X,y)
yPred1=logRegression1.predict(X)
print(accuracy_score(yPred1,y))
print(np.unique(yPred1))
'''Over sampling minority class'''
df_majority=df[df.balance==0]
df_minority=df[df.balance==1]
df_minority_upsampled=resample(df_minority,replace=True,n_samples=576,random_state=123) #replace sample with replacement,n_samples to match majority class
df_upsampled=pd.concat([df_majority,df_minority_upsampled]) # Combine majority class with upsampled minority class
df_upsampled.balance.value_counts() # Display new class counts
y=df_upsampled.balance #Train model on upsampled dataset
X=df_upsampled.drop('balance',axis=1)
logRegression2=LogisticRegression().fit(X,y)
yPred2=logRegression2.predict(X)
print(np.unique(yPred2))
print(accuracy_score(y,yPred2))
'''Under sampling minority class'''
df_majority=df[df.balance==0]
df_minority=df[df.balance==1]
df_majority_downsampled=resample(df_majority,replace=False,n_samples=49,random_state=123)
df_downsampled=pd.concat([df_majority_downsampled,df_minority])
df_downsampled.balance.value_counts()
y=df_downsampled.balance #Train model on upsampled dataset
X=df_downsampled.drop('balance',axis=1)
logRegression3=LogisticRegression().fit(X,y)
yPred3=logRegression3.predict(X)
print(np.unique(yPred3))
print(accuracy_score(y,yPred3))
'''Predict class probabilities'''
prob_y_2=logRegression3.predict_proba(X)
#AUROC of model trained on downsampled data
prob_y_2=[p[1] for p in prob_y_2] #Keep only the positive class
prob_y_2[:5]
print(roc_auc_score(y,prob_y_2))
#AUROC of model trained on imbalanced data
prob_y_0=logRegression1.predict_proba(X)
prob_y_0=[p[1] for p in prob_y_0]
print(roc_auc_score(y,prob_y_0))
'''SVM based algorithm for imbalanced data'''
y=df.balance
X=df.drop('balance',axis=1)
#Train model
clf_3=SVC(kernel='linear',class_weight='balanced',probability=True)
clf_3.fit(X,y)
pred_y_3=clf_3.predict(X) #Predict on training set
print(np.unique(pred_y_3))
print(accuracy_score(y, pred_y_3))
prob_y_3=clf_3.predict_proba(X) #ROC score
prob_y_3=[p[1] for p in prob_y_3]
print(roc_auc_score(y,prob_y_3))
'''Tree based algorithm for imbalanced data'''
y=df.balance
X=df.drop('balance',axis=1)
clf_4=RandomForestClassifier()
clf_4.fit(X,y)
pred_y_4=clf_4.predict(X)
print(np.unique(pred_y_4 ))
print(accuracy_score(y,pred_y_4))
prob_y_4=clf_4.predict_proba(X)
prob_y_4=[p[1] for p in prob_y_4]
print(roc_auc_score(y,prob_y_4))