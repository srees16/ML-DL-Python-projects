#Medication Compliance forcasting

import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Importing the dataset
train_dataset = pd.read_csv('Training Data.csv')
train_dataset.head()
train_dataset.info() #to check for the data type of variable and for any null values
train_dataset.describe()

#Visualization to check missing values
sb.heatmap(train_dataset.isnull(), cbar = True, yticklabels = False, cmap = 'viridis')
x = train_dataset.iloc[:,1:10].values
y = train_dataset.iloc[:,10].values

#correlation matrix based on Alcoholism
k = 10 #number of variables for heatmap
cols = train_dataset.corr().nlargest(k, 'Alcoholism')['Alcoholism'].index
correlmatrix = train_dataset[cols].corr()
plot.figure(figsize = (10, 6))
sb.heatmap(correlmatrix, annot = True, cmap = 'viridis')

#Encoding the categorical variables
labelEncoder = LabelEncoder()
x[:, 1] = labelEncoder.fit_transform(x[:, 1])
hotEncoder = OneHotEncoder(categorical_features = [1])
x = hotEncoder.fit_transform(x).toarray()
y = labelEncoder.fit_transform(y)

#Feature Scaling
scale = StandardScaler()
x = scale.fit_transform(x)

#Performing PCA since features are more and may over fit the data
pca = PCA(n_components = 3)
x = pca.fit_transform(x)

#Splitting the data into test and training set
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.20, random_state = 0)

#Fitting the data to Decision Tree model
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 6, max_depth = 10, min_samples_split = 2, splitter = 'best')
classifier.fit(xTrain, yTrain)
yPred = classifier.predict(xTest)

#Finding best hyper parameters for the Decision Tree model using GridSearch
parameters = {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random'], 'max_depth' : [1, 2, 3, 5, 8, 10], 'min_samples_split' : [2, 4, 6]}
gridSearch = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', n_jobs = -1)
gridSearch = gridSearch.fit(xTrain, yTrain)
best_accuracy = gridSearch.best_score_
best_parameters = gridSearch.best_params_

#Checking for optimal score that can be reached with K Fold cross validation
accuracy = cross_val_score(estimator = classifier, X = xTrain, y = yTrain, cv = 10)
accuracy.mean() #best accuracy
accuracy.std() #to verify variance

#Confusion matrix for Precision & Recall for +ve and -ve cases
cm = confusion_matrix(yTest, yPred)
print(accuracy_score(yTest, yPred))
precision_yes = cm[0][0] / (cm[0][0] + cm[0][1])
recall_yes = cm[0][0] / (cm[0][0] + cm[1][0])
precision_no = cm[1][1] / (cm[1][1] + cm[1][0])
recall_no = cm[1][1] / (cm[1][1] + cm[0][1])
print('Precision for YES-->', precision_yes, 'Recall for YES-->', recall_yes, 'Precision for NO-->', precision_no, 'Recall for NO-->', recall_no)

#Evaluating the classifier to the test dataset
test_data = pd.read_csv('Test Data.csv')
X_test_data = test_data.iloc[:, 1:10].values

#Encoding the categorical variables
labelEncoder = LabelEncoder()
X_test_data[:,1] = labelEncoder.fit_transform(X_test_data[:, 1])
hotEncoder = OneHotEncoder(categorical_features = [1])
X_test_data = hotEncoder.fit_transform(X_test_data).toarray()

#Feature Scaling
scale = StandardScaler()
X_test_data = scale.fit_transform(X_test_data)

#Performing PCA since features are more and may over fit the data
pca = PCA(n_components = 3)
X_test_data = pca.fit_transform(X_test_data)

#Fitting the above classifier to the test dataset
Y_test_pred = classifier.predict(X_test_data)

#Show the inputs and predicted outputs in the test dataset
Y_test_pred = (Y_test_pred > 0.5)
for i in range(len(X_test_data)):
    print('Adherence = %s, Probability Score = %s' %(Y_test_pred[i], X_test_data[i]))
    # Fill missing values with mean column values. Try this to update test data set
    X_test_data.fillna(X_test_data.mean(), inplace = True)
    # Count the number of NaN values in each column
    print(X_test_data.isnull().sum())
    #X_test_data['Adherence','Probability']=Y_test_pred[i],X_test_data[i]