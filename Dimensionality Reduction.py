#Dimensionality Reduction

'''Ref: https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Train_UWu5bXk.txt')

#checking the percentage of missing values in each variable
train.isnull().sum() / len(train) * 100

'''Using Missing value ratio'''

#saving missing values in a variable
missing = train.isnull().sum() / len(train) * 100

#saving column names in a variable and adding them to a list by setting a threshold and features above that can be removed from feature set
features = train.columns
feature = []
for i in range(0, 12):
    if missing[i] <= 20: #features to be used are stored in 'feature' that contains features where the missing values are less than 20%
        feature.append(features[i])

#Imputing missing values
train['Item_Weight'].fillna(train['Item_Weight'].median, inplace = True)
train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace = True)

#check if all the missing values are filled
train.isnull().sum()/len(train) * 100

'''Using Low Variance Filter'''

#variables with a low variance will not affect the target variable
train.var() #calculate the variance of all the numerical variables and drop columns that has very less variance as compared to the other features
numeric = train[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']]
var = numeric.var()
numeric = numeric.columns
variable = []
for i in range(0, len(var)):
    if var[i] >= 10:   #setting the threshold as 10%
       variable.append(numeric[i + 1]) #gives us the list of variables that have a variance greater than 10

'''Using High Correlation filter'''

df = train.drop('Item_Outlet_Sales', 1)
df.corr() #No variables with high correlation

'''Using Random Forest'''

#Used with in-built feature_importance function.Convert the data into numeric by OneHotEncoding, it takes only numeric inputs
from sklearn.ensemble import RandomForestRegressor
df = df.drop(['Item_Identifier', 'Outlet_Identifier'], axis = 1)
model = RandomForestRegressor(random_state = 1, max_depth = 10)
df = pd.get_dummies(df)
model.fit(df, train.Item_Outlet_Sales)

#After fitting the model,plot the feature importance graph
features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-9:] # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color = 'b', align = 'center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

#Alernate way using the SelectFromModel of sklearn. Selects the features based on their weights
from sklearn.feature_selection import SelectFromModel
feature = SelectFromModel(model)
fit = feature.fit_transform(df, train.Item_Outlet_Sales)

'''Using Factor Analysis'''
from glob import glob
import cv2
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

images = [cv2.imread(file) for file in glob('train/*.png')]

images = np.array(images)
images.shape

image = []
for i in range(0, 60000):
    img = images[i].flatten()
    image.append(img)
image = np.array(image)

train = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Train_UWu5bXk.txt')
feat_cols = ['pixel' + str(i) for i in range(image.shape[1])]
df = pd.DataFrame(image, columns = feat_cols)
df['label'] = train['label']

FA = FactorAnalysis(n_components = 3).fit_transform(df[feat_cols].values)

plt.figure(figsize = (12, 8))
plt.title('Factor Analysis Components')
plt.scatter(FA[:, 0], FA[:, 1])
plt.scatter(FA[:, 1], FA[:, 2])
plt.scatter(FA[:, 2], FA[:, 0])


'''Using Principal Component Analysis'''

#visualizing how much variance has been explained using these n components
rndperm = np.random.permutation(df.shape[0])
plt.gray()
fig = plt.figure(figsize=(20,10))
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1)
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28*3)).astype(float))

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
variance = pca.explained_variance_ratio_
plt.plot(range(4), variance)
plt.plot(range(4), np.cumsum(variance))
plt.title("Component-wise and Cumulative Explained Variance")


'''Using t-SNE'''

from sklearn.manifold import TSNE 
tsne = TSNE(n_components = 3, n_iter = 300).fit_transform(df[features][:6000].values)

plt.figure(figsize = (12, 8))
plt.title('t-SNE components')
plt.scatter(tsne[:,0], tsne[:,1])
plt.scatter(tsne[:,1], tsne[:,2])
plt.scatter(tsne[:,2], tsne[:,0])