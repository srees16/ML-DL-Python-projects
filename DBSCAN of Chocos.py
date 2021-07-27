# DBSCAN of chocos
'''https://www.kaggle.com/tejasrinivas/chocolate-ratings-outlier-analysis-with-dbscan'''

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

dataset = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\flavors_of_cacao.csv')

missing_values = pd.DataFrame(dataset.isnull().sum(), columns = ['Number of missing values'])
missing_values

dataset.head()

plt.figure(figsize = (8, 6))
sns.distplot(dataset['Rating'], bins = 5, color = 'brown')

dataset['Cocoa % as num'] = dataset['Cocoa\nPercent'].apply(lambda x : x.split('%')[0])
dataset['Cocoa % as num'] = dataset['Cocoa % as num'].astype(float)

plt.figure(figsize = (12, 6))
sns.distplot(dataset['Cocoa % as num'], bins = 20, color = 'Brown')

dataset['Review\nDate'] = dataset['Review\nDate'].astype(str)
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'Review\nDate', y = 'Rating', data = dataset)

''' Figures'''
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, figsize = (12, 15))

a = dataset.groupby(['Company\nLocation'])['Rating'].mean()
a = a.sort_values(ascending = False)

b = dataset.groupby(['Company\nLocation'])['Rating'].median()
b = b.sort_values(ascending = False)

a = pd.DataFrame(a)
b = pd.DataFrame(b)

Ratings_by_location = a.join(b, how='left', lsuffix = '_mean', rsuffix = '_median')
Ratings_by_location['Mean-Median'] = Ratings_by_location['Rating_mean']-Ratings_by_location['Rating_median']

Rating_difference = sns.barplot(x = Ratings_by_location.index,y=Ratings_by_location['Mean-Median'], ax = ax3)
Rating_difference.set_xticklabels(labels = Ratings_by_location.index, rotation =90)
Rating_difference.set_ylabel("Mean-Median of ratings")

# plt.figure(figsize=(12,6))
ratings_mean = sns.barplot(x = Ratings_by_location.index, y = Ratings_by_location['Rating_mean'], ax = ax1)
ratings_mean.set_xticklabels(labels = Ratings_by_location.index, rotation = 90)
ratings_mean.set_ylabel("Mean of Ratings")


# plt.figure(figsize=(12,6))
ratings_median = sns.barplot(x = Ratings_by_location.index, y = Ratings_by_location['Rating_median'], ax = ax2)
ratings_median.set_xticklabels(labels = Ratings_by_location.index, rotation = 90)
ratings_median.set_ylabel("Median of ratings")

plt.tight_layout()

df1 = dataset[['Cocoa % as num','Rating','Review\nDate']]

# non_numerical_columns = ['Review\nDate','Bean\nType', 'Broad Bean\nOrigin','Company\nLocation']
non_numerical_columns = ['Review\nDate']

for i in non_numerical_columns:
    x1 = pd.get_dummies(df1[i])
    df1 = df1.join(x1, lsuffix = '_l', rsuffix = '_r')
    df1.drop(i, axis = 1, inplace = True)

# Standardizing data is imp for clustering techniques to avoid a feature biasing the results of clustering

from sklearn.cluster import DBSCAN
# from sklearn import metrics
from sklearn.preprocessing import StandardScaler
df_num = StandardScaler().fit_transform(df1)
A = []
B = []
C = []

for i in np.linspace(0.1, 5, 50):
    db = DBSCAN(eps = i, min_samples = 10).fit(df_num)

    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    sum = 0
    for t in labels:
        if t == -1: 
            sum = sum + 1
    C.append(sum)

    A.append(i)
    B.append(int(n_clusters_))
    
results = pd.DataFrame([A,B,C]).T
results.columns = ['distance','Number of clusters', 'Number of outliers']
results.plot(x = 'distance', y = 'Number of clusters', figsize = (10, 6))

db = DBSCAN(eps = 1, min_samples = 10).fit(df_num)
core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

dataset = dataset.join(pd.DataFrame(labels))
dataset = dataset.rename(columns = {0 : 'Cluster'})

dataset['Cluster'].value_counts()

df_clusters = dataset.groupby('Cluster')['Rating','Cocoa % as num']
df_clusters.describe()

fig, (ax1,ax2) = plt.subplots(nrows = 2,figsize=(12,12))

plt.figure(figsize=(12, 8))
plot1 = sns.boxplot(x = dataset['Cluster'], y = dataset['Rating'], data = dataset, ax = ax1)


plt.figure(figsize=(12, 8))
plot2 = sns.boxplot(x = dataset['Cluster'],y = dataset['Cocoa % as num'], data = dataset, ax = ax2)

plt.figure(figsize = (16, 12))
X = df_num

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()