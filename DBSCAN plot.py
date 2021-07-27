# DBSCAN clustering algorithm

'''Ref: http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py'''

print(__doc__)
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plot
# Generate sample data
centers=[[1,1],[-1,-1],[1,-1]]
X,labels_true=make_blobs(n_samples=750,centers=centers,cluster_std=0.4,random_state=0)
X=StandardScaler().fit_transform(X)
# Compute DBSCAN
db=DBSCAN(eps=0.3,min_samples=10).fit(X)
core_samples_mask=np.zeros_like(db.labels_,dtype=bool)
core_samples_mask[db.core_sample_indices_]=True
labels=db.labels_
# Number of clusters in labels, ignoring noise if present
n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
#Plot result. Black removed and is used for noise instead
unique_labels=set(labels)
colors=[plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k,col in zip(unique_labels,colors):
    if k==-1:
        col=[0,0,0,1] #black used for noise
    class_member_mask=(labels==k)
    xy=X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=14)
    xy=X[class_member_mask& ~core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=6)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
'''
Ref: https://www.youtube.com/watch?v=TGad0nc-8gU
'''
#Using 'Churn modelling' dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
dataset=pd.read_csv('Churn_modelling.csv')
scale=MinMaxScaler()
y_actual=dataset.iloc[:,13].values
x=dataset.iloc[:,[8,12]].values
x=scale.fit_transform(x)
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=22)
yPred=kmeans.fit_predict(x)
cluster_labels=np.unique(y)
for i in cluster_labels:
    

