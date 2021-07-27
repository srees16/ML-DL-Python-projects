#K Means Clustering
#%reset -f clears the variable explorer pane
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from itertools import cycle
from sklearn.cluster import DBSCAN
from sklearn import metrics

#Importing the data set with pandas and taking the necessary variables
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values

DBSCAN(eps = 0.5, min_samples = 5, metric = 'euclidean', metric_params = None, algorithm = 'auto', leaf_size = 30, p = None, n_jobs = 1)
#Find the best epsilon
eps_grid= np.linspace(0.3, 1.2, num = 10)
sil_scores = []
eps_best = eps_grid[0]
sil_score_max = -1
model_best = None
labels_best = None
for eps in eps_grid:
    model = DBSCAN(eps = eps, min_samples = 5).fit(x) #Train DBSCAN model
    labels=model.labels_#extract labels
    sil_score=round(metrics.silhouette_score(x,labels),4)#extract performance metric
    sil_scores.append(sil_score)
    print('Epsilon: ',eps,'  > score: ',sil_score)
    if sil_score>sil_score_max:
        sil_score_max=sil_score
        eps_best=eps
        model_best=model
        labels_best=labels
plot.figure()
plot.bar(eps_grid,sil_scores,width=0.05,color='k',align='center')
#Using elbow method to find optimal no of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=10)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plot.plot(range(1,11),wcss)
plot.title('The elbow method')
plot.xlabel('No of clusters')
plot.ylabel('WCSS')
plot.show()
#Apply kMeans to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=10)
yKMeans=kmeans.fit_predict(x)
#Visualizing the results in chat
plot.scatter(x[yKMeans==0,0],x[yKMeans==0,1],s=100,c='red',label='Cluster 1')
plot.scatter(x[yKMeans==1,0],x[yKMeans==1,1],s=100,c='blue',label='Cluster 2')
plot.scatter(x[yKMeans==2,0],x[yKMeans==2,1],s=100,c='green',label='Cluster 3')
plot.scatter(x[yKMeans==3,0],x[yKMeans==3,1],s=100,c='cyan',label='Cluster 4')
plot.scatter(x[yKMeans==4,0],x[yKMeans==4,1],s=100,c='magenta',label='Cluster 5')
plot.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:, 1],s=300,c='yellow',label='Centroids')
plot.title('Clusters of Clients')
plot.xlabel('Annual Income in $')
plot.ylabel('Spending Score 1-100')
plot.legend()
plot.show()