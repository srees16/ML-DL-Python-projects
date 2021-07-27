#K Means Clustering
#%reset -f clears the variable explorer pane
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn import metrics
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values
#Using elbow method to find optimal no of clusters
from sklearn.cluster import KMeans
wcss=[]
scores=[]
ranges=np.arange(2,10)
for i in ranges:
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10)
    kmeans.fit(x)
    score=metrics.silhouette_score(x,kmeans.labels_,metric='euclidean',sample_size=len(x))
    print('No of clusters',i)
    print('Silhouette',score)
    scores.append(score)
plot.figure()
plot.bar(ranges,scores,width=0.6,color='k',align='center')
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