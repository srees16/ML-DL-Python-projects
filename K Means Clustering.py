#K Means Clustering
#%reset -f clears the variable explorer pane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values
#Using elbow method to find optimal no of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=10)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()

#Apply kMeans to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=10)
yKMeans=kmeans.fit_predict(x)

#Visualizing the results in chat
plt.scatter(x[yKMeans==0,0],x[yKMeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(x[yKMeans==1,0],x[yKMeans==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(x[yKMeans==2,0],x[yKMeans==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(x[yKMeans==3,0],x[yKMeans==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(x[yKMeans==4,0],x[yKMeans==4,1],s=100,c='magenta',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:, 1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income in $')
plt.ylabel('Spending Score 1-100')
plt.legend()
plt.show()