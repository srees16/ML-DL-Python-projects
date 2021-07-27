#Self Organizing Map
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Importing dataset
dataset=pd.read_csv('Credit_Card_Applications.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#Feature Scaling
from sklearn.preprocessing import MinMaxScaler as mms
sc=mms(feature_range=(0,1))
X=sc.fit_transform(X)
#Training an SOM
from minisom import MiniSom as ms
som=ms(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)
#Visualizing the SOM results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],markeredgecolor=colors[y[i]],markerfacecolor='None',markersize=10,markeredgewidth=2)
show()
#Finiding the frauds
mappings=som.win_map(X)
frauds=np.concatenate((mappings[(8,1)],mappings[(6,8)]),axis=0)
frauds=sc.inverse_transform(frauds)