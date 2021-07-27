# Auto encoders
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
# Importing the dataset
movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
# Preparing training & test set
trainSet=pd.read_csv('ml-100k/u1.base',delimiter='\t')
trainSet=np.array(trainSet,dtype='int')
testSet=pd.read_csv('ml-100k/u1.test',delimiter='\t')
testSet=np.array(testSet,dtype='int')
# Getting the no of users and movies
n_users=int(max(max(trainSet[:,0]),max(testSet[:,0])))
n_movies=int(max(max(trainSet[:,1]),max(testSet[:,1])))
# Converting the data into an array with users in rows and movies in columns
def convert(data):
    new_data=[]
    for id_users in range(1,n_users+1):
        id_movies=data[:,1][data[:,0]==id_users]
        id_ratings=data[:,2][data[:,0]==id_users]
        ratings=np.zeros(n_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data
trainSet=convert(trainSet)
testSet=convert(testSet)
# Converting data into torch tensors
trainSet=torch.FloatTensor(trainSet)
testSet=torch.FloatTensor(testSet)
# Creating the architecture of the neural network
class SAE(nn.Module):
    def __init__(self,):
        super(SAE,self).__init__()
        self.fc1=nn.Linear(n_movies,20)
        self.fc2=nn.Linear(20,10)
        self.fc3=nn.Linear(10,20)
        self.fc4=nn.Linear(20,n_movies)
        self.activation=nn.Sigmoid()
    def forward(self,x):
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x=self.activation(self.fc3(x))
        x=self.fc4(x)
        return x
sae=SAE()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(sae.parameters(),lr=0.01,weight_decay=0.5)
# Training the SAE
n_epoch=200
for epoch in range(1,n_epoch+1):
    trainLoss=0
    s=0.
    for id_user in range(n_users):
        input=Variable(trainSet[id_user]).unsqueeze(0)
        target=input.clone()
        if torch.sum(target.data>0)>0:
            output=sae(input)
            target.require_grad=False
            loss=criterion(output,target)
            mean_corrector=n_movies/float(torch.sum(target.data>0)+1e-10)
            loss.backward()
            trainLoss+=np.sqrt(loss.data[0]*mean_corrector)
            s+=1.
            optimizer.step()
    print('Epoch: '+str(epoch)+' Loss: '+str(trainLoss/s))
# Testing the SAE
testLoss=0
s=0.
for id_user in range(n_users):
    input=Variable(trainSet[id_user]).unsqueeze(0)
    target=input.clone()
    if torch.sum(target.data>0)>0:
        output=sae(input)
        target.require_grad=False
        loss=criterion(output,target)
        mean_corrector=n_movies/float(torch.sum(target.data>0)+1e-10)
        loss.backward()
        testLoss+=np.sqrt(loss.data[0]*mean_corrector)
        s+=1.
        optimizer.step()
print('Epoch: '+str(epoch)+' Loss: '+str(testLoss/s))