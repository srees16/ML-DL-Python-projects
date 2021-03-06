#Boltzmann Machine
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
#Importing Dataset
movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
#Preparing training & test set
trainSet=pd.read_csv('ml-100k/u1.base',delimiter='\t')
trainSet=np.array(trainSet,dtype='int')
testSet=pd.read_csv('ml-100k/u1.test',delimiter='\t')
testSet=np.array(testSet,dtype='int')
#Getting the no of users and movies
n_users=int(max(max(trainSet[:,0]),max(testSet[:,0])))
n_movies=int(max(max(trainSet[:,1]),max(testSet[:,1])))
#Converting the data into array with users in lines, movies in columns
def convert(data):
    newData=[]
    for id_users in range(1,n_users+1):
        id_movies=data[:,1][data[:,0]==id_users]
        id_rating=data[:,2][data[:,0]==id_users]
        ratings=np.zeros(n_movies)
        ratings[id_movies-1]=id_rating
        newData.append(list(ratings))
    return newData
trainSet=convert(trainSet)
testSet=convert(testSet)
#Converting the data into torch tensors
trainSet=torch.FloatTensor(trainSet)
testSet=torch.FloatTensor(testSet)
#Converting the ratings into binary ratings 1 and 0
trainSet[trainSet==0]=-1
trainSet[trainSet==1]=0
trainSet[trainSet==2]=0
trainSet[trainSet>=3]=1
testSet[testSet==0]=-1
testSet[testSet==1]=0
testSet[testSet==2]=0
testSet[testSet>=3]=1
#Creating the architecture of neural network
class RBM():
    def __init__(self,nv,nh):
        self.W=torch.randn(nh,nv)
        self.a=torch.randn(1,nh)
        self.b=torch.randn(1,nv)
    def sample_h(self,x):
        wx=torch.mm(x,self.W.t())
        activation=wx+self.a.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        wy=torch.mm(y,self.W)
        activation=wy+self.b.expand_as(wy)
        p_v_given_h=torch.sigmoid(activation)
        return p_v_given_h,torch.bernoulli(p_v_given_h)
    def train(self,v0,vk,ph0,phk):
        self.W+=torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)
        self.b+=torch.sum((v0-vk),0)
        self.a+=torch.sum((ph0-phk),0)
nv=len(trainSet[0])
nh=100
batch_size=100
rbm=RBM(nv,nh)
#Training the RBM
nb_epoch=10
for epoch in range(1,nb_epoch+1):
    train_loss=0
    s=0.
    for id_user in range(0,n_users-batch_size,batch_size):
        vk=trainSet[id_user:id_user+batch_size]
        v0=trainSet[id_user:id_user+batch_size]
        ph0,_=rbm.sample_h(v0)
        for k in range(10):
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]
        phk,_=rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss+=torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s+=1.
    print('Epoch: '+str(epoch)+' loss: '+str(train_loss/s))
#Testing the RBM
test_loss=0
s=0.
for id_user in range(n_users):
    v=trainSet[id_user:id_user+1]
    vt=testSet[id_user:id_user+1]
    if len(vt[vt>=0])>0:
        _,h=rbm.sample_h(v)
        _,v=rbm.sample_v(h)
        test_loss+=torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s+=1.
print('Test loss: '+str(test_loss/s))