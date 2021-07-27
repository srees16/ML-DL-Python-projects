#Spam classification Johar
import os
from collections import Counter

def makingDict():
    directory='D:\\Programming Tutorials\\Machine Learning\\Datasets\\Enron\\Emails\\'
    files=os.listdir(directory)
    #Preprocessing the emails
    emails=[directory+email for email in files] #appends the directory to the files,contains paths of all files
    words=[] #repo of all the words
    c=len(emails)
    for email in emails:
        f=open(email, encoding='latin1') #open & read the mails
        bloc=f.read()
        words=words+bloc.split(' ')
        print(c)
        c-=1
    #eliminate all non alphabetic words
    for i in range(len(words)):
        if not words[i].isalpha():
            words[i]=''
    dict=Counter(words)
    del dict[' ']
    return dict.most_common(3000) #counter of most common words in the files
#making dataset using NB which needs it to be fit as feature vector. Feature Vectorization helps
def makingDataset():
    #Preprocessing the emails
    directory='D:\\Programming Tutorials\\Machine Learning\\Datasets\\Enron\\Emails\\'
    files=os.listdir(directory)
    emails=[directory+email for email in files] #appends the directory to the files,contains paths of all files
    
    featureSet=[]
    labels=[]
    c=len(emails)
    for email in emails:
        data=[] #repo of all the words
        f=open(email)
        words=f.read().split(' ')
        for entry in dict:
            data.append(words.count(entry[0]))
        featureSet.append(data)
        if 'ham' in email:
            labels.append(0)
        if 'spam' in email:
            labels.append(1)
        print(c)
        c-=1
    return featureSet, labels
#Calling the above functions
d=makingDict()
features, labels=makingDataset(d)
print (len(features),len(labels))