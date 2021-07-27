#Spam Classification (Sundog tutorial)
import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
'''Reading the files, builds individual file path for each and reads them,while skipping 
header. Gives path to each file and body of the files
'''
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path=os.path.join(root,filename)
            inBody=False;
            lines=[]
            f=io.open(path,'r',encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line=='\n': #checks for the blank line, n skips it. Goes to next line
                    inBody=True
            f.close()
            message='\n'.join(lines)
            yield path,message
def dataFrameFromDirectory(path,classification):
    rows=[]
    index=[]
    for filename,message in readFiles(path):
        rows.append({'message':message,'class':classification})
        index.append(filename)
    return DataFrame(rows,index=index)
#Creating data frame list of objects with the messages and its class
data=DataFrame({'message':[],'class':[]})
data=data.append(dataFrameFromDirectory('file location of the mails','spam'))
data=data.append(dataFrameFromDirectory('file location of the mails','ham'))
data.head #To view the sample files
#Splitting the data
vectorizer=CountVectorizer()
counts=vectorizer.fit_transform(data['message'].values) #converts/tokenizes individual words into nos
classifier=MultinomialNB() #Performs Naive bayes on the dataset
targets=data['class'].values
classifier.fit(counts,targets) #expects acutual training data & labels for each file
#Testing the code
examples=['Free Viagra','Lets go on a data sir']
example_counts=vectorizer.transform(examples)
predictions=classifier.predict(example_counts)
predictions