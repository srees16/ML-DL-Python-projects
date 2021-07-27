#News Classification 2
#Ref: https://www.kaggle.com/arjunchandrasekhara/news-classification/notebook
import re
import pandas as pd # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix
#Function to get the words from the headlines
def get_words(headlines):
    headlines_onlyletters=re.sub("[^a-zA-Z]"," ",headlines) #Remove everything other than letters     
    words=headlines_onlyletters.lower().split() #Convert to lower case, split into individual words    
    stops=set(stopwords.words("english"))  #Convert the stopwords to a set for improvised performance                 
    meaningful_words=[w for w in words if not w in stops]   #Removing stopwords
    return(" ".join( meaningful_words)) #Joining the words
#Reading data and Splitting as train and test sets
news=pd.read_csv('uci-news-aggregator.csv') #Importing data from CSV
news=(news.loc[news['CATEGORY'].isin(['b','e'])]) #Retaining rows that belong to categories 'b' and 'e'
X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(news["TITLE"],news["CATEGORY"],test_size=0.2)
X_train=np.array(X_train);
X_test=np.array(X_test);
Y_train=np.array(Y_train);
Y_test=np.array(Y_test);
cleanHeadlines_train=[] #To append processed headlines
cleanHeadlines_test=[] #To append processed headlines
number_reviews_train=len(X_train) #Calculating the number of reviews
number_reviews_test=len(X_test) #Calculating the number of reviews
#Getting only the words from the headlines, removing the stopwords, numbers and special characters
for i in range(0,number_reviews_train):
    cleanHeadline=get_words(X_train[i]) #Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadlines_train.append(cleanHeadline)
for i in range(0,number_reviews_test):
    cleanHeadline=get_words(X_test[i]) #Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadlines_test.append( cleanHeadline )
#Creating the bag of words for each headline
vectorize=sklearn.feature_extraction.text.CountVectorizer(analyzer="word",max_features=1700)
bagOfWords_train=vectorize.fit_transform(cleanHeadlines_train)
X_train=bagOfWords_train.toarray()
bagOfWords_test=vectorize.transform(cleanHeadlines_test)
X_test=bagOfWords_test.toarray()
#Naive Bayes for training and calculating the accuracy of the model
vocab=vectorize.get_feature_names()
nb=MultinomialNB()
nb.fit(X_train,Y_train)
print(nb.score(X_test,Y_test))
#Logistic Regression
logistic_Regression=LogisticRegression()
logistic_Regression.fit(X_train,Y_train)
Y_predict=logistic_Regression.predict(X_test)
print(accuracy_score(Y_test,Y_predict))