#News classification based on its headline
#Ref: https://www.kaggle.com/andressotov/news-classification-based-on-its-headline
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
news=pd.read_csv('uci-news-aggregator.csv')
print(news.head(5))
encoder=LabelEncoder()
y=encoder.fit_transform(news['CATEGORY'])
print(y[:5])
categories=news['CATEGORY']
titles=news['TITLE']
N=len(titles)
print('Number of news',N)
labels=list(set(categories))
print('possible categories',labels)
for l in labels:
    print('number of ',l,' news',len(news.loc[news['CATEGORY']==l]))
ncategories=encoder.fit_transform(categories)
Ntrain=int(N*0.7)
titles,ncategories=shuffle(titles,ncategories,random_state=8)
X_train=titles[:Ntrain]
print('X_train.shape',X_train.shape)
y_train=ncategories[:Ntrain]
print('y_train.shape',y_train.shape)
X_test=titles[Ntrain:]
print('X_test.shape',X_test.shape)
y_test=ncategories[Ntrain:]
print('y_test.shape',y_test.shape)
print('Training...')
text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',MultinomialNB()),])
text_clf = text_clf.fit(X_train, y_train)
print('Predicting...')
predicted = text_clf.predict(X_test)
print('accuracy_score',metrics.accuracy_score(y_test,predicted))
print('Reporting...')
print(metrics.classification_report(y_test, predicted, target_names=labels))