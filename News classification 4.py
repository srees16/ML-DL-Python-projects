#NEWS ARTICLE CLASSIFICATION
#Ref: https://www.kaggle.com/cnokello/news-article-classification-using-naive-bayes
import pandas as pd
import string
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
news_df = pd.read_csv('uci-news-aggregator.csv', sep = ",")
#news_df.CATEGORY.unique()
news_df['CATEGORY']=news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE']=news_df.TITLE.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))
news_df.head(3)
X_train,X_test,y_train,y_test=tts(news_df['TITLE'],news_df['CATEGORY'],random_state=1)
print("Training dataset: ",X_train.shape[0])
print("Test dataset: ",X_test.shape[0])
count_vector=CountVectorizer(stop_words='english')
training_data=count_vector.fit_transform(X_train)
testing_data=count_vector.transform(X_test)
naive_bayes=MultinomialNB()
naive_bayes.fit(training_data,y_train)
predictions=naive_bayes.predict(testing_data)
predictions
print("Accuracy score: ", accuracy_score(y_test, predictions))
print("Recall score: ", recall_score(y_test, predictions, average = 'weighted'))
print("Precision score: ", precision_score(y_test, predictions, average = 'weighted'))
print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))