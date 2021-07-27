'''
Ref - https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f OR
https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb
'''

import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\consumer_complaints.csv', low_memory = False)
dataset_copy = dataset.copy()
dataset.head(3)
dataset = dataset[pd.notnull(dataset['consumer_complaint_narrative'])]
dataset.info()

'''Product vs consumer_complaint_narrative'''

col = ['product', 'consumer_complaint_narrative']
dataset = dataset[col]
dataset.columns
# df.columns = ['product', 'consumer_complaint_narrative'] # same as above
dataset['category_id'] = dataset['product'].factorize()[0]

# finding the numerical variables associated with category of each class
category_id_dataframe = dataset[['product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_dataframe.values)
id_to_category = dict(category_id_dataframe[['category_id', 'product']].values)

dataset.head(3)

# visualize the features count
hist_plot = plt.figure(figsize = (6, 4))
dataset.groupby('product').consumer_complaint_narrative.count().plot.bar(ylim = 0)
plt.show()

# creating vectorizing object for the text using TF-IDF library
tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'l2', encoding = 'latin-1', ngram_range = (1, 2), stop_words = 'english')

#  creating a sparse matrix by extracting the features from the text by TF-IDF object created above
features = tfidf.fit_transform(dataset.consumer_complaint_narrative)#.toarray()
labels = dataset.category_id
features.shape

# finding the correlation using chi2 library for unigrams and bigrams
N = 2
for product, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [grams for grams in feature_names if len(grams.split(' ')) == 1]
    bigrams = [grams for grams in feature_names if len(grams.split(' ')) == 2]
    print("# '{}':".format(product))
    print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

# splitting to training and test set
X_train, X_test, y_train, y_test = train_test_split(dataset['consumer_complaint_narrative'], dataset['product'], random_state = 0)

# fitting the countvectorizer and tfidf transformer to the x_train dataset
count_vect = CountVectorizer()
X_train_count_vect = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count_vect)

#Fitting the dataset to the Naive bayes classifier
classifier = MultinomialNB().fit(X_train_tfidf, y_train)

# sample predictions
classifier.predict(count_vect.transform(['This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine.']))
classifier.predict(count_vect.transform(["I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"]))

dataset[dataset['consumer_complaint_narrative'] == 'This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine.']
dataset[dataset['consumer_complaint_narrative'] == "I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"]

# probing for the best fit model for training the above dataset
models = [RandomForestClassifier(), LinearSVC(), MultinomialNB(), LogisticRegression(random_state = 0),]
cross_validation = 5
cv_dataset = pd.DataFrame(index = range(cross_validation * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring = 'accuracy', cv = cross_validation)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_dataset = pd.DataFrame(entries, columns = ['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x = 'model_name', y = 'accuracy', data = cv_dataset)
sns.stripplot(x = 'model_name', y = 'accuracy', data = cv_dataset, size = 8, jitter = True, edgecolor = "gray", linewidth = 2)
plt.show()

cv_dataset.groupby('model_name').accuracy.mean()

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, dataset.index, test_size = 0.33, random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# confusion matrix and its visualization
c_matrix = confusion_matrix(y_test, y_pred)
figure, ax = plt.subplots(figsize = (8, 6))
sns.heatmap(c_matrix, annot = True, fmt = 'd', xticklabels = category_id_dataframe.product(), yticklabels = category_id_dataframe.product())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

for predicted in category_id_dataframe.category_id:
  for actual in category_id_dataframe.category_id:
    if predicted != actual and c_matrix[actual, predicted] >= 6:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], c_matrix[actual, predicted]))
      display(dataset.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['product', 'consumer_complaint_narrative']])
      print('')

model.fit(features, labels)

N = 2
for Product, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(Product))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

texts = ["I requested a home loan modification through Bank of America. Bank of America never got back to me.",
         "It has been difficult for me to find my past due balance. I missed a regular monthly payment",
         "I can't get the money out of the country.",
         "I have no money to pay my tuition",
         "Coinbase closed my account for no reason and furthermore refused to give me a reason despite dozens of request"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

# final metrics report
print(metrics.classification_report(y_test, y_pred, target_names = dataset['product'].unique()))