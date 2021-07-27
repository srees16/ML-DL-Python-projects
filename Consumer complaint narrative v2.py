# Consumer complaint narrative analysis

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# importing dataset and dropping null observations
dataset = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\consumer_complaints.csv', low_memory = False)
dataset.head(3)
dataset = dataset[pd.notnull(dataset['consumer_complaint_narrative'])]
dataset = dataset[pd.notnull(dataset['sub_product'])]
dataset.info()

features = ['product', 'sub_product', 'consumer_complaint_narrative']
dataset = dataset[features]
dataset.columns

# factorizing the features as numerical variables
dataset['category_id'] = dataset['product'].factorize()[0]
dataset['subcategory_id'] = dataset['sub_product'].factorize()[0]

# tagging the numerical variables associated with category and sub category of each class
category_id_dataframe = dataset[['product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_dataframe.values)
id_to_category = dict(category_id_dataframe[['category_id', 'product']].values)
subcategory_id_dataframe = dataset[['sub_product', 'subcategory_id']].drop_duplicates().sort_values('subcategory_id')
subcategory_to_id = dict(subcategory_id_dataframe.values)
id_to_sub_category = dict(subcategory_id_dataframe[['subcategory_id', 'sub_product']].values)

#dictionary = {}
#dictionary['product'] = {}
#dictionary['product']['sub_product'] = dataset['consumer_complaint_narrative']
#dataset['consumer_complaint_narrative'] = dict(dataset['product'])

dataset.head(3)

# visualize the features count
hist_plot_product = plt.figure(figsize = (6, 4))
dataset.groupby('product').consumer_complaint_narrative.count().plot.bar(ylim = 0)
plt.show()

hist_plot_subproduct = plt.figure(figsize = (12, 4))
dataset.groupby('sub_product').consumer_complaint_narrative.count().plot.bar(ylim = 0)
plt.show()

# creating vectorizing object for the text using TF-IDF library
tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'l2', encoding = 'latin-1', ngram_range = (1, 2), stop_words = 'english')

# creating a sparse matrix by extracting the features from the text by TF-IDF object created above
features = tfidf.fit_transform(dataset.consumer_complaint_narrative)#.toarray()
labels = dataset.category_id
sublabels = dataset.subcategory_id
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

# picking the features and target
x = dataset['consumer_complaint_narrative']
y_product = dataset['product']
y_subproduct = dataset['sub_product']

# splitting the dataset into training and test set
X_train, X_test, y_train_product, y_test_product = train_test_split(x, y_product, random_state = 3)
X_train, X_test, y_train_subproduct, y_test_subproduct = train_test_split(x, y_subproduct, random_state = 3)

# fitting the countvectorizer and tfidf transformer to the x_train dataset
count_vect = CountVectorizer()
X_train_count_vect = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count_vect)

#Fitting the dataset to the Naive bayes classifier
classifier = MultinomialNB().fit(X_train_tfidf, y_train_product)
subclassifier = MultinomialNB().fit(X_train_tfidf, y_train_subproduct)

sample_customer_complaint = count_vect.transform([" DO NOT CLOSE THIS COMPLAINT '' I have contacted this office on numerous occasions regarding the issues with my mortgage company. Since XX/XX/2014, I have had nothing but issues with how things are handled. I reached out the VA office XXXX XXXX XXXX XXXX to assist me with the loan modification and getting things completed. After receiving the the XXXX package XXXX XXXX XXXX ) from Selene Finance I realize that the documents were expired ( probably the ones that should have been mailed out back in XXXX XXXX and the discrepancies with the information listed. I emailed XXXX XXXX the copies and she agreed that I should not sign them. She advised me that she was working closely with the VA Coordinator from Selene Finance. Shortly afterwards I received another loan modification ( XXXX XXXX ) ; which I returned to Selene Finance unsigned, with a note stating for the XXXX time that XXXX XXXX is no longer on the recorded on the deed of this property since I was advised by previous lender to remove him in order to do a deed in lieu ... if Selene Finance was really doing their job then they would know this already. Before sending the package I tried calling Selene Finance numerous times at the phone number listed on the documents only for the number to be busy each time. I do n't find them to have any concern for resolving my concerns. I thought with all of the incentives that mortgage companies receive for loan modifications I thought this would be completed."])
print("The product is {}, and sub product is {}".format(classifier.predict(sample_customer_complaint), subclassifier.predict(sample_customer_complaint)))