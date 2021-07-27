#LIME Spooky Author Identification
'''Ref: https://www.kaggle.com/emanceau/interpreting-machine-learning-lime-explainer/notebook
https://towardsdatascience.com/understanding-model-predictions-with-lime-a582fdff3a3b'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import ensemble, metrics, model_selection, naive_bayes
from sklearn.pipeline import make_pipeline
import lime
lime.__package__
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import itertools
%matplotlib inline
import warnings
warnings.simplefilter('ignore')

train_url = 'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Spooky Author Identification train.csv'
test_url = 'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Spooky Author Identification test.csv'
train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# verify if the dataset is imbalanced
train_df.author.value_counts()

#Using explainer with Tfidf vectorizer
class_names = ['EAP', 'HPL', 'MWS']
cols_to_drop = ['id', 'text']
train_X = train_df.drop(cols_to_drop + ['author'], axis = 1)

'''Mapping the categorical classes to numerical'''

# 1st way to map by creating a dict of key-value pairs
author_mapping_dict = {'EAP' : 0, 'HPL' : 1, 'MWS' : 2}
train_y = train_df['author'].map(author_mapping_dict)

# 2nd way to map by factorizing, but gives error at line 93 saying pandas has no .to_frame() attribute
train_y = train_df['author'].factorize()[0]

train_id = train_df['id'].values

tfidf_vec = TfidfVectorizer(ngram_range = (1, 5), analyzer = 'char')
full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

X_train, X_test, y_train, y_test = train_test_split(train_tfidf, train_y, test_size = 0.33, random_state = 14)
model_tf = naive_bayes.MultinomialNB()
model_tf.fit(X_train, y_train)

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    '''Function prints and plots the confusion matrix. Normalization can be applied by setting normalize=True'''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
y_pred = model_tf.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision = 2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = class_names, title = 'Confusion matrix, without normalization')
plt.show()

#Preparing lime explainer
c_tf = make_pipeline(tfidf_vec, model_tf)
explainer_tf = LimeTextExplainer(class_names = class_names)

comp = y_test.to_frame()
comp['idx'] = comp.index.values
comp['pred'] = y_pred
comp.rename(columns = {'author': 'real'}, inplace = True)

#True POE but classified in HPL
wrong_poe_hpl = comp[(comp.real == 0) & (comp.pred == 1)]
wrong_poe_hpl.shape
print(wrong_poe_hpl.idx)
idx = wrong_poe_hpl.idx.iloc[1]

# =============================================================================
# from IPython.core.display import display, HTML
# from IPython.display import IFrame
# IFrame(src='./nice.html', width=700, height=600)
# display(HTML('<h1>Hello, world!</h1>').data)
# 
# =============================================================================

# Using Lime explainer
explain = explainer_tf.explain_instance(train_df['text'][idx], c_tf.predict_proba, num_features = 4, top_labels = 2)
explain.show_in_notebook(text = train_df['text'][idx], labels = (0, 1))

# This error is created by the use of ancient greek words. Possible to improve the model?
idx = wrong_poe_hpl.idx.iloc[3]
exp = explainer_tf.explain_instance(train_df['text'][idx], c_tf.predict_proba, num_features = 4, top_labels = 2)
exp.show_in_notebook(text=train_df['text'][idx], labels = (0, 1))

# True POE but classified in MWS
wrong_poe_mws = comp[(comp.real == 0) & (comp.pred == 2)]
print(wrong_poe_mws.shape)
idx = wrong_poe_mws.idx.iloc[12]

exp = explainer_tf.explain_instance(train_df['text'][idx], c_tf.predict_proba, num_features = 4, top_labels = 3)
exp.show_in_notebook(text=train_df['text'][idx], labels = (0, 1))

# this text contains anaphora, possible to improve the model with anaphora feature
idx = wrong_poe_mws.idx.iloc[18]
exp = explainer_tf.explain_instance(train_df['text'][idx], c_tf.predict_proba, num_features = 4, top_labels = 3)
exp.show_in_notebook(text=train_df['text'][idx], labels=(0, 1, 2))

# probabilities (EAP and MWS) are very close. Possible to improve the model.

'''True MWS but classified in HPL'''
wrong_mws_hpl = comp[(comp.real == 2) & (comp.pred == 1)]
print(wrong_mws_hpl.shape)
idx = wrong_mws_hpl.idx.iloc[8]

exp = explainer_tf.explain_instance(train_df['text'][idx], c_tf.predict_proba, num_features = 4, top_labels = 3)
exp.show_in_notebook(text = train_df['text'][idx], labels = (0, 1, 2))

# probabilities (HPL and MWS) are very close. Possible to improve the model
idx = wrong_mws_hpl.idx.iloc[5]
exp = explainer_tf.explain_instance(train_df['text'][idx], c_tf.predict_proba, num_features = 4, top_labels = 3)
exp.show_in_notebook(text = train_df['text'][idx], labels = (0, 1, 2))

# probabilities (EAP, HPL, MWS ) are all very close. Possible to improve the model (using repetition pattern ?)
idx = wrong_mws_hpl.idx.iloc[5]
explain = explainer_tf.explain_instance(train_df['text'][idx], c_tf.predict_proba, num_features = 4, top_labels = 3)
explain.show_in_notebook(text = train_df['text'][idx], labels = (0, 1, 2))