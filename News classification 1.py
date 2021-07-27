# News Category classifier
'''Ref: https://www.kaggle.com/hengzheng/news-category-classifier-val-acc-0-65/notebook'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize']=(6,6)
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D, GRU,Concatenate, Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Flatten, SpatialDropout1D, GlobalAveragePooling1D, BatchNormalization, concatenate # Add, Reshape, merge, Lambda, Average, Activation, TimeDistributed
from keras.models import Model # Sequential, load_model
#from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
#from keras.layers.merge import add
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer # text_to_word_sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#importing the dataset to read json
dataset = pd.read_json(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\News_Category_Dataset.json', lines = True)
dataset.head(3)
categories = dataset.groupby('category')
print("No of categories:", categories.ngroups) #grouping the news as per their categories
print(categories.size()) #No of news in each category
dataset.category = dataset.category.map(lambda x : "WORLDPOST" if x == "THE WORLDPOST" else x) #Merging identical categories using lambda (anonymous) function

#using headlines and short_description as input
dataset['text'] = dataset.headline + " " + dataset.short_description

#tokenizing the text in 'text' column
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset.text)
X = tokenizer.texts_to_sequences(dataset.text)
dataset['words'] = X

#removing empty and short data
dataset['word_length'] = dataset.words.apply(lambda i : len(i))
dataset = dataset[dataset.word_length >= 5] #removing words less than 5 characters
dataset.head(3)
dataset.word_length.describe() #Word count

#using 50 for padding length
maxlen = 50
X = list(sequence.pad_sequences(dataset.words, maxlen = maxlen))

# list of the categories and the size
categories = dataset.groupby('category').size().index.tolist()
category_int = {}
int_category = {}
for i, k in enumerate(categories):
    category_int.update({k : i})
    int_category.update({i : k})
dataset['c2id'] = dataset['category'].apply(lambda x : category_int[x])

'''Glove embedding'''

word_index = tokenizer.word_index
EMBEDDING_DIM = 100
embeddings_index = {}
glove_file = open('D:\Programming Tutorials\Machine Learning\Projects\Datasets\glove.6B.100d.txt', encoding = 'utf-8')
for line in glove_file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
glove_file.close()
print('Found %s unique tokens.' % len(word_index))
print('Total %s word vectors.' % len(embeddings_index))

# Embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, embeddings_initializer = Constant(embedding_matrix), input_length = maxlen, trainable = False)

# Dependent and independent variables
X = np.array(X)
Y = np_utils.to_categorical(list(dataset.c2id))

#split to training set and validation set
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state = 15)

'''Text CNN'''

inp = Input(shape = (maxlen, ), dtype = 'int32')
embedding = embedding_layer(inp)
stacks = []
for kernel_size in [2, 3, 4]:
    conv = Conv1D(64, kernel_size, padding = 'same', activation = 'relu', strides = 1)(embedding)
    pool = MaxPooling1D(pool_size = 3)(conv)
    drop = Dropout(0.5)(pool)
    stacks.append(drop)
merged = Concatenate()(stacks)
flatten = Flatten()(merged)
drop = Dropout(0.5)(flatten)
outp = Dense(len(int_category), activation = 'softmax')(drop)
TextCNN = Model(inputs = inp, outputs = outp)
TextCNN.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
TextCNN.summary()

# Training
textcnn_history = TextCNN.fit(x_train, y_train, batch_size = 128, epochs = 20, validation_data = (x_val, y_val))

# Accuracy and visualization
acc = textcnn_history.history['acc']
val_acc = textcnn_history.history['val_acc']
loss = textcnn_history.history['loss']
val_loss = textcnn_history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label = 'Training acc')
plt.plot(epochs, val_acc, 'blue', label = 'Validation acc')
plt.legend()
plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label = 'Training loss')
plt.plot(epochs, val_loss, 'blue', label = 'Validation loss')
plt.legend()
plt.show()

'''Bidrectional LSTM with convolution from https://www.kaggle.com/eashish/bidirectional-gru-with-convolution'''

inp = Input(shape = (maxlen,), dtype = 'int32')
x = embedding_layer(inp)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1))(x)
x = Conv1D(64, kernel_size = 3)(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
outp = Dense(len(int_category), activation = "softmax")(x)
BiGRU = Model(inp,outp)
BiGRU.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
BiGRU.summary()

# Training
bigru_history = BiGRU.fit(x_train, y_train, batch_size = 128, epochs = 20, validation_data = (x_val, y_val))

# Plotting
plt.rcParams['figure.figsize'] = (6, 6)
acc = bigru_history.history['acc']
val_acc = bigru_history.history['val_acc']
loss = bigru_history.history['loss']
val_loss = bigru_history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label = 'Training acc')
plt.plot(epochs, val_acc, 'blue', label = 'Validation acc')
plt.legend()
plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label = 'Training loss')
plt.plot(epochs, val_loss, 'blue',label = 'Validation loss')
plt.legend()
plt.show()

'''LSTM with Attention'''

class Attention(Layer):
    def __init__(self, step_dim, W_regularizer = None, b_regularizer = None, W_constraint = None, b_constraint = None, bias = True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], ), initializer = self.init, name = '{}_W'.format(self.name), regularizer=self.W_regularizer, constraint = self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1], ), initializer = 'zero', name = '{}_b'.format(self.name), regularizer=self.b_regularizer, constraint = self.b_constraint)
        else:
            self.b = None
        self.built = True
    def compute_mask(self, input, input_mask = None):
        return None
    def call(self, x, mask = None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis = 1, keepdims = True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis = 1)
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
lstm_layer = LSTM(300, dropout = 0.25, recurrent_dropout = 0.25, return_sequences = True)
inp = Input(shape = (maxlen, ), dtype = 'int32')
embedding = embedding_layer(inp)
x = lstm_layer(embedding)
x = Dropout(0.25)(x)
merged = Attention(maxlen)(x)
merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.25)(merged)
merged = BatchNormalization()(merged)
outp = Dense(len(int_category), activation = 'softmax')(merged)
AttentionLSTM = Model(inputs = inp, outputs = outp)
AttentionLSTM.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
AttentionLSTM.summary()
attlstm_history = AttentionLSTM.fit(x_train, y_train, batch_size = 128, epochs = 20, validation_data = (x_val, y_val))
acc = attlstm_history.history['acc']
val_acc = attlstm_history.history['val_acc']
loss = attlstm_history.history['loss']
val_loss = attlstm_history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label = 'Training acc')
plt.plot(epochs, val_acc, 'blue', label = 'Validation acc')
plt.legend()
plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label = 'Training loss')
plt.plot(epochs, val_loss, 'blue', label = 'Validation loss')
plt.legend()
plt.show()

#confusion matrix
predicted = AttentionLSTM.predict(x_val)
cm = pd.DataFrame(confusion_matrix(y_val.argmax(axis = 1), predicted.argmax(axis = 1)))
from IPython.display import display
pd.options.display.max_columns = None
display(cm)

#evaluate accuracy
def evaluate_accuracy(model):
    predicted = model.predict(x_val)
    diff = y_val.argmax(axis = -1) - predicted.argmax(axis = -1)
    corrects = np.where(diff == 0)[0].shape[0]
    total = y_val.shape[0]
    return float(corrects/total)
print("model TextCNN accuracy:          %.6f" % evaluate_accuracy(TextCNN))
print("model Bidirectional GRU + Conv:  %.6f" % evaluate_accuracy(BiGRU))
print("model LSTM with Attention:       %.6f" % evaluate_accuracy(AttentionLSTM))

#Evaluate accuracy
def evaluate_accuracy_ensemble(models):
    res = np.zeros(shape = y_val.shape)
    for model in models:
        predicted = model.predict(x_val)
        res += predicted
    res /= len(models)
    diff = y_val.argmax(axis = -1) - res.argmax(axis = -1)
    corrects = np.where(diff == 0)[0].shape[0]
    total = y_val.shape[0]
    return float(corrects/total)
print(evaluate_accuracy_ensemble([TextCNN, BiGRU, AttentionLSTM]))