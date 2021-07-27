# CNN Cats & Dogs

'''https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3 # RGB color

filenames = os.listdir(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\CNN\cnn_training_set')

def label_filter():
    labels = []
    categories = []
    for filename in filenames:
        labels.append(filename)
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        elif category == 'cat':
            categories.append(0)
    return categories, labels

categorys, labels = label_filter()

df = pd.DataFrame({'filename': labels, 'category': categorys})
df.head()
df.tail()

df['filename'].unique()
df['category'].unique()
df['category'].value_counts().plot.bar()

# See sample image
sample = random.choice(filenames)
image = load_img('D:/Programming Tutorials/Machine Learning/Projects/Datasets/CNN/cnn_training_set/' + sample)
plt.imshow(image)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

model.summary()

# Early Stop
earlystop = EarlyStopping(patience = 10)

# Learning Rate Reduction
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 2, verbose = 1, factor = 0.5, min_lr = 0.00001)

callbacks = [earlystop, learning_rate_reduction]

# Prepare Test and Train Data
train_df, validate_df = train_test_split(df, test_size = 0.20, random_state = 42)
train_df = train_df.reset_index(drop = True)
validate_df = validate_df.reset_index(drop = True)

train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

# Traning Generator
train_datagen = ImageDataGenerator(rotation_range = 15, rescale = 1./255, shear_range = 0.1, zoom_range = 0.2, 
                                   horizontal_flip = True, width_shift_range = 0.1, height_shift_range = 0.1)

train_generator = train_datagen.flow_from_dataframe(train_df, 
'D:/Programming Tutorials/Machine Learning/Projects/Datasets/CNN/cnn_training_set/', x_col = 'filename', y_col = 'category', 
target_size = IMAGE_SIZE, class_mode = 'binary', batch_size = batch_size)

# Validation Generator
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df, 
    'D:/Programming Tutorials/Machine Learning/Projects/Datasets/CNN/cnn_training_set/', x_col='filename', y_col='category', 
    target_size=IMAGE_SIZE, class_mode='binary', batch_size=batch_size)

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(example_df, 
    'D:/Programming Tutorials/Machine Learning/Projects/Datasets/CNN/cnn_training_set/', 
    x_col='filename', y_col='category', target_size=IMAGE_SIZE, class_mode='binary')

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

# Fit Model
epochs=3 if FAST_RUN else 50
history = model.fit_generator(train_generator, epochs=epochs, validation_data=validation_generator, 
                    validation_steps=total_validate//batch_size, steps_per_epoch=total_train//batch_size, callbacks=callbacks)

# Save Model
model.save_weights("model.h5")

# Virtualize Training
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

# Prepare Testing Data
test_filenames = os.listdir("../input/test1/test1")
test_df = pd.DataFrame({'filename': test_filenames})
nb_samples = test_df.shape[0]

# Create Testing Generator
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(test_df, 
    "../input/test1/test1/", 
    x_col='filename', y_col=None, class_mode=None, target_size=IMAGE_SIZE, batch_size=batch_size, shuffle=False)

# Predict
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

threshold = 0.5
test_df['probability'] = predict
test_df['category'] = np.where(test_df['probability'] > threshold, 1,0)

# Virtaulize Result
test_df['category'].value_counts().plot.bar()

# See predicted result with images
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    probability = row['probability']
    img = load_img("../input/test1/test1/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' '(' + "{}".format(round(probability, 2)) + ')')
plt.tight_layout()
plt.show()

# Submission
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis = 1, inplace = True)
submission_df.to_csv('submission.csv', index = False)