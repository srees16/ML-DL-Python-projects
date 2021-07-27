# Autoencoder
# Ref: https://pythonmachinelearning.pro/all-about-autoencoders

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt

'''Vanilla Autoencoder'''
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.
X_train = X_train.reshape((X_train.shape[0],-1))
X_test=X_test.reshape((X_test.shape[0],-1))
INPUT_SIZE = 784
ENCODING_SIZE = 64
input_img = Input(shape = (INPUT_SIZE, ))
encoded = Dense(ENCODING_SIZE, activation = 'relu')(input_img)
decoded = Dense(INPUT_SIZE, activation = 'relu')(encoded)
autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
autoencoder.fit(X_train, X_train, epochs = 50, batch_size = 256, shuffle = True, validation_split = 0.2)
decoded_imgs = autoencoder.predict(X_test)
plt.figure(figsize = (20, 4))

for i in range(10):
    # original
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
    # reconstruction
    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
plt.tight_layout()
plt.show()

'''Deep Autoencoders'''
input_img = Input(shape=(INPUT_SIZE,))
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(ENCODING_SIZE, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(INPUT_SIZE, activation='relu')(decoded)
autoencoder = Model(input_img, decoded)

'''Convolutional Autoencoder'''
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential 
(X_train,_),(X_test,_) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
autoencoder = Sequential()
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
autoencoder.add(MaxPooling2D((2, 2), padding = 'same'))
autoencoder.add(Conv2D(8, (3, 3), activation = 'relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding = 'same'))
autoencoder.add(Conv2D(8, (3, 3), activation = 'relu', padding='same'))
#our encoding
autoencoder.add(MaxPooling2D((2, 2), padding = 'same'))
autoencoder.add(Conv2D(8, (3, 3), activation = 'relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation = 'relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(32, (3, 3), activation = 'relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation = 'relu', padding='same'))
autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
autoencoder.fit(X_train, X_train, epochs = 50, batch_size=256, shuffle=True, validation_split=0.2)

'''Denoising Autoencoder'''
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
X_train_noisy = X_train + 0.25 * np.random.normal(size=X_train.shape)
X_test_noisy = X_test + 0.25 * np.random.normal(size=X_test.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)
autoencoder.fit(X_train_noisy, X_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)
decoded_imgs = autoencoder.predict(X_test_noisy)
plt.figure(figsize=(20, 4))
for i in range(10):
    # original
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
    # reconstruction
    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    plt.axis('off')
plt.tight_layout()
plt.show()