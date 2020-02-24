import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Activation
from keras.datasets import mnist
import keras
import keras.models as models
from keras.utils.np_utils import to_categorical
from keras.callbacks import *
from keras.models import Sequential

(trainX, trainy), (testX, testy) = mnist.load_data()
x_train = trainX.astype('float32') / 255.
x_test = testX.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
trainy = to_categorical(trainy)
testy = to_categorical(testy)
print(x_train.shape[0], ' train samples')
print(x_test.shape[0], ' test samples')
    
def auto_encoder():

    # Input
    inp = Input(name='inputs', shape=(28, 28, 1), dtype='float32')
    
    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    enc = Dense(32, activation='relu', name='encoder')(x)
        
    # Decoder

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(enc)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    dec = Dense(1, activation='sigmoid', name='decoder')(x)
    
    return Model(inputs=inp, outputs=dec)

batch_size = 128
epochs = 40
autoenc = auto_encoder()
autoenc.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy"])

history = autoenc.fit(x_train_noisy, x_train, epochs=epochs, batch_size=batch_size,
            shuffle=True, validation_data=(x_test_noisy, x_test))
keras.Model.save_weights(autoenc,"Autoenc_checkpoints")

decoded_imgs = autoenc.predict(x_test_noisy)

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.fit(autoenc.predict(x_train_noisy), trainy, epochs=50, batch_size=64, validation_data=[decoded_imgs, testy])
keras.Model.save_weights(model,"model_checkpoints")
pred_result = model.predict(autoenc.predict(x_test_noisy))

