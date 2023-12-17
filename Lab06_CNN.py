import datetime
import os

import keras.models
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot as plt

logdir = "./logs/mnist/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_std"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


def load_data(verbose=False):
    (trainX, trainy), (testX, testy) = mnist.load_data()

    if verbose:
        print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
        print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
    plt.show()

    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainY = to_categorical(trainy)
    testY = to_categorical(testy)
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


def define_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def define_conv_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


trainX, trainY, testX, testY = load_data(True)

trainX, testX = prep_pixels(trainX, testX)

# model = define_model()
# if not os.path.isfile("model_std.keras"):
#     model.fit(trainX, trainY, epochs=10, batch_size=128, validation_split=0.1, callbacks=tensorboard_callback)
#     model.save("model_std.keras")
# else:
#     model = keras.models.load_model("model_std.keras")

model = define_conv_model()
if not os.path.isfile("model_conv.keras"):
    model.fit(trainX, trainY, epochs=10, batch_size=128, validation_split=0.1, callbacks=tensorboard_callback)
    model.save("model_conv.keras")
else:
    model = keras.models.load_model("model_conv.keras")

loss, acc = model.evaluate(testX, testY)

print("Loss: ", loss)
print("Accuracy: ", acc)

# predict on new data

from PIL import Image
import numpy as np
img = Image.open("zero.png").convert('L').resize((28, 28))
img = np.array(img)
prediction = model.predict(img[None,:,:])
print(prediction)

img = Image.open("dwa.png").convert('L').resize((28, 28))
img = np.array(img)
prediction = model.predict(img[None,:,:])
print(prediction)

img = Image.open("osiem.png").convert('L').resize((28, 28))
img = np.array(img)
prediction = model.predict(img[None,:,:])
print(prediction)
