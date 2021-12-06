import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import numpy as np
import time

from tqdm.keras import TqdmCallback


VERBOSE = 0

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def validate(model, x_val, y_val):
    _, test_acc = model.evaluate(x_val, y_val, verbose=0)
    return test_acc


def fully_connected():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28) / 255
    x_test = x_test.reshape(10000, 28, 28) / 255
    y_train = y_train.reshape(60000, 1)
    y_train = keras.utils.to_categorical(y_train)

    y_test = y_test.reshape(10000, 1)
    y_test = keras.utils.to_categorical(y_test)

    reg_lambda = 1e-6
    model = models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(28, 28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(30, activation="relu",
                                 kernel_initializer=initializers.he_normal,
                                 kernel_regularizer=keras.regularizers.l2(reg_lambda)))
    model.add(keras.layers.Dense(10, activation="softmax",
                                 kernel_initializer=initializers.he_normal,
                                 kernel_regularizer=keras.regularizers.l2(reg_lambda)))

    #model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    start = time.time_ns()
    history = model.fit(
        x_train, y_train,
        batch_size=50,
        epochs=10,
        validation_split=0.2,
        verbose=VERBOSE,
        callbacks=[TqdmCallback(verbose=0)]
    )

    ti = (time.time_ns() - start) / 1000000000
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Dokladnosc:", test_acc, "Czas:", ti, "s")
    return history, model


def convolutional(num_of_conv_layers=1, conv_size=8, kernel_size=3):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1) / 255
    x_test = x_test.reshape(10000, 28, 28, 1) / 255
    y_train = y_train.reshape(60000, 1)
    y_train = keras.utils.to_categorical(y_train)

    y_test = y_test.reshape(10000, 1)
    y_test = keras.utils.to_categorical(y_test)

    reg_lambda = 1e-6
    model: keras.models.Sequential = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(28, 28, 1)))
    for i in range(0, num_of_conv_layers):
        model.add(keras.layers.Conv2D(conv_size, kernel_size=kernel_size,
                                      activation="relu",
                                      kernel_initializer=initializers.he_normal,
                                      kernel_regularizer=keras.regularizers.l2(reg_lambda)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(30, activation="relu",
                                 kernel_initializer=initializers.he_normal,
                                 kernel_regularizer=keras.regularizers.l2(reg_lambda)))
    model.add(keras.layers.Dense(10, activation="softmax",
                                 kernel_initializer=initializers.he_normal,
                                 kernel_regularizer=keras.regularizers.l2(reg_lambda)))

    #model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    start = time.time_ns()
    history = model.fit(
        x_train, y_train,
        batch_size=50,
        epochs=10,
        validation_split=0.2,
        verbose=VERBOSE,
        callbacks=[TqdmCallback(verbose=0)]
    )

    ti = (time.time_ns() - start) / 1000000000
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Dokladnosc:", test_acc, "Czas:", ti, "s")
    return history, model


def conv_with_pooling(num_of_pool_layers=1, pool_size=(2, 2), pooling_type="max"):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1) / 255
    x_test = x_test.reshape(10000, 28, 28, 1) / 255
    y_train = y_train.reshape(60000, 1)
    y_train = keras.utils.to_categorical(y_train)

    y_test = y_test.reshape(10000, 1)
    y_test = keras.utils.to_categorical(y_test)

    reg_lambda = 1e-6
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(8, kernel_size=3,
                                  activation="relu",
                                  kernel_initializer=initializers.he_normal,
                                  kernel_regularizer=keras.regularizers.l2(reg_lambda)))
    if num_of_pool_layers == 1:
        if pooling_type == "max":
            model.add(layers.MaxPooling2D(pool_size=pool_size))
        elif pooling_type == "avg":
            model.add(layers.AveragePooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(8, kernel_size=3,
                                  activation="relu",
                                  kernel_initializer=initializers.he_normal,
                                  kernel_regularizer=keras.regularizers.l2(reg_lambda)))
    if num_of_pool_layers == 2:
        if pooling_type == "max":
            model.add(layers.MaxPooling2D(pool_size=pool_size))
        elif pooling_type == "avg":
            model.add(layers.AveragePooling2D(pool_size=pool_size))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(30, activation="relu",
                                 kernel_initializer=initializers.he_normal,
                                 kernel_regularizer=keras.regularizers.l2(reg_lambda)))
    model.add(keras.layers.Dense(10, activation="softmax",
                                 kernel_initializer=initializers.he_normal,
                                 kernel_regularizer=keras.regularizers.l2(reg_lambda)))

    #model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    start = time.time_ns()
    history = model.fit(
        x_train, y_train,
        batch_size=50,
        epochs=10,
        validation_split=0.2,
        verbose=VERBOSE,
        callbacks=[TqdmCallback(verbose=0)]
    )

    ti = (time.time_ns() - start) / 1000000000
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Dokladnosc:", test_acc, "Czas:", ti, "s")
    return history, model


if __name__ == "__main__":
    convolutional(2)
    #fully_connected()
    #conv_with_pooling()








