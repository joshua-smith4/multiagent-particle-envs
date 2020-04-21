import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_outputs):
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, input_shape=input_shape, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_outputs, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

