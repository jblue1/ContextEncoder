"""
Function to build CNN that takes in images of the broken particle tracks, and outputs a vector containing the predicted
z coordinate of each pad.
"""

import tensorflow as tf


def build_convnet(use_gpu):
    if use_gpu:
        data_format = 'channels_first'
        axis = 1
        input = tf.keras.Input(shape=(1, 128, 175))
    else:
        data_format = 'channels_last'
        axis = -1
        input = tf.keras.Input(shape=(128, 175, 1))

    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=4,
                               strides=1,
                               padding='same',
                               data_format=data_format)(input)
    x = tf.keras.layers.MaxPooling2D((2,2), data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis=axis)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', data_format=data_format)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis=axis)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', data_format=data_format)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis=axis)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', data_format=data_format)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis=axis)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', data_format=data_format)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis=axis)(x)
    output = tf.keras.layers.Flatten()(x)
    cnn = tf.keras.Model(input, output, name='cnn')
    return cnn


def main():
    model = build_convnet(False)
    model.summary()


if __name__ == '__main__':
    main()