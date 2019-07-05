"""
Functions to build the autoencoder/generator and discriminator as a keras model objects.
"""
import tensorflow as tf


def build_autoencoder(use_gpu, channels=1, height=128, width=175,):
    if use_gpu:
        data_format = 'channels_first'
        axis = 1
        encoder_input = tf.keras.Input(shape=(channels, height, width))
    else:
        data_format = 'channels_last'
        axis = -1
        encoder_input = tf.keras.Input(shape=(height, width, channels))

    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               data_format=data_format)(encoder_input)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(filters=1000, kernel_size=(2, 3), data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(2, 3),  data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.ReLU(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.ReLU(0.2)(x)
    x = tf.keras.layers.ZeroPadding2D(((0, 0), (0, 1)), data_format=data_format)(x)
    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.ReLU(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.ReLU(0.2)(x)
    x = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)), data_format=data_format)(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.ReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=1,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        activation='sigmoid',
                                        data_format=data_format)(x)
    decoder_output = tf.keras.layers.Cropping2D(((0, 0), (0, 1)), data_format=data_format)(x)
    autoencoder = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')

    return autoencoder


def build_discriminator(use_gpu, channels=1, height=128, width=175):
    if use_gpu:
        data_format = 'channels_first'
        axis = 1
        discriminator_inputs = tf.keras.Input(shape=(channels, height, width))
    else:
        data_format = 'channels_last'
        axis = -1
        discriminator_inputs = tf.keras.Input(shape=(height, width, channels))

    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               data_format=data_format)(discriminator_inputs)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.MaxPool2D(2, data_format=data_format)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.MaxPool2D(2, data_format=data_format)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.MaxPool2D(2, data_format=data_format)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.MaxPool2D(2, data_format=data_format)(x)
    x = tf.keras.layers.Flatten(data_format=data_format)(x)
    discriminator_output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    discriminator = tf.keras.Model(discriminator_inputs, discriminator_output, name='discriminator')

    return discriminator

