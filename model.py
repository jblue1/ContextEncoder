import tensorflow as tf


'''
Functions to build the autoencoder/generator and discriminator as a keras model objects. Assumes input data is of the dimensions
(batch_size, height, width, channels). 
'''


def build_autoencoder(use_gpu, channels=3, height=128, width=128,):
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
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    encoder_output = tf.keras.layers.Conv2D(filters=100, kernel_size=4, data_format=data_format)(x)

    x = tf.keras.layers.BatchNormalization(axis)(encoder_output)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=256,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.ReLU()(x)
    decoder_output = tf.keras.layers.Conv2DTranspose(filters=3,
                                                     kernel_size=4,
                                                     strides=2,
                                                     padding='same',
                                                     activation='tanh',
                                                     data_format=data_format)(x)
    autoencoder = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')

    return autoencoder


def build_discriminator(use_gpu, channels=3, height=64, width=64):
    if use_gpu:
        data_format = 'channels_first'
        axis = 1
        discriminator_inputs = tf.keras.Input(shape=(channels, height, width))
    else:
        data_format = 'channels_last'
        axis = -1
        discriminator_inputs = tf.keras.Input(shape=(height, width, channels))

    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               data_format=data_format, name='D1')(discriminator_inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', data_format=data_format, name='D2')(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', data_format=data_format, name='D3')(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', data_format=data_format, name='D4')(x)
    x = tf.keras.layers.BatchNormalization(axis)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=4, data_format=data_format)(x)
    x = tf.keras.layers.Flatten(data_format=data_format)(x)
    discriminator_output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    discriminator = tf.keras.Model(discriminator_inputs, discriminator_output, name='discriminator')

    return discriminator


def main():
    auto = build_discriminator(True)
    auto.summary()


if __name__ == '__main__':
    main()
