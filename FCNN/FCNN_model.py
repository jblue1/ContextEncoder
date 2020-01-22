"""
Function to build the FCNN
"""
import tensorflow as tf


def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5000, activation='relu', input_shape=(5632,)))
    model.add(tf.keras.layers.Dense(5000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(5000, activation='relu'))
    model.add(tf.keras.layers.Dense(5000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(5000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(5000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(4608, activation='sigmoid'))

    return model


def main():
    model = build_model()
    model.summary()


if __name__ == '__main__':
    main()
