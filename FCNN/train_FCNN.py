"""
Script to train a FCNN to predict overbiased pad response for the Mg22 run in the AT-TPC
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import click
from datetime import date
import FCNN_model
import load_data
import pandas as pd
import numpy as np
from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import Callback

class Metrics(Callback):
    def __init__(self, val_features, val_targets):
        self.val_features = val_features
        self.val_targ = val_targets

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.val_features))).round()
        _val_recall = recall_score(self.val_targ, val_predict, average='micro')
        _val_precision = precision_score(self.val_targ, val_predict, average='micro')
        _val_f1 = 2 * (_val_precision * _val_recall) / (_val_precision + _val_recall)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('- val_f1: {}  - val_precision: {}  -val_recall: {}'.format(_val_f1, _val_precision, _val_recall))
        return


def plot_loss(loss, val_loss, save_dir):
    """
    Makes a plot of training and validation loss over the course of training.
    :param loss: training loss
    :param val_loss: validation loss
    :param save_dir: directory to save the image to
    """
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    filename = os.path.join(save_dir, 'Loss_history.png')
    plt.savefig(filename)
    plt.close()


def write_info_file(save_dir, features_path, targets_path, batch_size, epochs, lr, run_number):
    """
    Writes a text file to the save directory with a summary of the hyper-parameters used for training
    :param str save_dir: path to directory to save the file to
    :param str data_path: path to .h5 data file
    :param int batch_size: size of batches used in training
    :param int epochs: number of epochs network was trained for
    :param float lr: learning rate for the optimizer
    :param str run_number: Run number of the day
    """
    filename = os.path.join(save_dir, 'run_info.txt')
    info_list = ['ContextEncoder Hyper-parameters: Run {} \n'.format(run_number),
                 'Training data found at: {} and {} \n'.format(features_path, targets_path),
                 'Batch Size: {} \n'.format(batch_size),
                 'Epochs: {} \n'.format(epochs),
                 'Learning Rate: {} \n'.format(lr)]

    with open(filename, 'w') as f:
        f.writelines(info_list)


@click.command()
@click.argument('features_path', type=click.Path(exists=True, readable=True))
@click.argument('targets_path', type=click.Path(exists=True, readable=True))
@click.option('--batch_size', default=32)
@click.option('--epochs', default=50)
@click.option('--lr', default=0.001, help='Learning rate for Adam optimizer')
@click.option('--run_number', default=1, help='ith run of the day')
def main(features_path, targets_path, batch_size, epochs, lr, run_number):
    today = str(date.today())
    run_number = '_' + str(run_number)
    save_dir = './Run_FCNN_' + today + run_number

    if os.path.exists(save_dir):
        ans = input(
            'The directory this run will write to already exists, would you like to overwrite it? ([y/n])')
        if ans == 'y':
            pass
        else:
            return
    else:
        os.makedirs(save_dir)
    write_info_file(save_dir, features_path, targets_path, batch_size, epochs, lr, run_number)

    model = FCNN_model.build_model()
    # write .txt file with model summary
    filename = os.path.join(save_dir, 'modelsummary.txt')
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            model.summary()

    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

    features = np.load(features_path)
    print(np.max(features))
    features /= np.max(features)
    print('Loaded features')
    targets = np.load(targets_path)
    assert np.max(targets) == 1
    assert np.min(targets) == 0

    split = round(0.8 * len(features))
    train_features = features[:split]
    val_features = features[split:]

    train_targets = targets[:split]
    val_targets = targets[split:]
    print('Loaded Targets')

    metrics = Metrics(val_features, val_targets)
    checkpoint_path = os.path.join(save_dir, "checkpoints/cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=5)
    history = model.fit(train_features,
                        train_targets,
                        # validation_split=0.2,
                        validation_data=(val_features, val_targets),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[cp_callback, metrics])

    loss = pd.Series(history.history['loss'])
    val_loss = pd.Series(history.history['val_loss'])
    loss_df = pd.DataFrame({'Training Loss': loss,
                            'Val Loss': val_loss})
    filename = os.path.join(save_dir, 'losses.csv')
    loss_df.to_csv(filename)  # save losses for further plotting/analysis
    plot_loss(loss, val_loss, save_dir)


if __name__ == '__main__':
    main()
