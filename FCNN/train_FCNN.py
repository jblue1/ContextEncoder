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
from contextlib import redirect_stdout


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


def write_info_file(save_dir, data_path, batch_size, epochs, lr, run_number):
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
                 'Training data found at: {} \n'.format(data_path),
                 'Batch Size: {} \n'.format(batch_size),
                 'Epochs: {} \n'.format(epochs),
                 'Learning Rate: {} \n'.format(lr)]

    with open(filename, 'w') as f:
        f.writelines(info_list)


@click.command()
@click.argument('data_path', type=click.Path(exists=True, readable=True))
@click.argument('indices_path', type=click.Path(exists=True, readable=True))
@click.argument('pads_path', type=click.Path(exists=True, readable=True))
@click.option('--batch_size', default=32)
@click.option('--num_events', default=-1)
@click.option('--epochs', default=50)
@click.option('--lr', default=0.001, help='Learning rate for Adam optimizer')
@click.option('--run_number', default=1, help='ith run of the day')
def main(data_path, indices_path, pads_path, batch_size, num_events, epochs, lr, run_number):
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
    write_info_file(save_dir, data_path, batch_size, epochs, lr, run_number)

    model = FCNN_model.build_model()
    # write .txt file with model summary
    filename = os.path.join(save_dir, 'modelsummary.txt')
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            model.summary()

    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='mse')

    features, targets = load_data.load_dataset(data_path,
                                               indices_path,
                                               pads_path,
                                               num_events)

    checkpoint_path = os.path.join(save_dir, "checkpoints/cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=5)
    history = model.fit(features,
                        targets,
                        validation_split=0.2,
                        # validation_data=(val_features, val_targets),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[cp_callback])

    loss = pd.Series(history.history['loss'])
    val_loss = pd.Series(history.history['val_loss'])
    loss_df = pd.DataFrame({'Training Loss': loss,
                            'Val Loss': val_loss})
    filename = os.path.join(save_dir, 'losses.csv')
    loss_df.to_csv(filename)  # save losses for further plotting/analysis
    plot_loss(loss, val_loss, save_dir)


if __name__ == '__main__':
    main()
