import tensorflow as tf
import CNN_model
import click
import os
import load_data
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import pandas as pd


def plot_loss(loss, val_loss, save_dir):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    filename = os.path.join(save_dir, 'Loss_history.png')
    plt.savefig(filename)
    plt.close()


@click.command()
@click.argument('data_path', type=click.Path(exists=True, readable=True))
@click.argument('lut_path', type=click.Path(exists=True, readable=True))
@click.argument('indices_path', type=click.Path(exists=True, readable=True))
@click.option('--num_events', default=-1)
@click.option('--batch_size', default=32)
@click.option('--use_gpu/--no_gpu', default=False)
@click.option('--epochs', default=50)
@click.option('--lr', default=0.01, help='Learning rate for Adam optimizer')
@click.option('--run_number', default=1, help='ith run of the day')
def main(data_path, lut_path, indices_path, num_events, batch_size, use_gpu, epochs, lr, run_number):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    today = str(date.today())
    run_number = '_' + str(run_number)
    save_dir = './Run_' + today + run_number

    if os.path.exists(save_dir):
        ans = input(
            'The directory this run will write to already exists, would you like to overwrite it? ([y/n])')
        if ans == 'y':
            pass
        else:
            return
    else:
        os.makedirs(save_dir)
    MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    model = CNN_model.build_covnet(use_gpu)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=MSE)
    train_broken_data, train_data, val_broken_data, val_data = load_data.load_dataset(data_path,
                                                                                      lut_path,
                                                                                      indices_path,
                                                                                      num_events)
    checkpoint_path = os.path.join(save_dir, "checkpoints/cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=5)
    history = model.fit(train_broken_data,
                        train_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(val_broken_data, val_data),
                        callbacks=[cp_callback])

    loss = pd.Series(history.history['loss'])
    val_loss = pd.Series(history.history['val_loss'])
    loss_df = pd.DataFrame({'Training Loss': loss,
                            'Val Loss': val_loss})
    filename = os.path.join(save_dir, 'losses.csv')
    loss_df.to_csv(filename)
    plot_loss(loss, val_loss, save_dir)


if __name__ == '__main__':
    main()
