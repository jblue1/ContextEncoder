import tensorflow as tf
import CNN_model
import click
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
from sklearn.metrics import precision_score, recall_score
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
        print(' - val_f1: {}  - val_precision: {}  - val_recall: {}'.format(_val_f1, _val_precision, _val_recall))
        print(' ')
        return


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
@click.argument('features_path', type=click.Path(exists=True, readable=True))
@click.argument('targets_path', type=click.Path(exists=True, readable=True))
@click.option('--batch_size', default=32)
@click.option('--use_gpu/--no_gpu', default=False)
@click.option('--epochs', default=50)
@click.option('--lr', default=0.01, help='Learning rate for Adam optimizer')
@click.option('--run_number', default=1, help='ith run of the day')
def main(features_path, targets_path, batch_size, use_gpu, epochs, lr, run_number):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    today = str(date.today())
    run_number = '_' + str(run_number)
    save_dir = './Run_CNN_' + today + run_number

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
    model = CNN_model.build_CNN(use_gpu)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['binary_accuracy'])
    print('Model is Compiled')
    features = np.load(features_path)[:20000]
    print(features.shape)
    features /= np.max(features)
    features = tf.expand_dims(features, -1)
    print(features.shape)
    features = tf.transpose(features, perm=[0, 3, 1, 2])
    print(features.shape)
    print('Loaded features')
    targets = np.load(targets_path)[:20000]
    targets /= np.max(targets)
    print(np.max(targets))
    print('Loaded Targets')

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
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(val_features, val_targets),
                        callbacks=[cp_callback, metrics])

    loss = pd.Series(history.history['loss'])
    val_loss = pd.Series(history.history['val_loss'])
    loss_df = pd.DataFrame({'Training Loss': loss,
                            'Val Loss': val_loss})
    filename = os.path.join(save_dir, 'losses.csv')
    loss_df.to_csv(filename)
    plot_loss(loss, val_loss, save_dir)


if __name__ == '__main__':
    main()
