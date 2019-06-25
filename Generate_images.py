import math

import click
import h5py
import matplotlib
import numpy as np
import os
import pandas as pd
import pytpc
from sklearn.utils import shuffle

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _l(a):
    return 0 if a == 0 else math.log10(a)


def simulated(projection, data_dir, save_path, prefix):
    # however many pads we're trying to predict

    print('Processing data...')
    print(data_dir)
    proton_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'proton.h5'), 'r')
    carbon_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'carbon.h5'), 'r')

    # Create empty arrays to hold data
    data = []

    # Add proton events to data array
    for i, event in enumerate(proton_events):
        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                          baseline_correction=False, cg_times=False)
        data.append([xyzs, 0])

        if i % 50 == 0:
            print('Proton event ' + str(i) + ' added.')

    # Add carbon events to data array
    for i, event in enumerate(carbon_events):
        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                          baseline_correction=False, cg_times=False)
        data.append([xyzs, 1])

        if i % 50 == 0:
            print('Carbon event ' + str(i) + ' added.')

    # Take the log of charge data
    log = np.vectorize(_l)
    for event in data:
        event[0][:, 3] = log(event[0][:, 3])

    data = shuffle(data)
    partition = int(len(data) * 0.8)
    train = data[:partition]
    test = data[partition:]

    # Normalize
    max_charge = np.array(list(map(lambda x: x[0][:, 3].max(), data))).max()

    for e in data:
        for point in e[0]:
            point[3] = point[3] / max_charge

    print('Making images...')

    # Make Training sets
    # Make numpy sets
    train_image_contexts = np.zeros((len(train), 128, 128, 3), dtype=np.uint8)
    train_images = np.zeros((len(train), 128, 128, 3), dtype=np.uint8)

    for i, event in enumerate(train):
        e = event[0]
        z = e[:, 1]
        c = e[:, 3]
        if projection == 'zy':
            x = e[:, 2]
        elif projection == 'xy':
            x = e[:, 0]
        else:
            raise ValueError('Invalid projection value.')
        # create lists for missing regions
        x_c = []
        z_c = []
        c_c = []
        for j in range(len(e)):

            # insert deleting condition here
            if not (-10 <= x[j] <= 127.5 and -117.5 <= z[j] <= 20):
                x_c.append(x[j])
                z_c.append(z[j])
                c_c.append(c[j])
                # c[j] = 0

        # make image context
        fig = plt.figure(figsize=(1, 1), dpi=128)
        ax = fig.add_axes([0, 0, 1, 1])
        if projection == 'zy':
            ax.set_xlim(0.0, 1250.0)
        elif projection == 'xy':
            ax.set_xlim(-275.0, 275.0)
        ax.set_ylim((-275.0, 275.0))
        ax.set_axis_off()
        ax.scatter(x_c, z_c, s=0.01, c=c_c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        train_image_contexts[i] = data
        plt.close()

        # make image
        fig = plt.figure(figsize=(1, 1), dpi=128)
        ax = fig.add_axes([0, 0, 1, 1])
        if projection == 'zy':
            ax.set_xlim(0.0, 1250.0)
        elif projection == 'xy':
            ax.set_xlim(-275.0, 275.0)
        ax.set_ylim((-275.0, 275.0))
        ax.set_axis_off()
        ax.scatter(x, z, s=0.01, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        train_images[i] = data
        plt.close()

    # Make Testing sets
    # Make numpy sets
    test_image_contexts = np.zeros((len(test), 128, 128, 3), dtype=np.uint8)
    test_images = np.zeros((len(test), 128, 128, 3), dtype=np.uint8)

    for i, event in enumerate(test):
        e = event[0]
        z = e[:, 1]
        c = e[:, 3]
        if projection == 'zy':
            x = e[:, 2]
        elif projection == 'xy':
            x = e[:, 0]
        else:
            raise ValueError('Invalid projection value.')
        # create lists for missing regions
        x_c = []
        z_c = []
        c_c = []
        for j in range(len(e)):
            # insert deleting condition here
            if not (-10 <= x[j] <= 127.5 and -117.5 <= z[j] <= 20):
                x_c.append(x[j])
                z_c.append(z[j])
                c_c.append(c[j])

        # make image context
        fig = plt.figure(figsize=(1, 1), dpi=128)
        ax = fig.add_axes([0, 0, 1, 1])
        if projection == 'zy':
            ax.set_xlim(0.0, 1250.0)
        elif projection == 'xy':
            ax.set_xlim(-275.0, 275.0)
        ax.set_ylim((-275.0, 275.0))
        ax.set_axis_off()
        ax.scatter(x_c, z_c, s=0.01, c=c_c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        test_image_contexts[i] = data
        plt.close()

        # make image
        fig = plt.figure(figsize=(1, 1), dpi=128)
        ax = fig.add_axes([0, 0, 1, 1])
        if projection == 'zy':
            ax.set_xlim(0.0, 1250.0)
        elif projection == 'xy':
            ax.set_xlim(-275.0, 275.0)
        ax.set_ylim((-275.0, 275.0))
        ax.set_axis_off()
        ax.scatter(x, z, s=0.01, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        test_images[i] = data
        plt.close()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Saving file...')

    filename = os.path.join(save_path, prefix + 'images.h5')
    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('train_image_contexts', data=train_image_contexts)
    h5.create_dataset('train_images', data=train_images)
    h5.create_dataset('test_image_contexts', data=test_image_contexts)
    h5.create_dataset('test_images', data=test_images)

    # h5.create_dataset('max_charge', data=np.array([max_charge]))
    h5.close()


@click.command()
@click.argument('projection', type=click.Choice(['xy', 'zy']), nargs=1)
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.argument('save_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), nargs=1)
@click.option('--num_batches', default=5, help='Number of event batches to load')
def main(projection, data_dir, save_dir, num_batches):
    for i in range(num_batches):
        simulated(projection, data_dir, save_dir, 'batch_{}_'.format(i + 1))


if __name__ == '__main__':
    main()
