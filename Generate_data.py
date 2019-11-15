"""
This script takes point clouds of simulated particle tracks in a .h5 format, and then
generates .npy files containing the training data to be used in the FCNN and CNN
"""
# import packages
import h5py
import click
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# set number of pads
NUM_PADS = 10240


def create_array_indices(overbiased_pads):
    """
    Create 3 arrays that assist with creating the feature and target arrays

    :param overbiased_pads: an array showing the overbiased pads (a 1 at a given row index means that pad was
    overbiased, a 0 means it was not)
    :return feature_pads: for a given feature array, gives the pad number that goes in each spot
    :return target_pads : for a given target array, gives the pad number that goes in each spot
    :return indices: indices[pad - 1] gives the index of either feature_pads or target_pads that corresponds to a given
    pad number
    """
    indices = np.zeros((NUM_PADS))
    feature_pads = np.zeros((int(NUM_PADS - sum(overbiased_pads))))
    target_pads = np.zeros((int(sum(overbiased_pads))))
    count_features = 0
    count_targets = 0
    for i in range(len(overbiased_pads)):
        if overbiased_pads[i] > 0:
            indices[i] = int(count_targets)
            target_pads[count_targets] = i + 1
            count_targets += 1
        else:
            indices[i] = int(count_features)
            feature_pads[count_features] = i + 1
            count_features += 1

    return feature_pads, target_pads, indices


def random_list(length):
    r = list(range(length))
    random.shuffle(r)
    return r


def plot_track(xs, ys, zs):
    """
    Makes a plot with subplots of the xy and zy projections of a particle track. Then loads the image data into a
    numpy array that can be used for CNN training/testing.
    :param xs: particle x coordinates
    :param ys: particle y coordinates
    :param zs: particle z coordinates
    :return: numpy array containing image data
    """
    fig = plt.figure(figsize=(1, 2), dpi=128, constrained_layout=True)
    gs = fig.add_gridspec(3, 1)
    # xy projection takes up the top 1/3 of the image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(xs, ys, s=1, alpha=1)
    ax1.set_xlim((-280, 280))
    ax1.set_ylim((-280, 280))
    # zy projection takes up bottom 2/3 of the image (note that the z axis is vertical)
    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.set_xlim((-280, 280))
    ax2.set_ylim((0, 1250))
    ax2.scatter(ys, zs, s=1)
    ax1.set_axis_off()
    ax2.set_axis_off()
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
    plt.close()
    return data


@click.command()
@click.argument('lut', type=click.Path(exists=True, readable=True))  # path to csv that gives pad number for
# a given (x,y) (in mm to one decimal) on the AT-TPC detector plane
@click.argument('event_xyz', type=click.Path(exists=True, readable=True))  # path to .h5 file containing point
# clouds of each event
@click.argument('problem_pads', type=click.Path(exists=True, readable=True))  # path to .csv that identifies
#  the overbiased pads
@click.argument('pad_centers', type=click.Path(exists=True, readable=True))  # path to .csv that contains the
# coordinates of the center of each pad
@click.option('--cnn/--no', default=False)  # whether or not to save data for cnn use
@click.option('--z_coord/--activated', default=True)  # whether to associate z-coordinates with each pad
# or simply mark if the pad was activated
@click.argument('suffix')
def main(lut, event_xyz, problem_pads, pad_centers, cnn, z_coord, suffix):
    # load LUT and overbiased pads
    LUT = np.loadtxt(lut, delimiter=',', skiprows=1)  # first row is labels
    overbiased_pads = np.loadtxt(problem_pads, delimiter=',')
    centers = np.loadtxt(pad_centers, delimiter=',', skiprows=1)

    feature_pads, target_pads, indices = create_array_indices(overbiased_pads)

    with h5py.File(event_xyz) as f:
        events = f['simul']
        num_events = len(events)
        if cnn:
            feature_images = np.zeros((num_events, 256, 128))
        if z_coord:
            targets = np.zeros((num_events, int(sum(overbiased_pads)))) - 1
            feature_vectors = np.zeros((num_events, 10240 - int(sum(overbiased_pads)))) - 1
        else:
            targets = np.zeros((num_events, int(sum(overbiased_pads))))
            feature_vectors = np.zeros((num_events, 10240 - int(sum(overbiased_pads))))
        count = 0
        for i in random_list(num_events):
            if count % 100 == 0:
                print(count)
            point_cloud = np.array(events['event{}'.format(i + 1)])
            xs = []
            ys = []
            z_plotting = []
            zs = []
            check_pads = []
            start = 0
            for j in range(len(point_cloud)):
                x = int((point_cloud[j, 0] + 280) * 10)  # convert x from (-280, 280) to (0, 5600)
                y = int((point_cloud[j, 1] + 280) * 10)

                # ignore points outside of chamber's active volume
                if y >= 5600 or x >= 5600:
                    start += 1
                    continue

                z = point_cloud[j, 2]
                pad = int(LUT[x, y])
                # The LUT gives (x,y) points that don't correspond to pad the pad# -1. Ignore these
                if pad < 0:
                    start += 1
                    continue

                if j == start:
                    start = 0
                    zs.append(z)
                    pad_prev = pad

                elif pad == pad_prev:
                    zs.append(z)
                    # handle case where last pad is the same as the one before it
                    if j == len(point_cloud) - 1:
                        z_avg = np.average(zs)
                        index = int(indices[pad_prev - 1])  # -1 because pads start at 1, array starts at 0
                        check_pads.append(pad_prev)
                        if overbiased_pads[pad - 1] > 0:
                            if z_coord:
                                targets[i, index] = z_avg
                            else:
                                targets[i, index] = 1
                        else:
                            feature_vectors[i, index] = z_avg
                            xs.append(centers[pad_prev - 1, 0])
                            ys.append(centers[pad_prev - 1, 1])
                            z_plotting.append(z_avg)
                else:
                    z_avg = np.average(zs)
                    index = int(indices[pad_prev - 1])
                    check_pads.append(pad_prev)
                    # normal cases
                    if overbiased_pads[pad_prev - 1] > 0:
                        if z_coord:
                            targets[i, index] = z_avg
                        else:
                            targets[i, index] = 1

                    else:
                        feature_vectors[i, index] = z_avg
                        xs.append(centers[pad_prev - 1, 0])
                        ys.append(centers[pad_prev - 1, 1])
                        z_plotting.append(z_avg)
                    zs = []
                    zs.append(z)
                    pad_prev = pad
                    # handle case where last pad is different that the one before it
                    if j == len(point_cloud) - 1:
                        check_pads.append(pad_prev)
                        index = int(indices[pad_prev - 1])
                        if overbiased_pads[pad - 1] > 0:
                            if z_coord:
                                targets[i, index] = z_avg
                            else:
                                targets[i, index] = 1
                        else:
                            feature_vectors[i, index] = z_avg
                            xs.append(centers[pad_prev - 1, 0])
                            ys.append(centers[pad_prev - 1, 1])
                            z_plotting.append(z_avg)
            if cnn:
                data = plot_track(xs, ys, z_plotting)
                feature_images[i, :, :] = data[:, :, 0]
            count += 1

    partition = int(0.8 * len(targets))
    train_targets = targets[:partition, :]
    test_targets = targets[partition:, :]

    if cnn:
        train_feature_images = feature_images[:partition, :]
        test_feature_images = feature_images[partition:, :]
        print(train_feature_images.shape)
        print(test_feature_images.shape)
        np.save('train_features' + suffix, train_feature_images)
        np.save('test_features' + suffix, test_feature_images)

    train_feature_vectors = feature_vectors[:partition, :]
    test_feature_vectors = feature_vectors[partition:, :]

    print(train_targets.shape)
    print(test_targets.shape)

    np.save('train_targets' + suffix, train_targets)
    np.save('test_targets' + suffix, test_targets)
    np.save('train_feature_vectors' + suffix, train_feature_vectors)
    np.save('test_feature_vectors' + suffix, test_feature_vectors)


if __name__ == '__main__':
    main()
