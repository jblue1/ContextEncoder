"""
Function to load data from .h5 file to numpy arrays for training
"""

import h5py
import numpy as np


def load_dataset(data_path, indices_path, pads_path, num_events):
    """
    Loads data from the .h5 file to numpy arrays. The .h5 file contains xyz and pad number for every point
    in the particle track for each event. This function takes each z-coordinate and puts in a vector, one
    of length 5632 (the number of normal pads) and one of length 4608 (the number of overbiased pads). Since
    it's possible to get the pad numbers back from these vectors, any model using this method of loading data
    will be using and predicting the z-coordinate of the response of each pad, if any.
    :param str data_path: path to where .h5 file with the event data is stored
    :param str indices_path: path to file with indices of events that contain >0 points when broken
    :param str pads_path: path to file indicating which pads were over biased
    :param int num_events: number of events to load, any numer < 0 results in all events being loaded
    :return: two arrays, one num_events x 5632 feature array, one num_events x 4608 targets array
    """
    indices = np.load(indices_path)
    if num_events < 0:
        num_events = len(indices)
    features = np.zeros((num_events, 5632), dtype=np.float32)
    targets = np.zeros((num_events, 4608), dtype=np.float32)
    broken_pads = np.loadtxt(pads_path, delimiter=',')
    print(broken_pads.shape)
    print(num_events)
    with h5py.File(data_path) as f:
        for count, i in enumerate(indices):
            if count > num_events - 1:
                print('Done!')
                break
            if count % 500 == 0:
                print(count)
            features_list = []
            targets_list = []
            data = np.zeros((10240, 2), dtype=np.float32)
            data[:, 0] = np.asarray((f['train/event{}/data'.format(i)])[:, 2])
            data[:, 1] = np.asarray((f['train/event{}/data'.format(i)])[:, 4])
            for j in range(10240):
                if broken_pads[j] > 0:
                    targets_list.append(data[j, 0] / 1250)  # change to range 0-1
                else:
                    features_list.append(data[j, 0] / 1250)  # change to range 0-1
            features[count, :] = np.array(features_list)
            targets[count, :] = np.array(targets_list)

    return features, targets

