import h5py
import numpy as np


def load_dataset(data_path, lut_path, indices_path, num_events, height=128, width=175):
    """
    Creates training and validation datasets to be used to train CNN
    :param str data_path: Path to .h5 file containing the events (see get_pad_numbers.py)
    :param str lut_path: Path to the look up table containing the designated row and column of each pad
    :param str indices_path: Path the a .npy file containing the indices of events with a non-zero number of points
    after being broken
    :param int num_events: Number of events to use (-1 means use all events)
    :param int height: image height
    :param int width: image width
    :return: Four numpy arrays, one containing the training broken images, one containing the validation broken
    images, one containing the training target data, one containing the validation target data
    """
    lut = np.load(lut_path)
    indices = np.load(indices_path)
    if num_events < 0:
        num_events = len(indices)
    data = np.zeros((num_events, 10240), dtype=np.float32)
    broken_images = np.empty((num_events, height, width, 1))
    with h5py.File(data_path) as f:
        for count, i in enumerate(indices):
            if count > num_events - 1:
                break
            if count % 500 == 0:
                print(count)
            data[count] = np.asarray(f['train/event{}/data'.format(i)])[:, 2]
            broken_image = np.zeros((height, width, 1))
            broken_data = np.asarray(f['train/event{}/broken_data'.format(i)])
            for j in range(len(broken_data)):
                pad = broken_data[j, 4]
                if pad > 0:
                    index = int(np.where(lut[:, 2] == pad)[0][0])
                    col = int(lut[index, 0])
                    row = int(lut[index, 1])
                    # normalize z range from 0-1
                    z = ((broken_data[j, 2] * -1) + 1250) / 1250
                    assert broken_image[row, col] == 0
                    broken_image[row, col] = z
            broken_images[count] = broken_image

    partition = int(0.8 * num_events)
    train_data= data[:partition]
    train_broken_images = broken_images[:partition]

    val_data = data[partition:]
    val_broken_images = broken_images[partition:]

    return train_broken_images, train_data, val_broken_images, val_data
