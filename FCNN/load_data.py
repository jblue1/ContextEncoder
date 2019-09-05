import h5py
import numpy as np


def load_data(num_events):
    train_num = int(num_events * 0.8)
    val_num = num_events - train_num

    train_features = np.zeros((train_num, 5632))
    val_features = np.zeros((val_num, 5632))
    train_targets = np.zeros((train_num, 4608))
    val_targets = np.zeros((val_num, 4608))
    for i in range(5632):
        train_features[:, i] = i / 5632 #+ np.random.rand(train_num, 1)
        val_features[:, i] = i / 5632#+ np.random.rand(val_num, 1)
        if i < 4608:
            train_targets[:, i] = (i**2)/(4608**2) #+ np.random.rand(train_num, 1)
            val_targets[:, i] = (i ** 2) / (4608 ** 2) #+ np.random.rand(val_num, 1)

    print(train_features)
    print(train_targets)
    return train_features, val_features, train_targets, val_targets


def load_dataset(data_path, indices_path, pads_path, num_events):
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
                    targets_list.append(data[j, 0] / 1250)
                else:
                    features_list.append(data[j, 0] / 1250)
            features[count, :] = np.array(features_list)
            targets[count, :] = np.array(targets_list)

    partition = int(0.8 * num_events)
    #train_features = features[:partition, :]
    #val_features = features[partition:, :]

    #train_targets = targets[:partition, :]
    #val_targets = targets[partition:, :]

    #return train_features, val_features, train_targets, val_targets
    return features, targets

