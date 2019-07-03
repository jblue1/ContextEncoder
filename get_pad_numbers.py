"""
Takes simulated mg22 event point clouds, and writes an h5 file containing all of the events.
Structure of h5 file:
               train                       test
          _______|_______ ...         _______|_______...
         |       |       |           |       |       |
       event0  event1   event...    event0  event1  event...
      ___|___
     |       |
   data  broken_data

Data and broken_data contain 10240x5 array with the columns
| x | y | z | c | pad|
where c = 1 if the pad was activated and c = 0 if not. Data contains the full simulated event, while broken data
 has had the problematic pads removed.
"""
import h5py
import numpy as np
import click
from random import shuffle
NUM_PADS = 10240


@click.command()
@click.argument('data_path', type=click.Path(exists=True, readable=True))
@click.argument('lut_path', type=click.Path(exists=True, readable=True))
@click.argument('pad_path', type=click.Path(exists=True, readable=True))
@click.argument('write_path', type=click.Path(exists=False))
def main(data_path, lut_path, pad_path, write_path):
    xy_to_pads_LUT = np.loadtxt(lut_path, delimiter=',')
    prob_pads = np.loadtxt(pad_path, delimiter=',')
    events = []
    with h5py.File(data_path) as f:
        num_events = len(list(f['simul'].keys()))
        print('Will load {} events'.format(num_events))
        for j in range(num_events):
            data = np.zeros((NUM_PADS, 5))
            broken_data = np.zeros((NUM_PADS, 5))
            event = np.asarray(f['simul/event{}'.format(j + 1)])
            for i in range(len(event)):
                x = round(event[i, 0], 1)
                y = round(event[i, 1], 1)
                # x and y range from (-280, 280), need to convert to (0, 5600) to use xy_to_pads_LUT
                col1 = int(round(10 * (x + 280)))
                row1 = int(round(10 * (280 - y) + 1))  # +1 because xy_to_pads_LUT has extra row on top
                # check to make sure the column we found matches the original x
                assert x == np.float32(xy_to_pads_LUT[0, col1])
                pad = xy_to_pads_LUT[row1, col1]

                # only use xyz points that correspond to a pad according to xy_to_pads_LUT
                if pad > -1:
                    data[int(pad), :] = np.array([event[i, 0], event[i, 1], event[i, 2], 1, int(pad)])
                    if prob_pads[int(pad)] == 0:
                        broken_data[int(pad), :] = np.array([event[i, 0], event[i, 1], event[i, 2], 1, int(pad)])
            events.append([data, broken_data])
            if j % 500 == 0:
                print('loaded {} events'.format(j))

    shuffle(events)
    split_index = int(num_events * 0.8)
    train = events[:split_index]
    test = events[split_index:]

    with h5py.File(write_path, 'w') as f:
        train_group = f.create_group('train')
        test_group = f.create_group('test')
        for i in range(len(train)):
            g = train_group.create_group('event{}'.format(i))
            g.create_dataset('data', data=events[i][0])
            g.create_dataset('broken_data', data=events[i][1])

        for i in range(len(test)):
            g = test_group.create_group('event{}'.format(i))
            g.create_dataset('data', data=events[i][0])
            g.create_dataset('broken_data', data=events[i][1])


if __name__ == '__main__':
    main()
