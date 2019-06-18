'''
Module with functions to take an .h5 file and load them into a zipped tensorflow dataset object
containing both a dataset with the images needed to be inpainted and a dataset with the real
center regions.
'''

import h5py
import numpy as np
import tensorflow as tf
import skimage


'''
Takes in a numpy representation of an image, and extracts the center
mask_size x mask_size chunk. Also sets the center region (minus an 
overlap on each side) to (0,0,0). 

image - numpy representation of an image
mask_size - size of chunk to extract
 
'''


def mask_image(image, mask_size, overlap):
    height, width, channels = image.shape
    start_index = int(height - mask_size * 1.5)
    end_index = int(start_index + mask_size)
    center = image[start_index:end_index, start_index:end_index, :]
    fill = np.zeros([mask_size-overlap*2, mask_size-overlap*2, channels])
    masked_image = np.copy(image)
    masked_image[start_index + overlap:end_index-overlap, start_index+overlap:end_index-overlap, :] = fill

    return center, masked_image


'''
Takes an .h5 file of images and converts in to a tensorflow dataset object.

file_path - path to .h5 file with images 
'''


def load_h5_to_dataset(file_path, overlap, shuffle, height=128, width=128, num_channels=3):
    with h5py.File(file_path) as f:
        list = []
        for i in range(len(f.keys())):
            try:
                name = 'img_{}'.format(i)
                g = np.asarray(f[name])
                list.append(i)
            except KeyError:
                pass
        data_images = np.empty((len(list), height, width, num_channels))
        data_centers = np.empty((len(list), int(height/2), int(width/2), num_channels))
        count = 0
        for i in list:
            # due to problems in download_data.py, for certain values i, img_i might not exist
            name = 'img_{}'.format(i)
            g = np.asarray(f[name])
            if len(g.shape) < 3:
                layer = g
                g = np.empty((layer.shape[0], layer.shape[1], 3))
                for j in range(3):
                    g[:, :, j] = layer
            g = skimage.transform.resize(g, (height, width))
            if g.shape[2] > 3:
                g = g[:, :, 0:2]
                print('Had an image with 4 channels')
            centers, images = mask_image(g, int(height/2), overlap)
            data_images[count, :, :, :] = images * 2 - 1  # normalize pixels to [-1,1]
            data_centers[count, :, :, :] = centers * 2 - 1
            count += 1
            if i % 1000 == 0:
                print('Loaded {} images'.format(i))

        print('Number of images: {}'.format(count))
    image_dataset = tf.data.Dataset.from_tensor_slices(np.float32(data_images))
    center_dataset = tf.data.Dataset.from_tensor_slices(np.float32(data_centers))
    # will do one run with the labels shuffled and then compare with not shuffled, to see if model is learning at all
    if shuffle:
        center_dataset = center_dataset.shuffle(len(list))
    dataset = tf.data.Dataset.zip((image_dataset, center_dataset))

    return dataset


'''
Takes an .h5 file containing images of simulated AT-TPC proton and carbon events, and returns a training and 
validation tf dataset object containing the image (128x128x3) and broken image element (32x32x3)
'''


def load_simulated_data(file_path):
    with h5py.File(file_path) as f:
        train_features = np.asarray(f['train_features'])
        train_features_broken = np.asarray(f['train_features_broken'])
    assert len(train_features) == len(train_features_broken)

    train_features = (train_features - 127.5) / 127.5
    train_features_broken = (train_features_broken - 127.5) / 127.5

    partition = int(len(train_features) * 0.8)

    val_features = train_features[partition:, :, :]
    val_features_broken = train_features_broken[partition:, :, :]

    train_features = train_features[:partition, :, :]
    train_features_broken = train_features_broken[:partition, :, :]

    train_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(train_features))
    train_centers_dataset = tf.data.Dataset.from_tensor_slices(np.float32(train_features_broken))

    val_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(val_features))
    val_centers_dataset = tf.data.Dataset.from_tensor_slices(np.float32(val_features_broken))

    train_dataset = tf.data.Dataset.zip((train_images_dataset, train_centers_dataset))
    val_dataset = tf.data.Dataset.zip((val_images_dataset, val_centers_dataset))

    return train_dataset, val_dataset
