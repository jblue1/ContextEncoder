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
            try:
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

            except KeyError:
                pass

    image_dataset = tf.data.Dataset.from_tensor_slices(np.float32(data_images))
    center_dataset = tf.data.Dataset.from_tensor_slices(np.float32(data_centers))
    # will do one run with the labels shuffled and then compare with not shuffled, to see if model is learning at all
    if shuffle:
        center_dataset = center_dataset.shuffle(len(list))
    dataset = tf.data.Dataset.zip((image_dataset, center_dataset))

    return dataset
