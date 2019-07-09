"""
Module with functions to take an .h5 file and load them into a zipped tensorflow dataset object
containing both a dataset with the images needed to be inpainted and a dataset with the real
center regions.
"""

import h5py
import numpy as np
import tensorflow as tf


def load_dataset(data_path, lut_path, height=128, width=175, small=False):
    lut = np.load(lut_path)
    with h5py.File(data_path) as f:
        if small:
            num_events = 100
        else:
            num_events = len(list(f['train'].keys()))
        partition = int(num_events * 0.8)
        images = np.empty((num_events, height, width, 1))
        broken_images = np.empty((num_events, height, width, 1))
        for i in range(num_events):
            image = np.zeros((height, width, 1))
            broken_image = np.zeros((height, width, 1))
            data = np.asarray(f['train/event{}/data'.format(i)])
            broken_data = np.asarray(f['train/event{}/broken_data'.format(i)])

            for j in range(len(data)):
                pad1 = data[j, 4]
                pad2 = broken_data[j, 4]
                if pad1 > 0:
                    index = int(np.where(lut[:, 2] == pad1)[0][0])
                    col = int(lut[index, 0])
                    row = int(lut[index, 1])
                    # normalize z range from -1 to 1
                    z = (data[j, 2] / 1250)
                    assert image[row, col] == 0
                    image[row, col] = z
                if pad2 > 0:
                    index = int(np.where(lut[:, 2] == pad2)[0][0])
                    col = int(lut[index, 0])
                    row = int(lut[index, 1])
                    # normalize z range from 0-1
                    z = (broken_data[j, 2] / 1250)
                    assert broken_image[row, col] == 0
                    broken_image[row, col] = z
            if i % 500 == 0:
                print(i)
            images[i] = image
            broken_images[i] = broken_image
        train_images = images[:partition]
        val_images = images[partition:]
        train_broken_images = broken_images[:partition]
        val_broken_images = broken_images[partition:]

    train_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(train_images))
    train_broken_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(train_broken_images))

    val_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(val_images))
    val_broken_image_dataset = tf.data.Dataset.from_tensor_slices(np.float32(val_broken_images))

    train_dataset = tf.data.Dataset.zip((train_broken_images_dataset, train_images_dataset))
    val_dataset = tf.data.Dataset.zip((val_broken_image_dataset, val_images_dataset))

    return train_dataset, val_dataset


def load_simulated_data(file_path):
    """
    Takes an .h5 file containing images of simulated AT-TPC proton and carbon events, and returns a training and
    validation tf dataset object containing the image (128x128x3) and broken image element (32x32x3)
    """
    with h5py.File(file_path) as f:
        images = np.asarray(f['train_images'])
        image_contexts = np.asarray(f['train_image_contexts'])
    assert len(images) == len(image_contexts)

    images = (images - 127.5) / 127.5
    image_contexts = (image_contexts - 127.5) / 127.5

    partition = int(len(images) * 0.8)

    val_images = images[partition:, :, :]
    val_image_contexts = image_contexts[partition:, :, :]

    train_images = images[:partition, :, :]
    train_image_contexts = image_contexts[:partition, :, :]

    train_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(train_images))
    train_image_contexts_dataset = tf.data.Dataset.from_tensor_slices(np.float32(train_image_contexts))

    val_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(val_images))
    val_image_contexts_dataset = tf.data.Dataset.from_tensor_slices(np.float32(val_image_contexts))

    train_dataset = tf.data.Dataset.zip((train_image_contexts_dataset, train_images_dataset))
    val_dataset = tf.data.Dataset.zip((val_image_contexts_dataset, val_images_dataset))

    return train_dataset, val_dataset

