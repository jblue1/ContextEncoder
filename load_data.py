"""
Module with functions to take an .h5 file and load them into a zipped tensorflow dataset object
containing both a dataset with the images needed to be inpainted and a dataset with the real
center regions.
"""

import h5py
import numpy as np
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt


def load_data(data_path, indices_path, num_events):
    with h5py.File(data_path) as f:
        assert num_events <= len(list(f['train'].keys()))
        images = np.empty((num_events, 128, 128, 1), dtype=np.uint8)
        broken_images = np.empty((num_events, 128, 128, 1), dtype=np.uint8)
        indices = np.load(indices_path)
        for count, i in enumerate(indices):
            if count > num_events - 1:
                break
            print(count)
            data = np.asarray(f['train/event{}/data'.format(i)])
            broken_data = np.asarray(f['train/event{}/broken_data'.format(i)])
            x = data[:, 0]
            y = data[:, 1]

            x_b = broken_data[:, 0]
            y_b = broken_data[:, 1]

            fig = plt.figure(figsize=(1, 1), dpi=128)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(-275.0, 275.0)
            ax.set_ylim((-275.0, 275.0))
            ax.set_axis_off()
            ax.scatter(x, y, s=0.1, cmap='Greys')
            fig.canvas.draw()
            plt.close()
            image = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
            image = np.delete(image, 3, axis=2)
            image = image[:, :, 0:1]
            image = ((image * -1) + 255) / 255
            images[count] = image

            fig = plt.figure(figsize=(1, 1), dpi=128)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(-275.0, 275.0)
            ax.set_ylim((-275.0, 275.0))
            ax.set_axis_off()
            ax.scatter(x_b, y_b, s=0.1, cmap='Greys')
            fig.canvas.draw()
            plt.close()
            broken_image = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
            broken_image = np.delete(broken_image, 3, axis=2)
            broken_image = broken_image[:, :, 0:1]
            broken_image = ((broken_image * -1) + 255) / 255
            broken_images[count] = broken_image

        partition = int(len(images) * 0.8)

        val_images = images[partition:, :, :]
        val_broken_images = broken_images[partition:, :, :]

        train_images = images[:partition, :, :]
        train_broken_images = broken_images[:partition, :, :]

        train_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(train_images))
        train_broken_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(train_broken_images))

        val_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(val_images))
        val_broken_images_dataset = tf.data.Dataset.from_tensor_slices(np.float32(val_broken_images))

        train_dataset = tf.data.Dataset.zip((train_broken_images_dataset, train_images_dataset))
        val_dataset = tf.data.Dataset.zip((val_broken_images_dataset, val_images_dataset))

        print(train_dataset)
        print(val_dataset)

def mask_image(image, mask_size, overlap):
    """
    Takes in a numpy representation of an image, and extracts the center
    mask_size x mask_size chunk. Also sets the center region (minus an
    overlap on each side) to (0,0,0).

    image - numpy representation of an image
    mask_size - size of chunk to extract
    """
    height, width, channels = image.shape
    start_index = int(height - mask_size * 1.5)
    end_index = int(start_index + mask_size)
    center = image[start_index:end_index, start_index:end_index, :]
    fill = np.zeros([mask_size-overlap*2, mask_size-overlap*2, channels])
    masked_image = np.copy(image)
    masked_image[start_index + overlap:end_index-overlap, start_index+overlap:end_index-overlap, :] = fill

    return center, masked_image


def load_h5_to_dataset(file_path, overlap, shuffle, height=128, width=128, num_channels=3):
    """
    Takes an .h5 file of images and converts in to a tensorflow dataset object.

    file_path - path to .h5 file with images
    """
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
