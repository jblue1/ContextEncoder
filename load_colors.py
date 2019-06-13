import numpy as np
import tensorflow as tf
import random


def load_colors(num_images, overlap):
    images = np.empty((num_images, 128, 128, 3), dtype=np.float32)
    centers = np.empty((num_images, 64, 64, 3), dtype=np.float32)
    for i in range(num_images):
        image = np.empty((128, 128, 3))
        center = np.empty((64, 64, 3), dtype=np.float32)
        for j in range(3):
            pixel_value = random.random()
            image[:, :, j].fill(pixel_value)
            center[:, :, j].fill(pixel_value)

        zeros = np.zeros([64-overlap*2, 64-overlap*2, 3])
        image[32 + overlap:96 - overlap, 32 + overlap:96 - overlap, :] = zeros

        images[i, :, :, :] = image
        centers[i, :, :, :] = center

    image_dataset = tf.data.Dataset.from_tensor_slices(np.float32(images))
    center_dataset = tf.data.Dataset.from_tensor_slices(np.float32(centers))

    dataset = tf.data.Dataset.zip((image_dataset, center_dataset))
    return dataset





