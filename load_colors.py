import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


def load_colors(num_images, overlap):
    images = np.empty((num_images, 128, 128, 3), dtype=np.float32)
    centers = np.empty((num_images, 64, 64, 3), dtype=np.float32)
    for i in range(num_images):
        image = np.empty((128, 128, 3))
        vert_index_1 = random.randrange(0,128)
        vert_index_2 = random.randrange(0, 128)
        horiz_index_1 = random.randrange(0, 128)
        horiz_index_2 = random.randrange(0, 128)

        while vert_index_1 == vert_index_2:
            vert_index_2 = random.randrange(0, 128)
        while horiz_index_1 == horiz_index_2:
            horiz_index_2 = random.randrange(0, 128)

        if vert_index_1 < vert_index_2:
            vert_start_index = vert_index_1
            vert_end_index = vert_index_2
        else:
            vert_start_index = vert_index_2
            vert_end_index = vert_index_1

        if horiz_index_1 < horiz_index_2:
            horiz_start_index = horiz_index_1
            horiz_end_index = horiz_index_2
        else:
            horiz_start_index = horiz_index_2
            horiz_end_index = horiz_index_1

        for j in range(3):
            background_color = random.random()
            line_color = random.random()

            # decides whether the line is vertical or horizontal
            if random.random() < 0.5:
                image[:vert_start_index, :, j].fill(background_color)
                image[vert_start_index:vert_end_index, :, j].fill(line_color)
                image[vert_end_index:, :, j].fill(background_color)
            else:
                image[:, :horiz_start_index, j].fill(background_color)
                image[:, horiz_start_index:horiz_end_index, j].fill(line_color)
                image[:, horiz_end_index:, j].fill(background_color)

        zeros = np.zeros([64-overlap*2, 64-overlap*2, 3])
        center = image[32:96, 32:96, :]
        center = np.copy(center)
        image[32 + overlap:96 - overlap, 32 + overlap:96 - overlap, :] = zeros
        images[i, :, :, :] = image
        centers[i, :, :, :] = center

    image_dataset = tf.data.Dataset.from_tensor_slices(np.float32(images))
    center_dataset = tf.data.Dataset.from_tensor_slices(np.float32(centers))

    dataset = tf.data.Dataset.zip((image_dataset, center_dataset))
    return dataset





