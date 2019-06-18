'''
Downloads the desired number of images from the ImageNet database using a .txt file
with the urls of the images listed. Stores a testing set and a training set of these
images as numpy arrays in an .h5 file. Can be run from command line with click.
'''

import click
import h5py
from skimage import io
import os
import numpy as np
import urllib

'''
num_images - number of images to download
read_path - path to .txt file provided by ImageNet
returns - two lists with each entry being a url of an image, one for training
and the other for testing
'''


def choose_images(num_images, read_path, train_percent=0.8, val_percent=0.8):
    urls = []
    with open(read_path) as myfile:
        count = 0
        while count < num_images:
            x = next(myfile)
            if 'flickr' in x:
                count += 1
                url = x[x.find('h'):-1]
                urls.append(url)

    np.random.shuffle(urls)
    sep_index1 = int(num_images * train_percent)
    sep_index2 = int(sep_index1 * val_percent)

    return urls[:sep_index2], urls[sep_index2: sep_index1], urls[sep_index1:]


'''
Writes the images as a series of numpy arrays to an .h5 file

urls - list of urls
write_path - directory to write the .h5 file to
prefix - desired filename
'''


def image_to_h5(urls, write_path, prefix):
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    filename = os.path.join(write_path, prefix + '.h5')

    with h5py.File(filename, "w") as f:
        for i in range(len(urls)):
            # some urls don't have images anymore and need to be skipped
            try:
                image = io.imread(urls[i])
                if i % 100 == 0:
                    print('Reached image {}'.format(i))
                name = 'img_{}'.format(i)
                f.create_dataset(name, data=image)
            except (urllib.error.URLError, ValueError):
                print('Could not download image {}'.format(i))


@click.command()
@click.option('--num_images', default=10000, help='Number of images to download')
@click.argument('read_path', type=click.Path(exists=True, readable=True))
@click.argument('write_path', type=click.Path(exists=False))
@click.argument('prefix')
def main(num_images, read_path, write_path, prefix):
    urls_train, urls_val, urls_test = choose_images(num_images, read_path)
    image_to_h5(urls_train, write_path, prefix + '_training')
    image_to_h5(urls_val, write_path, prefix + '_validation')
    image_to_h5(urls_test, write_path, prefix + '_testing')


if __name__ == '__main__':
    main()
    print('done')
