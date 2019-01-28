"""
This file loads the Salicon annotations into memory and creates fixation images.
The same type of images is created for the fiwi dataset and placed into a train/validation split.
These images can be used for more efficient training.
"""

import argparse
import numpy as np
import os
import random
import math

import scipy.ndimage as ndimage
from os.path import isfile, join

from salicon.salicon import SALICON
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array

def image_to_heatmap(path):
    img = load_img(path, grayscale=True)

    img = np.squeeze(np.asarray(img_to_array(img), dtype=float))

    sal_map = ndimage.filters.gaussian_filter(img, FLAGS.sigma)
    sal_map -= np.min(sal_map)
    sal_map = sal_map / np.max(sal_map)

    im = Image.fromarray(np.uint8(sal_map*255))

    return im

def convert_salicon(annotatation_path, save_path):
    
    salicon = SALICON(annotatation_path)
    imgIds = salicon.getImgIds();

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load each image, create heatmap and store to disk.
    for i, imgId in enumerate(imgIds):
        img = salicon.loadImgs(imgId)[0]
        if i % 1000 is 0:
            print(i, img['file_name'])
        annIds = salicon.getAnnIds(imgIds=img['id'])
        anns = salicon.loadAnns(annIds)
        sal_map = salicon.buildFixMap(anns, doBlur=True, sigma=FLAGS.sigma)
        im = Image.fromarray(np.uint8(sal_map*255))

        im.save(save_path + img['file_name'])

    # Cleanup, just to be sure. 
    del salicon

"""
Make the train and val split for the FiWi dataset and add a gauss blur to the 
fixation points.
"""
def convert_fiwi():
    stimuli = join(FLAGS.fiwi_data_path, 'eyeMaps/all5/')
    files = [f for f in os.listdir(stimuli) if isfile(join(stimuli, f))]

    random.shuffle(files)

    train = math.ceil(FLAGS.fiwi_train_split * len(files))
    # val = math.floor((1 - FLAGS.fiwi_train_split) * len(files))

    train_files = files[:train]
    val_files = files[train:]

    # Create heatmap folder
    if not os.path.exists(join(FLAGS.fiwi_data_path, 'heatmaps')):
        os.makedirs(join(FLAGS.fiwi_data_path, 'heatmaps'))

    if not os.path.exists(join(FLAGS.fiwi_data_path, 'heatmaps/train')):
        os.makedirs(join(FLAGS.fiwi_data_path, 'heatmaps/train'))

    if not os.path.exists(join(FLAGS.fiwi_data_path, 'heatmaps/val')):
        os.makedirs(join(FLAGS.fiwi_data_path, 'heatmaps/val'))

    # Create image folder
    if not os.path.exists(join(FLAGS.fiwi_data_path, 'images')):
        os.makedirs(join(FLAGS.fiwi_data_path, 'images'))

    if not os.path.exists(join(FLAGS.fiwi_data_path, 'images/train')):
        os.makedirs(join(FLAGS.fiwi_data_path, 'images/train'))

    if not os.path.exists(join(FLAGS.fiwi_data_path, 'images/val')):
        os.makedirs(join(FLAGS.fiwi_data_path, 'images/val'))
 
    for file in train_files:
        print(FLAGS.fiwi_data_path + 'eyeMaps/all5/' + file)
        heatmap = image_to_heatmap(FLAGS.fiwi_data_path + 'eyeMaps/all5/' + file)
        img = load_img(FLAGS.fiwi_data_path + 'stimuli/' + file, grayscale=False)

        heatmap.save(FLAGS.fiwi_data_path + 'heatmaps/train/' + file)
        img.save(FLAGS.fiwi_data_path + 'images/train/' + file)

    for file in val_files:
        print(FLAGS.fiwi_data_path + 'eyeMaps/all5/' + file)
        heatmap = image_to_heatmap(FLAGS.fiwi_data_path + 'eyeMaps/all5/' + file)
        img = load_img(FLAGS.fiwi_data_path + 'stimuli/' + file, grayscale=False)

        heatmap.save(FLAGS.fiwi_data_path + 'heatmaps/val/' + file)
        img.save(FLAGS.fiwi_data_path + 'images/val/' + file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--salicon_train_annotation_path', type=str, default='storage/salicon/fixations_train2014.json',
                        help='The location of the salicon annotatation data for training.')
    parser.add_argument('--salicon_train_save_path', type=str, default='storage/salicon/heatmaps/train/',
                        help='The location to store the fixations for training.')

    parser.add_argument('--salicon_val_annotation_path', type=str, default='storage/salicon/fixations_val2014.json',
                        help='The location of the salicon annotatation data for validation.')
    parser.add_argument('--salicon_val_save_path', type=str, default='storage/salicon/heatmaps/val/',
                        help='The location to store the fixations for validation.')

    parser.add_argument('--fiwi_data_path', type=str, default='storage/FiWi/',
                        help='The root location for the fiwi dataset')
    parser.add_argument('--fiwi_train_split', type=int, default=.8)

    parser.add_argument('--sigma', type=int, default=19,
                        help='The sigma for the gaussain blur to be applied on each fixation map')

    FLAGS, unparsed = parser.parse_known_args()
    
    convert_salicon(FLAGS.salicon_val_annotation_path, FLAGS.salicon_val_save_path)
    convert_salicon(FLAGS.salicon_train_annotation_path, FLAGS.salicon_train_save_path)

    convert_fiwi()
