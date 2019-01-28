from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
import time
import os
from PIL import Image

from models.vgg16 import vgg16_bn, vgg16
import evaluation.saliconeval.eval as evaluation
from dataIterators import ImageDataset 

import argparse

available_models = {'vgg16': vgg16, 'vgg16_bn': vgg16_bn}

def save_image(data, name, grayscale=False):
    if grayscale:
        # Small negative values will screw up the image, so are clipped to 0
        data = (255.0 / data.max() * (np.clip(data, 0, None))).astype(np.uint8)
    im = Image.fromarray(data)
    if not grayscale:
        im.mode = "RGB"
    im.save(name)

def save_images(data, names, path, grayscale=False):
    for image, name in zip(data, names):
        save_image(image, os.path.join(path, name), grayscale=True)

def infer_model(model, dataloader, use_gpu):
    model.train(False)  # Set model to evaluate mode

    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, names = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # forward
        outputs = model(inputs)

        outputs = outputs.data.cpu().numpy()
        save_images(outputs, names, FLAGS.target_path, True)

def infer():
    image_dataset = ImageDataset(FLAGS.image_path)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=FLAGS.batch_size,
                                             shuffle=False, num_workers=4)

    use_gpu = torch.cuda.is_available()

    model_type = available_models[FLAGS.model_type]
    model = model_type(pretrained=False, state_dict=FLAGS.weights_path)


    if use_gpu:
        model = model.cuda()

    infer_model(model, dataloader, use_gpu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='storage/inference/images',
                        help='The location to store the stage one model weights.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size to use during inference.')
    parser.add_argument('--target_path', type=str, default='storage/inference/output',
                        help='The location to store the stage one model weights.')
    parser.add_argument('--model_type', type=str, default="vgg16",
                        help='The model type to use for inference (vgg16 or vgg16_bn).')
    parser.add_argument('--weights_path', type=str, default='storage/weights/s1_weights.pth',
                        help='The location to store the model weights.')

    FLAGS, unparsed = parser.parse_known_args()
    
    infer()
