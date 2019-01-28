from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
from PIL import Image

from models.vgg16 import vgg16_bn, vgg16
import evaluation.saliconeval.eval as evaluation
from dataIterators import SaliencyDataset

import argparse
import tensorboard_logger as tfl

available_models = {'vgg16': vgg16, 'vgg16_bn': vgg16_bn}

def save_image(data, name, grayscale=False):
    # image = data.data.cpu().numpy()[0]
    if not grayscale:
        data = data.mean(axis=0)
    # if grayscale:
    data = (255.0 / data.max() * (np.clip(data, 0, None))).astype(np.uint8)
    im = Image.fromarray(data)
        # im.mode = "RGB"
    im.save(name)

"""
This method prepares the dataloaders for training and returns a training/validation dataloader.
"""
def prepare_dataloaders(image_path, heatmap_path, batch_size):
    # Get the train/val datasets
    image_datasets = {x: SaliencyDataset(os.path.join(image_path, x), 
                                     os.path.join(heatmap_path, x)) for x in ['train', 'val']}
    # Prepare the loaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Get the datasizes for logging purposes.
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders

"""
Prepare the model with the correct weights and format the the configured use.
"""
def prepare_model(phase=1, use_scheduler=True):
    use_gpu = torch.cuda.is_available()

    model_type = available_models[FLAGS.model_type]

    if FLAGS.from_weights is not None:
        model = model_type(pretrained=False, state_dict=FLAGS.from_weights)
    else:
        model = model_type(pretrained=True)

    for param in model.features.parameters():
            param.requires_grad = False

    opt_parameters = model.classifier.parameters()

    if phase > 1:
        for param in model.classifier.parameters():
            param.requires_grad = False
        for param in model.classifier._modules['6'].parameters():
            param.requires_grad = True

        opt_parameters = model.classifier._modules['6'].parameters()

    if use_gpu:
        model = model.cuda()


    optimizer = optim.Adam(opt_parameters, lr=FLAGS.learning_rate, weight_decay=1e-5)
    # optimizer = optim.SGD(opt_parameters, lr=0.01, momentum=0.9)

    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, optimizer, scheduler, use_gpu

"""
Handle the storage and removal of checkpoints.
"""
def handle_checkpoints(model, epoch, meta_data=None, keep=3):
    checkpoint_info = "{}_epoch_{}_{}".format(FLAGS.description, epoch, meta_data)
    remove_checkpoint = "{}_epoch_{}_{}".format(FLAGS.description, epoch - keep, meta_data)

    try:
        os.remove(FLAGS.checkpoint.format(remove_checkpoint))
    except OSError:
        pass

    torch.save(model.state_dict(), FLAGS.checkpoint.format(checkpoint_info))

def train_model(model, criterion, dataloaders, use_gpu, optimizer, scheduler, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_sauc = 0.0

    tfl.configure(FLAGS.log_dir.format(FLAGS.description))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            cc_score = 0.0
            nss_score = 0.0
            auc_score = 0.0
            sauc_score = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # Get the inputs and wrap them into varaibles
                if use_gpu:
                    inputs, labels = Variable(data[0].cuda()), Variable(data[1].cuda())
                else:
                    inputs, labels = Variable(data[0]), Variable(data[1])

                # Do the forward prop.
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Print the loss
                print('{} Loss: {}'.format(phase, loss.data[0]))

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += 1
                labels = labels.data.cpu().numpy()
                outputs = outputs.data.cpu().numpy()

                # Do the evaluation
                scores = evaluation.compute_scores(labels, outputs)
                cc_score += scores['cc']
                auc_score += scores['auc']
                sauc_score += scores['sauc']
                nss_score += scores['nss']


            running_loss = running_loss / running_corrects
            cc_score = cc_score / running_corrects
            auc_score = auc_score / running_corrects
            sauc_score = sauc_score / running_corrects
            nss_score = nss_score / running_corrects
            # Print evaluation scores
            tfl.log_value('{}_loss'.format(phase), running_loss, epoch)
            tfl.log_value('{}_cc'.format(phase), cc_score, epoch)
            tfl.log_value('{}_auc'.format(phase), auc_score, epoch)
            tfl.log_value('{}_sauc'.format(phase), sauc_score, epoch)
            tfl.log_value('{}_nss'.format(phase), nss_score, epoch)

        # outputs = model(inputs)
        save_image(inputs.data.cpu().numpy()[0], "{}{}_epoch_{}_input.png".format(FLAGS.tmp_dir, 
                                                                                  FLAGS.description,
                                                                                  epoch))
        save_image(outputs[0], "{}{}_epoch_{}_output.png".format(FLAGS.tmp_dir, 
                                                                 FLAGS.description,
                                                                 epoch), True)
        save_image(labels[0][0], "{}{}_epoch_{}_label.png".format(FLAGS.tmp_dir, 
                                                                  FLAGS.description,
                                                                  epoch), True)

        handle_checkpoints(model, epoch, phase, keep=3)

        if sauc_score / running_corrects > best_sauc:
            best_sauc = sauc_score
            best_model_wts = copy.deepcopy(model.state_dict())

    # load and save best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), FLAGS.weights_path.format(FLAGS.description))
    return model

def train():
    dataloaders = prepare_dataloaders(FLAGS.image_path, FLAGS.heatmap_path, FLAGS.batch_size)

    model, optimizer, scheduler, use_gpu = prepare_model(phase=FLAGS.phase)

    model = train_model(model, nn.MSELoss(), dataloaders, use_gpu, optimizer, scheduler,
                       num_epochs=FLAGS.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--heatmap_path', type=str, default='storage/salicon/heatmaps/',
                        help='The location of the salicon heatmaps data for training.')
    parser.add_argument('--image_path', type=str, default='storage/salicon/images/',
                        help='The location of the salicon images for training.')

    parser.add_argument('--weights_path', type=str, default='storage/weights/{}.pth',
                        help='The location to store the model weights.')
    parser.add_argument('--checkpoint', type=str, default='storage/weights/{}_checkpoint.pth',
                        help='The location to store the model intermediate checkpoint weights.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size used for training.')
    parser.add_argument('--model_type', type=str, default="vgg16",
                        help='The pretrained vgg model to start from. (if training from loaded weights, the same models has to be used.).')
    parser.add_argument('--epochs', type=int, default=10,
                        help='The amount of epochs used to train.')
    parser.add_argument('--from_weights', type=str, default=None,
                        help='The model to start training from, if None it will start from scratch (pretrained vgg). ')
    parser.add_argument('--log_dir', type=str, default='storage/logs/{}',
                        help='The location to place the tensorboard logs.')
    parser.add_argument('--tmp_dir', type=str, default='storage/tmp/',
                        help='The location to place temporary files that are generated during training.')
    parser.add_argument('--phase', type=int, default=1,
                        help='The transfer learning phase to start')
    parser.add_argument('--description', type=str, default='example_run',
                        help='The description of the run, for logging, output and weights naming.')
    parser.add_argument('--learning_rate', type=float, default='0.0001',
                        help='The learning rate to use for the experiment')

    FLAGS, unparsed = parser.parse_known_args()

    train()
