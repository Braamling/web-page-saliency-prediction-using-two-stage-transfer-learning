from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader, accimage_loader, default_loader 
from torchvision import transforms

import os
import numpy as np


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class SaliencyDataset(Dataset):
    def __init__(self, image_dir, heatmap_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.make_dataset(image_dir, heatmap_dir)
        self.img_transform = transforms.Compose([transforms.Resize((224,224), interpolation=2), 
                                                 transforms.ToTensor()])
        self.heatmap_transform = transforms.Compose([transforms.Resize((64,64), interpolation=2), 
                                                 transforms.Grayscale(),
                                                 transforms.ToTensor()])

    def make_dataset(self, image_dir, heatmap_dir):
        images = []

        for fname in os.listdir(image_dir):
            image = os.path.join(image_dir, fname)
            heatmap = os.path.join(heatmap_dir, fname)
            if is_image_file(image) and is_image_file(heatmap):
                item = (image, heatmap)
                images.append(item)

        self.dataset = images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.img_transform(default_loader(self.dataset[idx][0]))
        heatmap = self.heatmap_transform(default_loader(self.dataset[idx][1]))

        sample = (image, heatmap)

        return sample


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.make_dataset(image_dir)
        self.img_transform = transforms.Compose([transforms.Resize((224,224), interpolation=2), 
                                                 transforms.ToTensor()])

    def make_dataset(self, image_dir):
        images = []

        for fname in os.listdir(image_dir):
            image = os.path.join(image_dir, fname)
            if is_image_file(image):
                item = (image, fname)
                images.append(item)

        self.dataset = images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.img_transform(default_loader(self.dataset[idx][0]))

        sample = (image, self.dataset[idx][1])

        return sample