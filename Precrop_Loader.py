import time
import os
import sys

import cv2
import numpy as np
from numpy import random as nprand
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from helpers import helpers

# calculates loss, handles image cropping
# feature_quant = feature map downsampling ratio from raw image
# crop_quant = output from fastenet downsampling ratio from raw image

class Precrop_Loader(Dataset):
    def __init__(self, directory, number_of_images, feature_map_quant=8, crop_quant=64, height_quant=6):
        # general params
        self.directory = directory

        self.crop_height = crop_quant * height_quant
        self.height_quant = height_quant
        self.feature_map_quant = feature_map_quant
        self.crop_quant = crop_quant
        self.number_of_images = number_of_images

        # operational params
        self.start_location = 0
        self.end_location = 0

        self.crop_start_location = 0
        self.crop_end_location = 0

        # raw images
        self.image = []
        self.truth = []

        # sampled images
        self.cropped_image = []
        self.cropped_truth = []

        print(f'Precrop Loader initiated: crop_quant={crop_quant}, height quant={height_quant}, pulling images from: {directory}')



    def __len__(self):
        return self.number_of_images * 100



    # returns uncropped image and ground truth
    def get_sample(self):
        index = nprand.randint(0, self.number_of_images)

        image_path = os.path.join(self.directory, f'Dataset/image/image_{index}.png')
        truth_path = os.path.join(self.directory, f'Dataset/label/label_{index}.png')

        # read images
        self.image = TF.to_tensor(cv2.imread(image_path))[0]
        self.truth = TF.to_tensor(cv2.imread(truth_path))[0]
        
        # normalize inputs, 1e-6 for stability as some images don't have truth masks (no fasteners)
        self.image /= torch.max(self.image + 1e-6)
        self.truth /= torch.max(self.truth + 1e-6)



    # returns cropped image and ground truth
    def __getitem__(self, idx):
        self.get_sample()

        # get the starting locations, unscaled and scaled
        self.start_location = nprand.randint(0, self.image.shape[0] - self.crop_height)
        self.end_location = self.crop_height + self.start_location

        # crop ground truth and image
        self.cropped_image = self.image[self.start_location:self.end_location, :1600].unsqueeze(0)
        self.cropped_truth = self.truth[self.start_location:self.end_location, :1600].unsqueeze(0).unsqueeze(0)
        output_size = int(self.cropped_truth.shape[2] / self.feature_map_quant), int(self.cropped_truth.shape[3] / self.feature_map_quant)
        self.cropped_truth = F.interpolate(self.cropped_truth, size=output_size).squeeze(0)

        return self.cropped_image, self.cropped_truth