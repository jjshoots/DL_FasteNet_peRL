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
import torchvision.transforms.functional as TF

from helpers import helpers

# calculates loss, handles image cropping
# feature_quant = feature map downsampling ratio from raw image
# crop_quant = output from fastenet downsampling ratio from raw image

class Assistant():
    def __init__(self, directory, number_of_images, feature_map_quant=8, crop_quant=64, height_quant=6):

        # general params
        self.directory = directory

        self.crop_height = crop_quant * height_quant
        self.height_quant = height_quant
        self.feature_map_quant = feature_map_quant
        self.crop_quant = crop_quant
        self.number_of_images = number_of_images

        print(f'Assistant initiated: crop_quant={crop_quant}, height quant={height_quant}, pulling images from: {directory}')

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

        # cropped feature map
        self.cropped_feature_map = []
        self.saliency_map = []
        self.computational_loss = 0.
        self.precision_loss = 0.



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
    def get_cropped_sample(self):
        self.get_sample()

        # get the starting locations, unscaled and scaled
        self.start_location = nprand.randint(0, self.image.shape[0] - self.crop_height)
        self.end_location = self.crop_height + self.start_location

        # crop ground truth and image
        self.cropped_image = self.image[self.start_location:self.end_location, :1600]
        self.cropped_truth = self.truth[self.start_location:self.end_location, :1600]

        return self.cropped_image, self.cropped_truth



    # crops image from cropped sample based on action values
    def crop_feature_map(self, action, feature_maps):

        action = action.squeeze()
        
        # we compute the computational loss directly using the action
        self.computational_loss = abs(action[:, 1])
        action[:, 1] = action[:, 1] + 0.1
        
        self.precision_loss = torch.zeros_like(action[:, 0])
        self.cropped_feature_map = torch.zeros_like(feature_maps)

        for i, feature_map in enumerate(feature_maps):
            # get crop start location (based on cropped image ground) based on action 1
            self.crop_start_location = max(int((action[i, 0] - 0.5 * action[i, 1]) *  self.height_quant) * self.crop_quant, 0)
            feature_map_start = int(self.crop_start_location / self.feature_map_quant)

            # get crop end location (based on cropped image ground) based on action 2
            self.crop_end_location = min(int((action[i, 0] + 0.5 * action[i, 1]) * self.height_quant) * self.crop_quant, self.crop_height)
            feature_map_end = int(self.crop_end_location / self.feature_map_quant)

            feature_map[..., :feature_map_start, :] = 0
            feature_map[..., feature_map_end:, :] = 0
            self.cropped_feature_map[i] = feature_map

        return self.cropped_feature_map

    

    # upsamples final output feature map
    def parse_saliency_map(self, input):
        # resize the output saliency map
        scaled_saliency_map = cv2.resize(input, (input.shape[1] * self.feature_map_quant, input.shape[0] * self.feature_map_quant), interpolation=cv2.INTER_NEAREST)

        # plot the saliency map onto a map the same size as the cropped truth
        output = np.zeros_like(self.cropped_truth)
        output[self.crop_start_location:self.crop_end_location, :] = scaled_saliency_map

        self.saliency_map = output


    # calculates the loss based on the width of the crop and the accuracy of the net
    def calculate_loss(self, saliency_maps, labels):
        
        for i, saliency_map in enumerate(saliency_maps):
            _, contour_number = helpers.saliency_to_contour(input=saliency_map.unsqueeze(0), original_image=None, fastener_area_threshold=1, input_output_ratio=1)
            _, ground_number = helpers.saliency_to_contour(input=F.max_pool2d(labels[i].unsqueeze(0), 8), original_image=None, fastener_area_threshold=1, input_output_ratio=1)
            
            self.precision_loss[i] = abs(contour_number - ground_number)

        # calculate total loss
        total_loss = self.precision_loss * 1. + self.computational_loss * 10.

        return total_loss


