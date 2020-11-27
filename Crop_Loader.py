import time
import os
import sys
import pickle

import cv2
import numpy as np
from numpy import random as nprand
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from helpers import helpers

# calculates loss, handles image cropping
# feature_quant = feature map downsampling ratio from raw image
# crop_quant = output from fastenet downsampling ratio from raw image

class Crop_Loader():
    def __init__(self, dump_location, length):
        # general params
        self.dump_location = dump_location
        self.length = length

        self.height = 0



    def __len__(self):
        return self.length



    # returns cropped image and ground truth
    def __getitem__(self, idx):
        # read from dump location
        f = open(os.path.join(self.dump_location, f'example{idx}.pckl'), 'rb')
        crop_F_map, labels, action_p_rewards = pickle.load(f)

        # parse out action and values and get image height
        action1 = action_p_rewards[..., 0]
        action2 = action_p_rewards[..., 1]
        predicted_rewards = action_p_rewards[..., 2]
        self.height = crop_F_map.shape[1]

        # create saliency mask
        saliency_mask = torch.ones(1, crop_F_map.shape[1], crop_F_map.shape[2])

        # get the crop limits
        top_border = max(int((action1 - 0.5 * action2) * self.height), 0)
        bottom_border = min(int((action1 + 0.5 * action2) * self.height), self.height)

        # cropped feature map
        crop_F_map[:, :top_border, :] = 0
        crop_F_map[:, bottom_border:, :] = 0

        # create a saliency mask to delete extra predictions later
        saliency_mask[:, :top_border, :] = 0
        saliency_mask[:, bottom_border:, :] = 0

        return crop_F_map, labels, predicted_rewards, action2, saliency_mask

