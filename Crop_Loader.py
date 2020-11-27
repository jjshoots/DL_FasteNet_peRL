import time
import os
import sys

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

class Crop_Loader(Dataset):
    def __init__(self, F_maps, labels, action_p_rewards):
        # general params
        self.F_maps = F_maps
        self.height = self.F_maps[0].shape[1]
        self.action1 = action_p_rewards[..., 0]
        self.action2 = action_p_rewards[..., 1]
        
        # straight pass through
        self.labels = labels
        self.predicted_rewards = action_p_rewards[..., 2]



    def len(self):
        return self.F_maps.shape[0]



    # returns cropped image and ground truth
    def getitem(self, idx):
        # select an image and get the height
        crop_F_map = self.F_maps[idx]
        saliency_mask = torch.ones(1, crop_F_map.shape[1], crop_F_map.shape[2])

        # get the crop limits
        top_border = max(int((self.action1[idx] - 0.5 * self.action2[idx]) * self.height), 0)
        bottom_border = min(int((self.action1[idx] + 0.5 * self.action2[idx]) * self.height), self.height)

        # cropped feature map
        crop_F_map[:, :top_border, :] = 0
        crop_F_map[:, bottom_border:, :] = 0

        # create a saliency mask to delete extra predictions later
        saliency_mask[:, :top_border, :] = 0
        saliency_mask[:, bottom_border:, :] = 0

        return crop_F_map, saliency_mask

    def stack(self):
        crop_F_maps = []
        saliency_masks = []
        
        for idx in range(self.len()):
            temp = self.getitem(idx)
            crop_F_maps.append(temp[0])
            saliency_masks.append(temp[1])

        crop_F_maps = torch.stack(crop_F_maps)
        saliency_masks = torch.stack(saliency_masks)

        return crop_F_maps, self.labels, self.predicted_rewards, self.action2, saliency_masks
