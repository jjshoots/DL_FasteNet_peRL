#!/usr/bin/env python3
from pickle import dump
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
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF

from helpers import helpers
from ActorCritic_Netv2 import ActorCritic_v2
from FasteNet_Net_v2 import FasteNet_v2

from Precrop_Loader import Precrop_Loader
from Crop_Loader import Crop_Loader

# params
DIRECTORY = 'C:\AI\DATA' # os.path.dirname(__file__)
DIRECTORY2 = 'C:\AI\WEIGHTS'
DUMP_LOCATION = os.path.join(os.path.dirname(__file__), 'dump')
SHUTDOWN_AFTER_TRAINING = True
NUMBER_OF_IMAGES = 700
BATCH_SIZE = 2



##### SET UP FASTENET #####
##### SET UP FASTENET #####
##### SET UP FASTENET #####

VERSION_NUMBER = 5
MARK_NUMBER = 506

# instantiate helper object
FasteNet_helper = helpers(mark_number=MARK_NUMBER, version_number=VERSION_NUMBER, weights_location=DIRECTORY2)
device = FasteNet_helper.get_device()

# set up net and make sure in inference mode
FasteNet = FasteNet_v2().to(device)
FasteNet.eval()
FasteNet.freeze_model()

# get latest weight file
weights_file = FasteNet_helper.get_latest_weight_file()
if weights_file != -1:
    FasteNet.load_state_dict(torch.load(weights_file))



##### SET UP ACTORCRITIC #####
##### SET UP ACTORCRITIC #####
##### SET UP ACTORCRITIC #####

VERSION_NUMBER = 51
MARK_NUMBER = 1

# instantiate helper object
ActorCritic_helper = helpers(mark_number=MARK_NUMBER, version_number=VERSION_NUMBER, weights_location=DIRECTORY2)

# set up ActorCritic and make sure in inference mode
ActorCritic = ActorCritic_v2().to(device)

# set up optimizer
optimizer = optim.Adam(ActorCritic.parameters(), lr=1e-6, weight_decay=1e-2)

# get latest weight file
weights_file = ActorCritic_helper.get_latest_weight_file()
if weights_file != -1:
    ActorCritic.load_state_dict(torch.load(weights_file))

# prepare cropped images and truths
# input is raw images, output is data and labels
precrop_set = Precrop_Loader(directory=DIRECTORY, number_of_images=NUMBER_OF_IMAGES)
precrop_loader = torch.utils.data.DataLoader(precrop_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# TRAIN
for epoch in range(4000):
    dump_index = 0
    for j, (data, labels) in enumerate(precrop_loader):
        # pass cropped images and truths to fastenet and actor critic to get value and actions and states
        # input is data and labels, output is actions and feature maps
        fn_output_1 = FasteNet.module_one(data.to(device))
        ac_input = F.adaptive_max_pool2d(fn_output_1[..., :32], 32)
        ac_output = ActorCritic.forward(ac_input)

        for i, _ in enumerate(ac_output):
            f = open(os.path.join(DUMP_LOCATION, f'example{dump_index}.pckl'), 'wb')
            pickle.dump([fn_output_1[i], labels[i], ac_output[i]], f)
            dump_index += 1
    
    # pass feature map and actions to crop handler
    # input is feature maps, labels, actions and rewards, output is cropped feature maps, labels and predicted rewards
    crop_set = Crop_Loader(DUMP_LOCATION, NUMBER_OF_IMAGES-1)
    crop_loader = torch.utils.data.DataLoader(crop_set, batch_size=100, shuffle=False, drop_last=False)
    
    print(f'BEGINNIGN TRAINING EPOCH: {epoch}')

    for iteration, (cropped_F_map, labels, predicted_rewards, crop_width, saliency_mask) in enumerate(crop_loader):
        # input is cropped feature maps, labels, and predicted rewards
        # output is true reward, predicted reward, loss
        saliency_map = FasteNet.module_two(cropped_F_map) * saliency_mask.to(device)
    
        # calculate loss
        precision_loss = torch.sum(abs(saliency_map - labels.to(device)), (-1, -2)).squeeze() * 0.0025
        computational_loss = crop_width
    
        # print(computational_loss)
        # print(precision_loss)
        # print(predicted_rewards)
        loss = torch.sum(abs(predicted_rewards - (precision_loss + computational_loss)) + predicted_rewards)
    
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # checkpoint our training
        weights_file = ActorCritic_helper.training_checkpoint(loss=loss, iterations=iteration, epoch=epoch)
    
        if weights_file != -1:
            torch.save(ActorCritic.state_dict(), weights_file)
    
    
        if False:
    
            figure = plt.figure()
    
            figure.add_subplot(2, 2, 1)
            # plt.imshow(data[0].squeeze().to('cpu').numpy())
    
            figure.add_subplot(2, 2, 2)
            plt.imshow(cropped_F_map[0][0].squeeze().to('cpu').numpy())
    
            figure.add_subplot(2, 2, 3)
            plt.imshow(saliency_map[0].squeeze().to('cpu').numpy())
            # plt.axhline(y=max(int((ac_output[0][0] - 0.5 * ac_output[0][1]) * 48), 0),color='red')
            # plt.axhline(y=min(int((ac_output[0][0] + 0.5 * ac_output[0][1]) * 48), 48),color='red')
    
            figure.add_subplot(2, 2, 4)
            plt.imshow(labels[0].squeeze().to('cpu').numpy())
    
            plt.show()
    

if SHUTDOWN_AFTER_TRAINING:
    os.system("shutdown /s /t 30")
    exit()

exit()


# TRAIN
for epoch in range(40):
    dump_index = 0
    for iteration, (data, labels) in enumerate(precrop_loader):
        # pass cropped images and truths to fastenet and actor critic to get value and actions and states
        # input is data and labels, output is actions and feature maps
        fn_output_1 = FasteNet.module_one(data.to(device))
        ac_input = F.adaptive_max_pool2d(fn_output_1[..., :32], 32)
        ac_output = ActorCritic.forward(ac_input)

        for _ in ac_output:
            f = open(os.path.join(DUMP_LOCATION, f'example{dump_index}.pckl'), 'wb')
            pickle.dump([fn_output_1, labels, ac_output])
    
        # pass feature map and actions to crop handler
        # input is feature maps, labels, actions and rewards, output is cropped feature maps, labels and predicted rewards
        crop_set = Crop_Loader(fn_output_1, labels, ac_output)
        crop_loader = torch.utils.data.DataLoader(crop_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
        # input is cropped feature maps, labels, and predicted rewards
        # output is true reward, predicted reward, loss
        cropped_F_map, labels, predicted_rewards, crop_width, saliency_mask = crop_set.stack()
        saliency_map = FasteNet.module_two(cropped_F_map) * saliency_mask.to(device)
    
        # calculate loss
        precision_loss = torch.sum(abs(saliency_map - labels.to(device)), (-1, -2)).squeeze() * 0.0025
        computational_loss = crop_width
    
        # print(computational_loss)
        # print(precision_loss)
        # print(predicted_rewards)
        loss = torch.sum(abs(predicted_rewards - (precision_loss + computational_loss)) + predicted_rewards)
    
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # checkpoint our training
        weights_file = ActorCritic_helper.training_checkpoint(loss=loss, iterations=iteration, epoch=epoch)
    
        if weights_file != -1:
            torch.save(ActorCritic.state_dict(), weights_file)
    
    
        if False:
    
            figure = plt.figure()
    
            figure.add_subplot(2, 2, 1)
            plt.imshow(data[0].squeeze().to('cpu').numpy())
    
            figure.add_subplot(2, 2, 2)
            plt.imshow(cropped_F_map[0][0].squeeze().to('cpu').numpy())
    
            figure.add_subplot(2, 2, 3)
            plt.imshow(saliency_map[0].squeeze().to('cpu').numpy())
            plt.axhline(y=max(int((ac_output[0][0] - 0.5 * ac_output[0][1]) * 48), 0),color='red')
            plt.axhline(y=min(int((ac_output[0][0] + 0.5 * ac_output[0][1]) * 48), 48),color='red')
    
            figure.add_subplot(2, 2, 4)
            plt.imshow(labels[0].squeeze().to('cpu').numpy())
    
            plt.show()
    

if SHUTDOWN_AFTER_TRAINING:
    os.system("shutdown /s /t 30")
    exit()

exit()