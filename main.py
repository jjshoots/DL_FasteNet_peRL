#!/usr/bin/env python3
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
import torch.optim as optim
import torchvision.transforms.functional as TF

from helpers import helpers
from ActorCritic_Net import ActorCritic
from FasteNet_Net_v2 import FasteNet_v2
from Assistant import Assistant

# params
DIRECTORY = 'C:\AI\DATA' # os.path.dirname(__file__)
DIRECTORY2 = 'C:\AI\WEIGHTS'
SHUTDOWN_AFTER_TRAINING = True
NUMBER_OF_IMAGES = 700



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

VERSION_NUMBER = 50
MARK_NUMBER = 20

# instantiate helper object
ActorCritic_helper = helpers(mark_number=MARK_NUMBER, version_number=VERSION_NUMBER, weights_location=DIRECTORY2)

# set up ActorCritic and make sure in inference mode
ActorCritic = ActorCritic().to(device)

# set up loss optimizer
Actor_optimizer = optim.Adam(ActorCritic.actor.parameters(), lr=1e-6, weight_decay=1e-2)
Critic_optimizer = optim.Adam(ActorCritic.critic.parameters(), lr=1e-6, weight_decay=1e-2)

# get latest weight file
weights_file = ActorCritic_helper.get_latest_weight_file()
if weights_file != -1:
    ActorCritic.load_state_dict(torch.load(weights_file))

# set up assistant
Assistant = Assistant(directory=DIRECTORY, number_of_images=700)



##### SET UP TRAINING #####
##### SET UP TRAINING #####
##### SET UP TRAINING #####

for iteration in range(300000):
    # zero the graph gradient
    ActorCritic.zero_grad()

    # get a cropped image sample
    image = Assistant.get_cropped_sample()[0].unsqueeze(0).unsqueeze(0).to(device)
    
    # pass image through first module of FasteNet and get feature map
    F_map_1 = FasteNet.module_one(image)

    # based on feature map, scale to get inputs for AC
    AC_input = F.adaptive_max_pool2d(F_map_1[..., :32], 32)

    # based on feature map get reward and actions
    actions, estimated_reward = ActorCritic.forward(AC_input)

    # crop the feature map
    cropped_F_map = Assistant.crop_feature_map(actions[0].squeeze().item(), actions[1].squeeze().item(), F_map_1).to(device)

    # passed cropped feature map to FasteNet to get saliency map
    saliency_map = FasteNet.module_two(cropped_F_map).to('cpu').squeeze().numpy()

    # calculate 'reward' from saliency map
    Assistant.parse_saliency_map(saliency_map)
    reward = Assistant.calculate_loss()

    # zero the gradients of both actor and critic
    Actor_optimizer.zero_grad()
    Critic_optimizer.zero_grad()

    # calculate loss
    loss = (estimated_reward - reward) ** 2

    # backpropagate loss
    loss.backward()
    Actor_optimizer.step()
    Critic_optimizer.step()

    # checkpoint our training
    weights_file = ActorCritic_helper.training_checkpoint(loss=loss, iterations=iteration, epoch=None)

    if weights_file != -1:
        torch.save(ActorCritic.state_dict(), weights_file)

    if False:

        figure = plt.figure()

        figure.add_subplot(3, 1, 1)
        plt.imshow(image.squeeze().to('cpu').numpy())

        figure.add_subplot(3, 1, 2)
        filter_view = AC_input[:, :1, ...].contiguous().view(32, -1)
        plt.imshow(filter_view.to('cpu').numpy())

        figure.add_subplot(3, 1, 3)
        plt.imshow(Assistant.saliency_map)
        plt.title(f'Miscounts: {Assistant.precision_loss}')
        plt.axhline(y=Assistant.crop_start_location,color='red')
        plt.axhline(y=Assistant.crop_end_location,color='red')

        plt.show()\

if SHUTDOWN_AFTER_TRAINING:
    os.system("shutdown /s /t 30")
    exit()