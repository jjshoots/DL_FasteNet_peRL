#!/usr/bin/env python3
import time
import os
import sys

import cv2
from matplotlib import image
import numpy as np
from numpy import random as nprand
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torchvision.transforms.functional as TF

from helpers import helpers
from ActorCritic_Net import ActorCritic
from FasteNet_Net_v2 import FasteNet_v2
from Assistant import Assistant

# params
torch.autograd.set_detect_anomaly(True)
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
MARK_NUMBER = 1

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


##### GENERATE STATES #####
##### GENERATE STATES #####
##### GENERATE STATES #####
# precompute stuff
variance_stack = (torch.ones(1, 2).to(device) * 0.3334)

for epoch in range(10000):
    # generate data stacks
    data_stack = [Assistant.get_cropped_sample() for _ in range(35)]
    image_stack = torch.stack([data_stack[0] for data_stack in data_stack]).to(device).unsqueeze(1)
    label_stack = torch.stack([data_stack[1] for data_stack in data_stack]).to(device).unsqueeze(1)
    
    # pass image stack into FasteNet to generate stack of feature maps
    feature_map_stack = FasteNet.module_one(image_stack)
    feature_map_reshaped = F.adaptive_max_pool2d(feature_map_stack, 16).detach()
    
    # pass feature maps into actor module to get actions distributions and log probabilties
    action_stack = ActorCritic.take_action(feature_map_reshaped).detach()
    
    # generate log probabilities of the actions, in theory they should all be constants
    dist_stack = Normal(action_stack.squeeze(), variance_stack.squeeze())
    logprobs_stack = dist_stack.log_prob(dist_stack.mean).exp()

    # mask off pixels on feature_map_stack
    masked_feature_map = Assistant.crop_feature_map(action_stack, feature_map_stack)
    
    # compute saliency map by passing the masked feature map through FasteNet
    saliency_stack = FasteNet.module_two(masked_feature_map)
    
    # compute the true value by passing saliency map to the contour finding algorithm
    true_value_stack = Assistant.calculate_loss(saliency_stack, label_stack)

    # checkpoint our training
    weights_file = ActorCritic_helper.training_checkpoint(loss=torch.sum(true_value_stack), iterations=999, epoch=epoch)
    if weights_file != -1:
        torch.save(ActorCritic.state_dict(), weights_file)

    # normalize the true values
    true_value_stack = ((true_value_stack - true_value_stack.mean()) / torch.std(true_value_stack)).detach()

    # set to true to visualize
    if False:
        figure = plt.figure()

        figure.add_subplot(5, 1, 1)
        plt.imshow(image_stack[-1].squeeze().to('cpu').numpy())

        figure.add_subplot(5, 1, 2)
        filter_view = feature_map_reshaped[-1, 5, :].contiguous()
        plt.imshow(filter_view.to('cpu').numpy())

        figure.add_subplot(5, 1, 3)
        plt.imshow(saliency_stack[-1].squeeze().to('cpu').numpy())
        plt.title(f'Miscounts: {Assistant.precision_loss[-1]}')
        plt.axhline(y=Assistant.feature_map_start,color='red')
        plt.axhline(y=Assistant.feature_map_end,color='red')

        figure.add_subplot(5, 1, 4)
        plt.imshow(masked_feature_map[-1, 0, :].squeeze().to('cpu').numpy())
        plt.axhline(y=Assistant.feature_map_start,color='red')
        plt.axhline(y=Assistant.feature_map_end,color='red')


        plt.show()

    for iteration in range(100):
        # pass feature maps into actor module to get new actions
        new_action_stack = ActorCritic.take_action(feature_map_reshaped)

        # get estimated values from critic
        estimated_value_stack = ActorCritic.estimate_reward(feature_map_reshaped, new_action_stack).squeeze()
        
        # gather advantages
        advantage_stack = -(true_value_stack - estimated_value_stack).unsqueeze(-1)

        # get new log probs based on actions
        new_dist_stack = Normal(new_action_stack.squeeze(), variance_stack.squeeze())
        new_logprobs_stack = new_dist_stack.log_prob(action_stack.squeeze())

        # get importance sampling ratio
        ratio = (new_logprobs_stack - logprobs_stack).exp()
        # calculate surrogate losses
        surrogate1 = torch.sum(ratio * advantage_stack, dim=1)
        surrogate2 = torch.sum(torch.clamp(ratio, 0.8, 1.2) * advantage_stack, dim=1)

        # distribute losses
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        critic_loss = (true_value_stack - estimated_value_stack).pow(2).mean()

        # sum losses
        overall_loss = 0.5 * critic_loss + actor_loss

        # optimize!
        Actor_optimizer.zero_grad()
        Critic_optimizer.zero_grad()
        overall_loss.backward()
        Actor_optimizer.step()
        Critic_optimizer.step()

    data_stack            = None
    image_stack           = None
    label_stack           = None
    feature_map_stack     = None
    feature_map_reshaped  = None
    action_stack          = None
    dist_stack            = None
    logprobs_stack        = None
    estimated_value_stack = None
    masked_feature_map    = None
    saliency_stack        = None
    true_value_stack      = None

if SHUTDOWN_AFTER_TRAINING:
    os.system("shutdown /s /t 30")
    exit()