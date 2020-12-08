#!/usr/bin/env python3
import torch
import torch.nn as nn

# actor ctiric module
# expects inputs that are already viewed to 250 long tensor
# runs on cpu (I think it should)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # select device
        device = 'cpu'
        if(torch.cuda.is_available()):
            device = torch.device('cuda:0')

        self.blank_slate = torch.ones([1, 1, 32, 32]).to(device)
        self.blank_slate = torch.cat([self.blank_slate, self.blank_slate], dim=1)

        # actor outputs two values representing y locations to perform crop
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=4),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=4),
            nn.BatchNorm2d(num_features=4),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
            nn.Sigmoid(),
        )

        # actor outputs an estimated loss function based on the critic actions
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=66, out_channels=8, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=4),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=4),
            nn.BatchNorm2d(num_features=4),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1),
            nn.AvgPool2d(kernel_size=2)
        )



    # get actor output
    def take_action(self, input):
        return self.actor(input)



    # get estimated reward
    def estimate_reward(self, input, actions):
        action_planes = self.blank_slate * actions

        input = torch.cat((input, action_planes), 1)
        
        return self.critic(input)



    # perform action and criticism at the same time
    def forward(self, input):
        actions = self.take_action(input)
        estimated_reward = self.estimate_reward(input, actions)

        return actions, estimated_reward