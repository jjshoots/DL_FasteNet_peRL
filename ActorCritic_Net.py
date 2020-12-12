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

        self.blank_slate = torch.ones([1, 1, 16, 16]).to(device)
        self.blank_slate = torch.cat([self.blank_slate, self.blank_slate], dim=1)

        location_vector = torch.range(-1., 1., (2 / 15))
        location_encoding_x = torch.stack([location_vector for _ in range(16)])
        location_encoding_y = torch.transpose(location_encoding_x, 0, 1)
        self.location_encoding = torch.stack([location_encoding_x, location_encoding_y]).unsqueeze(0).to(device)
        self.location_encoding = torch.cat([self.location_encoding for _ in range(35)], dim=0)

        # actor outputs two values representing y locations to perform crop
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=66, out_channels=32, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1),
            nn.AvgPool2d(kernel_size=4),
            nn.Sigmoid(),
        )

        # actor outputs an estimated loss function based on the critic actions
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=68, out_channels=32, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            nn.AvgPool2d(kernel_size=4)
        )



    # get actor output
    def take_action(self, input):
        input = torch.cat([self.location_encoding, input], dim=1)
        return self.actor(input)



    # get estimated reward
    def estimate_reward(self, input, actions):
        action_planes = self.blank_slate * actions

        input = torch.cat((self.location_encoding, input, action_planes), dim=1)
        
        return self.critic(input)



    # perform action and criticism at the same time
    def forward(self, input):
        actions = self.take_action(input)
        estimated_reward = self.estimate_reward(input, actions)

        return actions, estimated_reward