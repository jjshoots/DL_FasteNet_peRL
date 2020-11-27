#!/usr/bin/env python3
import torch
import torch.nn as nn

# actor ctiric module
# expects inputs that are already viewed to 250 long tensor
# runs on cpu (I think it should)

class ActorCritic_v2(nn.Module):
    def __init__(self):
        super().__init__()

        # actor outputs two values representing y locations to perform crop
        self.actor_critic = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.1),

            # output straight away value, action1, action2
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
            nn.Sigmoid(),
        )

    # perform action and criticism at the same time
    def forward(self, input):
        return self.actor_critic(input).squeeze()