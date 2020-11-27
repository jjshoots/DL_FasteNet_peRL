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
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
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

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.1),

            # output straight away value, action1, action2
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
            nn.Sigmoid(),
        )

        # critic outputs two values representing y locations to perform crop
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=66, out_channels=32, kernel_size=3, padding=1),
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

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.1),

            # output straight away value, action1, action2
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
            nn.Sigmoid(),
        )

    # perform action and criticism at the same time
    def forward(self, input):
        # print(f'input shape : {input.shape}')

        # blank slate to formulate input to critic
        blank_slate = torch.ones(1, 1, input.shape[2], input.shape[3]).to('cuda:0')
        # print(f'blank_slate shape : {blank_slate.shape}')

        # get action from actor
        action = self.actor(input).squeeze().unsqueeze(-1).unsqueeze(-1)
        # print(f'action shape : {action.shape}')

        # formulate action plane for critic
        action_planes = blank_slate * action
        # print(f'action_planes shape : {action_planes.shape}')

        # stack together to get input to critic
        critic_input = torch.cat([action_planes, input], dim=1)
        # print(f'critic_input shape : {critic_input.shape}')

        # get the value from critic
        value = self.critic(critic_input).squeeze().unsqueeze(-1)
        # print(f'value shape : {value.shape}')

        # stack value and actions together
        action = action.squeeze()
        # print(action.shape)

        return torch.cat([action, value], dim=-1)