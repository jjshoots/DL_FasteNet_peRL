#!/usr/bin/env python3
import torch
import torch.nn as nn

# weights file is version 5

class FasteNet_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.squeeze = nn.Sequential(
            # input normalization
            nn.BatchNorm2d(num_features=1),

            # hidden layer 1, downscale 2
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.1),

            # hidden layer 2, downscale 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.1),

            # hidden layer 3, downscale 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.hourglass = nn.Sequential(
            # hidden layer 4, downscale 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1),

            # hidden layer 5, downscale 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1),

            # hidden layer 6, downscale 2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1),

            # hidden layer 7, no downscale
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1),

            # hidden layer 8, upscale 2
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1),

            # hidden layer 9, upscale 2
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1),

            # hidden layer 10, upscale 2
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.output_layer = nn.Sequential(
            # output layer, no downscale
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # threshold layer to crush outputs during inference, don't do this during training
        self.threshold_layer = nn.Threshold(0.8, value=0)



    def module_one(self, input):
        return self.squeeze(input)



    def module_two(self, input1):
        input2 = self.hourglass(input1)
        c = self.output_layer(torch.cat([input1, input2], 1))

        if not self.training:
            c = self.threshold_layer(c)
            c = torch.ceil(c)
            
        return c



    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

            assert(not param.requires_grad)
            