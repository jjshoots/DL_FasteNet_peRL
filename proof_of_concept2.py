#!/usr/bin/env python3
import torch
from torch import random
from torch import tensor
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.nn.modules.activation import Sigmoid

import numpy as np

# This is a proof of concept that one step deterministic policy gradient works

class ac(nn.Module):
    def __init__(self):
        super().__init__()

        # tries to predict value of x1 = 1-input and x2 = 1 - input - x1
        self.network = nn.Sequential(
            nn.Linear(1, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 2),
            nn.Tanh(),
        )


    def one(self, x):
        return self.network(x)


ac = ac().to('cuda:0')
optimizer = optim.Adam(ac.parameters(), lr=1e-4, weight_decay=1e-2)

running_error = 0.

for i in range(10000000):

    # get random float
    random_float = np.random.rand(10000, 1)

    # pass random float to actorcritic
    tensor_float = torch.tensor(random_float).float().to('cuda:0')
    output = ac.one(tensor_float)

    output1 = output[:, 0].detach().to('cpu').numpy()
    output2 = output[:, 1]

    # critic prediction target
    # the target cost is 2 - random_float - sum_of_actor_outputs
    target = abs(1 - random_float - output1)

    # optimize
    optimizer.zero_grad()

    cost = abs(torch.tensor(target).to('cuda:0') - output2) + torch.tensor(target).to('cuda:0')
    cost = torch.sum(cost)
    cost.backward()

    optimizer.step()

    running_error += cost

    if i % 5000 == 0:
        print(running_error.item(), target[0, 0])
        running_error = 0.
    