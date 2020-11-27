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

        # tries to predict value of x1+x2 = 1-input
        self.actor = nn.Sequential(
            nn.Linear(1, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )

        # tries to predict value of 1 - input1 - input 2
        self.critic = nn.Sequential(
            nn.Linear(2, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1),
            nn.Tanh(),
        )


    def one(self, x):
        return self.actor(x)

    def two(self, x, y):
        return 10 * self.critic(torch.cat([x, y], dim=1).squeeze())


ac = ac().to('cuda:0')
a_optimizer = optim.Adam(ac.actor.parameters(), lr=1e-6, weight_decay=1e-2)
c_optimizer = optim.Adam(ac.critic.parameters(), lr=1e-6, weight_decay=1e-2)

running_error = 0.

for i in range(10000000):

    # get random float
    random_float = np.random.rand(10000, 1)

    # pass random float to actor
    tensor_float = torch.tensor(random_float).float().to('cuda:0')
    output1 = ac.one(tensor_float)

    # recalculate and pass random float to critic
    # to get estimated cost
    output2 = ac.two(tensor_float, output1)

    # critic prediction target
    # the target cost is 2 - random_float - sum_of_actor_outputs
    target = abs(1 - random_float - output1.detach().to('cpu').numpy())

    # optimize
    c_optimizer.zero_grad()
    a_optimizer.zero_grad()

    cost = abs(torch.tensor(target).to('cuda:0') - output2) + torch.tensor(target).to('cuda:0')
    cost = torch.sum(cost)
    cost.backward()

    a_optimizer.step()
    c_optimizer.step()

    running_error += cost

    if i % 5000 == 0:
        print(running_error.item())
        running_error = 0.
    