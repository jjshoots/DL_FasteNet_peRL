#!/usr/bin/env python3
import torch
from torch import random
from torch import tensor
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.nn.modules.activation import Sigmoid
import torch.random as random

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

for i in range(2000000):
    if i < 1000000:
        # pass random float to actor
        tensor_float = torch.rand(100000, 1).to('cuda:0')
        output1 = ac.one(tensor_float)

        # recalculate and pass random float to critic to get estimated cost
        output2 = ac.two(tensor_float, output1)
        target = abs(1 - tensor_float.detach().to('cpu').numpy() - output1.detach().to('cpu').numpy())

        # optimize critic against error in prediction value
        c_optimizer.zero_grad()
        a_optimizer.zero_grad()
        cost = abs(torch.tensor(target).to('cuda:0') - output2)
        cost = torch.sum(cost)
        cost.backward()
        # print(list(ac.critic.parameters())[0].grad)
        c_optimizer.step()

        running_error += cost

        if i % 5000 == 0:
            print(running_error.item())
            running_error = 0.

    if i > 1000000:
        # pass random float to actor
        tensor_float = torch.rand(100000, 1).to('cuda:0')
        output1 = ac.one(tensor_float)

        # critic prediction target
        # the target cost is 1 - random_float - sum_of_actor_outputs
        target = abs(1 - tensor_float.detach().to('cpu').numpy() - output1.detach().to('cpu').numpy())

        # optimize actor against cost of action
        c_optimizer.zero_grad()
        a_optimizer.zero_grad()
        output2 = torch.sum(abs(ac.two(tensor_float, output1)))
        output2.backward()
        # print(list(ac.actor.parameters())[0].grad)
        a_optimizer.step()

        running_error += output2

        if i % 5000 == 0:
            print('a', running_error.item())
            running_error = 0.
    

for i in range(10):
    # pass random float to actor
    tensor_float = torch.rand(1, 1).to('cuda:0')
    output1 = ac.one(tensor_float)

    # recalculate and pass random float to critic to get estimated cost
    output2 = ac.two(tensor_float, output1)

    print(f'The remainder of 1 minus {tensor_float.item()} is {output1.item()} and the cost is {output2.item()}')