import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.utils import init

# weight initialization

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                       constant_(x, 0), np.sqrt(2))

class MLP_Actor(nn.Module):
    def __init__(self, obs_shape, hidden_size, dist):
        super(MLP_Actor, self).__init__()
        
        self.hidden_actor = nn.Sequential(
            init_(nn.Linear(obs_shape, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )

        self.dist = dist

    def forward(self, obs):
        hidden_actor = self.hidden_actor(obs)
        return self.dist(hidden_actor)



class MLP_Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size):
        super(MLP_Critic, self).__init__()

        num_inputs = obs_shape + action_shape

        self.q1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )

        self.q2 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )        

        self.hidden_critic = lambda x: (self.q1(x), self.q2(x))

        self.q1_linear = init_(nn.Linear(hidden_size, 1))
        self.q2_linear = init_(nn.Linear(hidden_size, 1))
        

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], 1)

        hidden_critic = self.hidden_critic(x)
        q = (self.q1_linear(hidden_critic[0]), self.q2_linear(hidden_critic[1]))
        
        return q
