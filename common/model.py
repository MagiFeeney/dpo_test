import torch
import torch.nn as nn
import torch.nn.functional as F

from common.distributions import Bernoulli, Categorical, DiagGaussian
from common.utils import init
from common.network import MLP_Actor, MLP_Critic


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy():
    def __init__(self, obs_shape, action_space, base=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
            dist = Categorical(self.base.output_size, action_shape)
            self.base = base(obs_shape[0], action_shape, dist)
        elif action_space.__class__.__name__ == "Box":
            action_shape = action_space.shape[0]
            dist = DiagGaussian(self.base.output_size, action_shape)
            self.base = base(obs_shape[0], action_shape, dist)            
        elif action_space.__class__.__name__ == "MultiBinary":
            action_shape = action_space.shape[0]
            dist = Bernoulli(self.base.output_size, action_shape)
            self.base = base(obs_shape[0], action_shape, dist)            
        else:
            raise NotImplementedError

    def to(self, device):
        self.base.critic.to(device)
        self.base.actor.to(device)
        self.dist.to(device)
        
    def act(self, obs, deterministic=False):
        dist = self.base.actor(obs)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        log_probs = dist.log_probs(action)
        entropy = dist.entropy().mean()

        return action, log_probs, entropy

    def get_q_value(self, obs, actions):
        q = self.base.critic(obs, actions)

        return q

    def sample_action(self, obs, deterministic=False):
        dist = self.base.actor(obs)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        log_probs = dist.log_probs(action)

        return action, log_probs

    def evaluate_actions(self, obs, actions):
        dist = self.base.actor(obs)
        q = self.base.critic(obs, actions)
        
        log_probs = dist.log_probs(actions)
        entropy = dist.entropy().mean()
        
        return q, log_probs, entropy

    
class NNBase(nn.Module):
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size

class CNNBase(NNBase):
    def __init__(self, obs_shape, action_shape, hidden_size=512):
        super(CNNBase, self).__init__(hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(obs_shape, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, action_shape))

        self.train()

    def forward(self, obs, actions, rnn_hxs, masks):
        x = self.main(obs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, obs_shape, action_shape, dist, hidden_size=64):
        num_inputs = obs_shape + action_shape
        super(MLPBase, self).__init__(hidden_size)

        self.critic = MLP_Critic(obs_shape, action_shape, hidden_size)
        self.actor  = MLP_Actor(obs_shape, hidden_size, dist)
