import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from .OnTheFly import OnTheFly
from .retrace import get_flat_params_from

class TRPO():
    def __init__(self,
                 actor_critic,
                 device,
                 epochs=1,
                 num_mini_batch=1,
                 max_kl=None,
                 damping=None,
                 l2_reg=None):

        self.actor_critic = actor_critic

        self.epochs = epochs
        self.num_mini_batch = num_mini_batch

        self.max_kl = max_kl
        self.damping = damping
        self.l2_reg = l2_reg
        self.device = device
        
    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        action_loss_epoch = 0
        dist_entropy_epoch = 0

        with torch.no_grad():
            old_dists = self.actor_critic.get_features(rollouts.obs[:-1].view(-1, *rollouts.obs.size()[2:]), \
                                                       rollouts.recurrent_hidden_states[:-1].view(-1, rollouts.recurrent_hidden_states.size(-1)), \
                                                       rollouts.masks[:-1].view(-1, 1))

        for e in range(self.epochs):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, old_dists, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, old_dists, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        old_dist, adv_targ = sample

                controller = OnTheFly(self.actor_critic, obs_batch, recurrent_hidden_states_batch, \
                                      masks_batch, actions_batch, return_batch, old_dist, \
                                      old_action_log_probs_batch, adv_targ)
                
                value_loss, action_loss, dist_entropy = controller.step(self.max_kl, self.damping, self.l2_reg, self.device)

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                
        num_updates = self.epochs * self.num_mini_batch
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss, action_loss_epoch, dist_entropy_epoch
