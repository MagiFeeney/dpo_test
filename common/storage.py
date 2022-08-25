import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.qvalues = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.log_probs = torch.zeros(num_steps + 1, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.qvalues = self.qvalues.to(device)
        self.returns = self.returns.to(device)
        self.log_probs = self.log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, actions, log_probs,
               qvalues, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs)
        self.qvalues[self.step].copy_(qvalues)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        
    def compute_returns(self,
                        gamma,
                        alpha, 
                        use_proper_time_limits=True):
        if use_proper_time_limits:

            target = gamma * (self.qvalues[1:] - alpha * self.log_probs[1:]) * self.masks[1:] \
                + self.rewards * self.bad_masks[1:] \
                + (1 - self.bad_masks[1:]) * self.qvalues[:-1]
                       
        else:
            target = gamma * (self.qvalues[1:] - alpha * self.log_probs[1:]) * self.masks[1:] \
                + self.rewards

        self.returns[:-1].copy_(target)
        
    # def compute_returns(self,
    #                     use_gae,
    #                     gamma,
    #                     alpha,
    #                     gae_lambda,
    #                     use_proper_time_limits=True):
    #     if use_proper_time_limits:
    #         if use_gae:
    #             self.qvalues[-1] = self.qvalues[-1] - alpha * self.log_probs[-1]
    #             gae = 0
    #             for step in reversed(range(self.rewards.size(0))):
    #                 delta = self.rewards[step] + gamma * self.qvalues[
    #                     step + 1] * self.masks[step +
    #                                            1] - self.qvalues[step]
    #                 gae = delta + gamma * gae_lambda * self.masks[step +
    #                                                               1] * gae
    #                 gae = gae * self.bad_masks[step + 1]
    #                 self.returns[step] = gae + self.qvalues[step]
    #         else:
    #             self.returns[-1] = self.qvalues[-1] - alpha * self.log_probs[-1]
    #             for step in reversed(range(self.rewards.size(0))):
    #                 self.returns[step] = (self.returns[step + 1] * \
    #                     gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
    #                     + (1 - self.bad_masks[step + 1]) * self.qvalues[step]
    #     else:
    #         if use_gae:
    #             self.qvalues[-1] = self.qvalues[-1] - alpha * self.log_probs[-1]
    #             gae = 0
    #             for step in reversed(range(self.rewards.size(0))):
    #                 delta = self.rewards[step] + gamma * self.qvalues[
    #                     step + 1] * self.masks[step +
    #                                            1] - self.qvalues[step]
    #                 gae = delta + gamma * gae_lambda * self.masks[step +
    #                                                               1] * gae
    #                 self.returns[step] = gae + self.qvalues[step]
    #         else:
    #             self.returns[-1] = self.qvalues[-1] - alpha * self.log_probs[-1]
    #             for step in reversed(range(self.rewards.size(0))):
    #                 self.returns[step] = self.returns[step + 1] * \
    #                     gamma * self.masks[step + 1] + self.rewards[step]

    
    def feed_forward_generator(self,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            qvalues_batch = self.qvalues[:-1].view(-1, 1)[indices]
            returns_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_log_probs_batch = self.log_probs[:-1].view(-1, 1)[indices]
            
            yield obs_batch, actions_batch, qvalues_batch, \
                returns_batch, masks_batch, old_log_probs_batch
