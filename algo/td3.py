import gym
import torch
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor


class td3():
    def __init__(self, env_name, max_steps, seed, use_cuda, log_dir, log_interval):
        self.env_name = env_name
        self.max_steps = max_steps
        self.seed = seed
        self.use_cuda = use_cuda
        self.log_dir = log_dir
        self.log_interval = log_interval


    def make_env(self):
        env = gym.make(self.env_name)
        self.env = Monitor(env, self.log_dir)

    def run(self):

        self.make_env()
        n_actions = self.env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        device = torch.device("cuda:0" if self.use_cuda else "cpu")
        model = TD3("MlpPolicy",
                    self.env,
                    action_noise=action_noise,
                    verbose=1,
                    seed=self.seed,
                    device=device)
    
        model.learn(total_timesteps=self.max_steps, log_interval=self.log_interval)
        print("TD3 finished one run!")
