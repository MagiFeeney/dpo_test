import gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


class sac():
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
        device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.make_env()
        model = SAC("MlpPolicy",
                    self.env,
                    verbose=1,
                    seed=self.seed,
                    device=device)
        
        model.learn(total_timesteps=self.max_steps, log_interval=self.log_interval)
        print("SAC finished one run!")
