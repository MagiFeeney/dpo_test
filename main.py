import copy
import glob
import os
import time
from collections import deque
from tqdm import tqdm

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from common import utils
from common.arguments import get_args
from common.envs import make_vec_envs
from common.model import Policy
from common.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space)
    actor_critic.to(device)

    if args.algo == 'DPO':
        agent = algo.DPO(
            actor_critic,
            args.clip_param,
            args.dpo_epoch,
            args.num_mini_batch,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
       
  
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    params = actor_critic.dist.parameters()
    for j in tqdm(range(num_updates)):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.critic_optimizer, j, num_updates,
                args.lr)
            
            utils.update_linear_schedule(
                agent.actor_optimizer, j, num_updates,
                args.lr)
            
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                action, action_log_prob, entropy = actor_critic.act(rollouts.obs[step])

                q1, q2 = actor_critic.get_q_value(
                    rollouts.obs[step],
                    action)
                
            q = torch.min(q1, q2)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
                        
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, action, action_log_prob, q,
                            reward, masks, bad_masks)

        with torch.no_grad():
            next_action, next_log_prob = actor_critic.sample_action(rollouts.obs[-1])        
            next_q1, next_q2 = actor_critic.get_q_value(rollouts.obs[-1], next_action)
            
            next_q = torch.min(next_q1, next_q2)
            rollouts.qvalues[-1]   = next_q
            rollouts.log_probs[-1] = next_log_prob

            
        rollouts.compute_returns(args.gamma, args.alpha, args.use_proper_time_limits)

        critic_loss, actor_loss, dist_entropy = agent.update(rollouts)

        new_params = actor_critic.dist.parameters()
        print(params == new_params)
        params = new_params
        
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), critic_loss,
                        actor_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
