import os
import algo
import argparse
import torch
from common import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser("SAC and TD3")
    parser.add_argument(
        '--algo',
        default='sac',
        help='algorithm to use: sac | td3 ')    
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=1e6,
        help='number of environment steps to train (default: 1e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=None,
        help='Interval to log (default: None)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    assert args.algo in ['sac', 'td3']

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    if args.algo == 'sac':
        agent = algo.sac(args.env_name,
                         args.num_env_steps,
                         args.seed,
                         args.cuda,
                         args.log_dir,
                         args.log_interval)
    elif args.algo == 'td3':
        agent = algo.td3(args.env_name,
                         args.num_env_steps,
                         args.seed,
                         args.cuda,
                         args.log_dir,
                         args.log_interval)


    agent.run()
        
