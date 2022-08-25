import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='Distillation-Policy-Optimization')
    parser.add_argument(
        '--algo', default='DPO', help='algorithm to use: DPO')
    parser.add_argument(
        '--lr',
        type=float,
        default=7e-4,
        help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')    
    parser.add_argument(
        '--max-kl',
        type=float,
        default=0.01,
        help='max kl divergence radius (default: 0.01)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--dpo-epoch',
        type=int,
        default=10,
        help='number of dpo epochs (default: 10)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=3e-7,
        help='Temperature parameter α determines the relative importance of the entropy\
        term against the reward (default: 3e-7)')
    parser.add_argument(
        '--opt-alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer alpha (default: 0.99)')    
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
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
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')    
    parser.add_argument(
        '--automatic_entropy_tuning',
        action='store_true',
        default=False,
        help='Automaically adjust α (default: False)')
    parser.add_argument(
        '--augment-type',
        type=str,
        default=None,
        help='Augmentation type to choose: shifted | invariant | traditional')
    parser.add_argument(
        '--l2-reg',
        type=float,
        default=1e-3,
        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument(
        '--damping',
        type=float,
        default=1e-1,
        help='damping (default: 1e-1)')
    parser.add_argument(
        '--radius',
        type=int,
        default=10,
        help='smooth radius (default: 10)')
        
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['DPO']
    
    return args
