import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    parser.add_argument('--discount', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
                        
    parser.add_argument('--tau', type=float, default=0.001,
                        help='for soft update')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2',
                        help='environment to train on (default: HalfCheetah-v2)')

    parser.add_argument('--save_dir', type=str, default='./snapshots',
                        help='directory for snapshots (default: ./snapshots)')
                        
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory for tensorboard logs (default: ./log)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--soft_update', action='store_true', default=False,
                        help='soft update target net')

    parser.add_argument("--num_epochs",  type = int, default = 5000,
                        help = " num of epochs " )

    parser.add_argument("--epoch_frames", type = int, default = 1000,
                        help = " frames of an epoch " )

    parser.add_argument("--max_episode_frames", type = int, default = 999,
                        help = " max frames of an episodes " )

    parser.add_argument("--hard_update_interval", type = int, default = 1000,
                        help = " interval for hard update " )

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for update (default: 128)')

    parser.add_argument('--eval_episodes', type=int, default=1,
                        help='batch size for ppo (default: 1)')

    parser.add_argument('--plr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--vlr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--qlr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')

    parser.add_argument('--no_norm', action='store_true', default=False,
                        help='disables normalization of advantages')

    parser.add_argument("--net", help = " hidden units ", type = int, default = 300 )

    parser.add_argument("--min_pool", help = " min_pool for update ", type = int, default = 0 )
    
    parser.add_argument("--pretrain_frames", help = " pretrain frames ", type = int, default = 0 )

    parser.add_argument("--reward_scale", help = " reward scale ", type = float, default = 1.0 )

    parser.add_argument("--buffer_size", help = " replay buffer size ", type = int, default = 1000000 )

    parser.add_argument("--opt_times",   help = " opt times ", type = int, default = 1 )

    parser.add_argument("--device",      help="gpu secification", type=int, default=0 )
	#tensorboard
    parser.add_argument("--id",            help="id for tensorboard",           type=str,   default="origin" )

    parser.add_argument('--no-reparameterization', action='store_true', default=False,
                        help='no reparameterization trick')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.norm = not args.no_norm
    args.reparameterization = not args.no_reparameterization

    return args
