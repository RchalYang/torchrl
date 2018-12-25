import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')

    parser.add_argument('--discount', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
                        
    parser.add_argument('--tau', type=float, default=0.001,
                        help='for soft update')

    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')

    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')

    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')

    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')

    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')

    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--soft_update', action='store_true', default=False,
                        help='soft update target net')

    # parser.add_argument("--max_time_steps",   help = " max time steps ", type = int, default = 1000000 )
    parser.add_argument("--num_epochs",   help = " num of epochs ", type = int, default = 5000 )

    parser.add_argument("--epoch_frames",   help = " frames of an epoch ", type = int, default = 1000 )

    parser.add_argument("--hard_update_interval",   help = " interval for hard update ", type = int, default = 1000 )

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

    parser.add_argument("--buffer_size", help = " replay buffer size ", type = int, default = 1000000 )

    parser.add_argument("--opt_times",   help = " opt times ", type = int, default = 1 )

    parser.add_argument("--device",      help="gpu secification", type=int, default=0 )
	#tensorboard
    parser.add_argument("--id",            help="id for tensorboard",           type=str,   default="origin" )


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.norm = not args.no_norm 

    return args
