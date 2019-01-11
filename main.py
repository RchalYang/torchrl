import torch
import torch.optim as optim

import gym
from tensorboardX import SummaryWriter
import copy
import glob
import os
import time
import shutil
import os.path as osp
import stat
from collections import deque
import numpy as np

from algo.sac import SAC
from env import NormalizedContinuousEnv
from utils import get_args
from utils import get_params
from utils import get_agent
from policies import MLPGuassianPolicy
from policies import UniformPolicy
from networks import QNet
from networks import VNet

from replay_buffer import SimpleReplayBuffer
from logger import Logger

args = get_args()
params = get_params(args.config)

def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env_param = params['env']
    env = NormalizedContinuousEnv( gym.make(env_param['env_name']),
            reward_scale = env_param['reward_scale'],
            obs_norm = env_param['obs_norm'],
            obs_alpha= env_param['obs_alpha']
        )

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']    
    replay_buffer = SimpleReplayBuffer( int(buffer_param['size']), env.observation_space.shape[0], env.action_space.shape[0] )
    
    experiment_name = os.path.split( os.path.splitext( args.config )[0] ) if args.id is None \
        else args.id
    logger = Logger( experiment_name , env_param['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['replay_buffer'] = replay_buffer
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    agent = get_agent( params )
    agent.train()

if __name__ == "__main__":
    # main()
    experiment(args)
