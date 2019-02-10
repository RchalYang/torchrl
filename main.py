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

from replay_buffer import SimpleReplayBuffer
from logger import Logger
from env import get_env

args = get_args()
params = get_params(args.config)

def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env = get_env( params['env_name'], params['env'])

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']    
    replay_buffer = SimpleReplayBuffer( int(buffer_param['size']), env.observation_space.shape, env.action_space.shape )
    
    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger( experiment_name , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['replay_buffer'] = replay_buffer
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    agent = get_agent( params )
    print(env)
    agent.train()

if __name__ == "__main__":
    # main()
    experiment(args)
