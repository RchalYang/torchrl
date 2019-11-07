import torch

import os
import time
import os.path as osp

import numpy as np

from torchrl.utils import get_args
from torchrl.utils import get_params
from get_agent import get_agent
from torchrl.env import get_env

# from torchrl.replay_buffers import SimpleReplayBuffer
from torchrl.replay_buffers import BaseReplayBuffer
from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import SAC
from torchrl.collector import BaseCollector
import gym

def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env = get_env( params['env_name'], params['env'])

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']
    replay_buffer = BaseReplayBuffer( int(buffer_param['size']) )

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger( experiment_name , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['replay_buffer'] = replay_buffer
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type']=networks.MLPBase
    # agent = get_agent( params )
    # print(env)
    # params['general_setting']['collector'] = BaseCollector(
    #     env, pf, replay_buffer
    # )

    pf = policies.GuassianContPolicy (
        input_shape = env.observation_space.shape[0], 
        output_shape = 2 * env.action_space.shape[0],
        **params['net'] )
    vf = networks.Net( 
        input_shape = env.observation_space.shape[0],
        output_shape = 1,
        **params['net'] )
    qf = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
        output_shape = 1,
        **params['net'] )
    pretrain_pf = policies.UniformPolicyContinuous(env.action_space.shape[0])
    
    params['general_setting']['collector'] = BaseCollector(
        env, pf, replay_buffer, device=device
    )
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    agent = SAC(
        pf = pf,
        vf = vf,
        qf = qf,
        pretrain_pf = pretrain_pf,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()

if __name__ == "__main__":
    experiment(args)
