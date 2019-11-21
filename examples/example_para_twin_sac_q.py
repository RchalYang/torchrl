import sys
# import sys
sys.path.append(".") 

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
from torchrl.algo import TwinSACQ
from torchrl.collector import BaseCollector
from torchrl.collector.para import ParallelCollector
from torchrl.collector.para import AsyncParallelCollector
from torchrl.replay_buffers.shared import SharedBaseReplayBuffer
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
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

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger( experiment_name , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type']=networks.MLPBase

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    pf = policies.GuassianContPolicy (
        input_shape = env.observation_space.shape[0], 
        output_shape = 2 * env.action_space.shape[0],
        **params['net'] )
    qf1 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
        output_shape = 1,
        **params['net'] )

    qf2 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
        output_shape = 1,
        **params['net'] )

    # pretrain_pf = policies.UniformPolicyContinuous(env.action_space.shape[0])
    
    example_ob = env.reset()
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False]
    }
    replay_buffer = AsyncSharedReplayBuffer( int(buffer_param['size']),
            args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']
    print(epochs)
    params['general_setting']['collector'] = AsyncParallelCollector(
        env, pf, replay_buffer, device=device,
        worker_nums=args.worker_nums, eval_worker_nums= args.eval_worker_nums,
        train_epochs = epochs,
        eval_epochs= params['general_setting']['num_epochs']
    )

    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    agent = TwinSACQ(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()

if __name__ == "__main__":
    experiment(args)
