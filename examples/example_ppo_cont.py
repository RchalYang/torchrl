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
from torchrl.env import get_env

# from torchrl.replay_buffers.on_policy import SharedOnPolicyReplayBuffer
from torchrl.replay_buffers.on_policy import OnPolicyReplayBuffer
from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks

from torchrl.algo import PPO

from torchrl.collector.para import ParallelOnPlicyCollector
from torchrl.collector.on_policy import OnPlicyCollectorBase
import gym
import random
import torchrl.networks.init as init

def experiment(args):
    """
    Run experiment

    Args:
    """

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env = get_env(params['env_name'], params['env'])

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']

    experiment_name = os.path.split(os.path.splitext(args.config)[0])[-1] if args.id is None \
        else args.id
    logger = Logger(experiment_name , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env

    # replay_buffer = OnPolicyReplayBuffer(int(buffer_param['size']))

    # example_ob = env.reset()
    # example_dict = { 
    #     "obs": example_ob,
    #     "next_obs": example_ob,
    #     "acts": env.action_space.sample(),
    #     "values": [0],
    #     "rewards": [0],
    #     "terminals": [False]
    # }
    # replay_buffer = SharedOnPolicyReplayBuffer( int(buffer_param['size']),
    #         args.worker_nums
    # )
    # replay_buffer.build_by_example(example_dict)

    replay_buffer = OnPolicyReplayBuffer(int(buffer_param['size']),
        time_limit_filter=buffer_param['time_limit_filter']
    )
    params['general_setting']['replay_buffer'] = replay_buffer

    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type'] = networks.MLPBase
    params['net']['activation_func'] = torch.tanh
    pf = policies.GuassianContPolicyBasicBias(
        input_shape=env.observation_space.shape[0],
        output_shape=env.action_space.shape[0],
        init_func=lambda x: init.orthogonal_init(
            x, scale=np.sqrt(2), constant=0
        ),
        net_last_init_func=lambda x: init.orthogonal_init(
            x, scale=0.01, constant=0
        ),
        **params['net'],
        **params['policy']
    )
    vf = networks.Net(
        input_shape=env.observation_space.shape,
        output_shape=1,
        init_func=lambda x: init.orthogonal_init(
            x, scale=np.sqrt(2), constant=0
        ),
        net_last_init_func=lambda x: init.orthogonal_init(
            x, scale=1, constant=0
        ),
        **params['net']
    )
    params['general_setting']['collector'] = OnPlicyCollectorBase(
        vf, env=env, pf=pf, replay_buffer=replay_buffer, device=device,
        train_render=False, epoch_frames=params["general_setting"]["epoch_frames"]
    )

    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    agent = PPO(
            pf=pf,
            vf=vf,
            **params["ppo"],
            **params["general_setting"]
        )
    print(params["general_setting"])
    print(agent.epoch_frames)
    agent.train()

if __name__ == "__main__":
    experiment(args)
