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
import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import TRPO
from torchrl.collector.para import ParallelOnPlicyCollector
from torchrl.collector.on_policy import OnPlicyCollectorBase
import gym
import random
import torchrl.networks.init as init
import torch.nn as nn

args = get_args()
params = get_params(args.config)

def experiment(args):
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn')

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env = get_env(params['env_name'], params['env'])

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    buffer_param = params['replay_buffer']

    experiment_name = os.path.split(os.path.splitext(args.config)[0])[-1] if args.id is None \
        else args.id
    logger = Logger(experiment_name , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env

    replay_buffer = OnPolicyReplayBuffer(
        int(buffer_param['size']),
        time_limit_filter=buffer_param['time_limit_filter'])

    params['general_setting']['replay_buffer'] = replay_buffer

    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type'] = networks.MLPBase
    params['net']['activation_func'] = nn.Tanh
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
        train_render=False, **params["collector"]
    )

    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    agent = TRPO(
            pf=pf,
            vf=vf,
            **params["trpo"],
            **params["general_setting"]
        )
    print(params["general_setting"])
    print(agent.epoch_frames)
    agent.train()


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('experiment(args)')
    experiment(args)
    # cProfile.run('experiment(args)', 'result')
    # # cProfile.run("foo()", "result")

    # import pstats
    # p = pstats.Stats("result")
    # #这一行的效果和直接运行cProfile.run("foo()")的显示效果是一样的
    # p.strip_dirs().sort_stats(-1).print_stats()
    # #strip_dirs():从所有模块名中去掉无关的路径信息
    # #sort_stats():把打印信息按照标准的module/name/line字符串进行排序
    # #print_stats():打印出所有分析信息
 
    # #按照函数名排序 
    # # p.strip_dirs().sort_stats("name").print_stats()
 
    # #按照在一个函数中累积的运行时间进行排序
    # #print_stats(3):只打印前3行函数的信息,参数还可为小数,表示前百分之几的函数信息
    # p.strip_dirs().sort_stats("cumulative").print_stats(0.4)
 
    # #还有一种用法
    # # p.sort_stats('time', 'cum').print_stats(.5, 'foo')
    # #先按time排序,再按cumulative时间排序,然后打倒出前50%中含有函数信息

    # #如果想知道有哪些函数调用了bar,可使用
    # # p.print_callers(0.5, "bar")

    # #同理,查看foo()函数中调用了哪些函数
    # # p.print_callees("foo")