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
from env import RewardScale
from env import NormalizeObs
from env import NormalizedActions
from env import NormalizedBoxEnv
from argument import get_args
from model import MLPPolicy
from model import UniformPolicy
from model import QNet
from model import VNet

from replay_buffer import SimpleReplayBuffer
from logger import Logger

args = get_args()

def experiment(args):
    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    env = NormalizedBoxEnv(gym.make(args.env_name), reward_scale= args.reward_scale)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    pf = MLPPolicy( env.observation_space.shape[0], env.action_space.shape[0], [args.net, args.net] )
    vf = VNet( env.observation_space.shape[0], [args.net, args.net] )
    qf = QNet( env.observation_space.shape[0], env.action_space.shape[0], [args.net, args.net] )

    pretrain_policy = UniformPolicy(env.action_space.shape[0])
    
    replay_buffer = SimpleReplayBuffer( args.buffer_size, env.observation_space.shape[0], env.action_space.shape[0] )
    logger = Logger( args.id, args.env_name, args.seed )

    agent = SAC(
                pf,
                vf,
                qf,
                pretrain_pf = pretrain_policy,

                plr = args.plr,
                vlr = args.vlr,
                qlr = args.qlr,
                
                reparameterization=args.reparameterization,

                env = env,
                replay_buffer = replay_buffer,
                logger = logger,

                target_hard_update_period=args.hard_update_interval,
                tau=args.tau,
                use_soft_update=args.soft_update,
                optimizer_class=optim.Adam,

                discount = args.discount,
                device = device,

                num_epochs = args.num_epochs,
                
            )
    agent.train()

def main():

    
    
    # pretrain_step = 10000
    # pretrain_ob = []

    # env = gym.make(args.env_name)
    training_env = NormalizedBoxEnv(gym.make(args.env_name), reward_scale= args.reward_scale)
    training_env.seed(args.seed)
    
    eval_env = NormalizedBoxEnv(gym.make(args.env_name))
    eval_env.seed(args.seed)
    # env.seed(args.seed)
    torch.manual_seed(args.seed)
    # random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    # ob = env.reset()
    # for _ in range(pretrain_step):
    #     pretrain_ob.append( ob )
    #     ob, r, done, _ = env.step( env.action_space.sample() )
    #     if done:
    #         ob = env.reset()

    # ob_mean = np.mean( pretrain_ob, axis=0 )
    # ob_var = np.var( pretrain_ob, axis=0 )

    #For half Cheetah
    reward_scale = args.reward_scale
    #training_env = RewardScale( NormalizeObs( NormalizedActions( env ) , ob_mean, ob_var ) ,reward_scale = reward_scale )
    #eval_env = RewardScale( NormalizeObs( NormalizedActions( env ), ob_mean, ob_var ) ,reward_scale = 1 )
    
    # training_env = RewardScale( NormalizeObs( NormalizedActions( env ) ) ,reward_scale = reward_scale )
    # eval_env = RewardScale( NormalizeObs( NormalizedActions( env ) ) ,reward_scale = 1 )

    # training_env = RewardScale( NormalizedActions( env )  ,reward_scale = reward_scale )
    # eval_env = RewardScale( NormalizedActions( env )  ,reward_scale = 1 )

    #training_env.train()
    #eval_env.eval()

    pf = Policy( training_env.observation_space.shape[0], training_env.action_space.shape[0], [args.net, args.net] )
    vf = VNet( training_env.observation_space.shape[0], [args.net, args.net] )
    qf = QNet( training_env.observation_space.shape[0], training_env.action_space.shape[0], [args.net, args.net] )

    agent = SAC(
                pf,
                vf,
                qf,

                plr = args.plr,
                vlr = args.vlr,
                qlr = args.qlr,
                
                target_hard_update_period=args.hard_update_interval,
                tau=args.tau,
                use_soft_update=args.soft_update,
                optimizer_class=optim.Adam,

                discount = args.discount,
                device = device,
                max_grad_norm = args.max_grad_norm,
                norm = args.norm,
                reparameterization=args.reparameterization
            )
    print(args.reparameterization)


    ob = training_env.reset()

    episode_rewards = deque(maxlen=10)

    work_dir = osp.join("log", args.env_name, str(args.seed),  args.id )
    if osp.exists( work_dir ):
        shutil.rmtree(work_dir)
    writer = SummaryWriter( work_dir )

if __name__ == "__main__":
    # main()
    experiment(args)
