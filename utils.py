import argparse
import json
import gym

import torch

import algo
import policies
import networks
import env

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument("--config", type=str,   default=None,
                        help="config file", )

    parser.add_argument('--save_dir', type=str, default='./snapshots',
                        help='directory for snapshots (default: ./snapshots)')
                        
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory for tensorboard logs (default: ./log)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--device", type=int, default=0,
                        help="gpu secification", )

	# tensorboard
    parser.add_argument("--id", type=str,   default=None,
                        help="id for tensorboard", )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def get_params(file_name):
    with open(file_name) as f:
        params = json.load(f)
    return params


def wrap_deepmind(env, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    # if episode_life:
    #     env = EpisodicLifeEnv(env)
    # if 'FIRE' in env.unwrapped.get_action_meanings():
    #     env = FireResetEnv(env)
    env = env.WarpFrame(env)
    if scale:
        env = env.ScaledFloatFrame(env)
    # if clip_rewards:
    #     env = ClipRewardEnv(env)
    if frame_stack:
        env = env.FrameStack(env, 4)
    return env

def get_env( env_id, env_param ):

    env = gym.make(env_id)
    ob_space = env.observation_space
    if len(ob_space.shape) == 3:
        return wrap_deepmind(env, **env_param)
    else:
        return env.NormalizedContinuousEnv(env, **env_param)

def get_agent( params):

    env = params['general_setting']['env']


    base_type = networks.MLPBase

    params['net']['base_type']=base_type

    if params['agent'] == 'sac':
        pf = policies.GuassianContPolicy (
            input_shape = env.observation_space.shape[0], 
            output_shape = 2 * env.action_space.shape[0],
            **params['net'] )
        vf = networks.FlattenNet( 
            input_shape = env.observation_space.shape[0],
            output_shape = 1,
            **params['net'] )
        qf = networks.FlattenNet( 
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            output_shape = 1,
            **params['net'] )
        pretrain_pf = policies.UniformPolicy(env.action_space.shape[0])

        return algo.SAC(
            pf = pf,
            vf = vf,
            qf = qf,
            pretrain_pf = pretrain_pf,
            **params['sac'],
            **params['general_setting']
        )
    
    if params['agent'] == 'twin_sac':
        pf = policies.GuassianContPolicy (
            input_shape = env.observation_space.shape[0], 
            output_shape = 2 * env.action_space.shape[0],
            **params['net'] )
        vf = networks.FlattenNet( 
            input_shape = env.observation_space.shape[0],
            output_shape = 1,
            **params['net'] )
        qf1 = networks.FlattenNet( 
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            output_shape = 1,
            **params['net'] )
        qf2 = networks.FlattenNet( 
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            output_shape = 1,
            **params['net'] )
        pretrain_pf = policies.UniformPolicy(env.action_space.shape[0])

        return algo.TwinSAC(
            pf = pf,
            vf = vf,
            qf1 = qf1,
            qf2 = qf2,
            pretrain_pf = pretrain_pf,
            **params['twin_sac'],
            **params['general_setting']
        )

    if params['agent'] == 'td3':
        pf = policies.DetContPolicy (
            input_shape = env.observation_space.shape[0], 
            output_shape = env.action_space.shape[0],
            **params['net'] )
        qf1 = networks.FlattenNet( 
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            output_shape = 1,
            **params['net'] )
        qf2 = networks.FlattenNet( 
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            output_shape = 1,
            **params['net'] )
        pretrain_pf = policies.UniformPolicy(env.action_space.shape[0])

        return algo.TD3(
            pf = pf,
            qf1 = qf1,
            qf2 = qf2,
            pretrain_pf = pretrain_pf,
            **params['td3'],
            **params['general_setting']
        )

    if params['agent'] == 'ddpg':
        pf = policies.DetContPolicy (
            input_shape = env.observation_space.shape[0], 
            output_shape = env.action_space.shape[0],
            **params['net'] )
        qf = networks.FlattenNet( 
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            output_shape = 1,
            **params['net'] )
        pretrain_pf = policies.UniformPolicy(env.action_space.shape[0])
            
        return algo.DDPG(
            pf = pf,
            qf = qf,
            pretrain_pf = pretrain_pf,
            **params['ddpg'],
            **params['general_setting']
        )
    
    if params['agent'] == 'dqn':
        
        params['net']['base_type']=networks.CNNBase
        qf = networks.Net(
            input_shape = env.observation_space.shape,
            output_shape = env.action_space.n,
            **params['net']
        )
        pf = policies.EpsilonGreedyDQNDiscretePolicy(
            qf = qf,
            action_shape = env.action_space.n,
            **params['policy']
        )
        pretrain_pf = policies.UniformPolicyDiscrete(
            action_num = env.action_space.n
        )
        return algo.DQN(
            pf = pf,
            qf = qf,
            pretrain_pf = pretrain_pf,
            **params["dqn"],
            **params["general_setting"]
        )

    raise Exception("specified algorithm is not implemented")
