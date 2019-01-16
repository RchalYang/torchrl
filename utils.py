import argparse
import json

import torch

import algo
import policies
import networks

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

def get_agent( params):

    env = params['general_setting']['env']

    pretrain_pf = policies.UniformPolicy(env.action_space.shape[0])

    if params['agent'] == 'sac':
        pf = policies.MLPGuassianPolicy( env.observation_space.shape[0], 
            env.action_space.shape[0],
            params['net'] )
        vf = networks.VNet( env.observation_space.shape[0], params['net'] )
        qf = networks.QNet( env.observation_space.shape[0], 
            env.action_space.shape[0],
             params['net'] )
        return algo.SAC(
            pf = pf,
            vf = vf,
            qf = qf,
            pretrain_pf = pretrain_pf,
            **params['sac'],
            **params['general_setting']
        )
    
    if params['agent'] == 'twin_sac':
        pf = policies.MLPGuassianPolicy( env.observation_space.shape[0], 
            env.action_space.shape[0],
            params['net'] )
        vf = networks.VNet( env.observation_space.shape[0], params['net'] )
        qf1 = networks.QNet( env.observation_space.shape[0], 
            env.action_space.shape[0],
            params['net'] )
        qf2 = networks.QNet( env.observation_space.shape[0], 
            env.action_space.shape[0],
            params['net'] )
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
        pf = policies.MLPPolicy( env.observation_space.shape[0], 
            env.action_space.shape[0],
            params['net'],
            params['norm_std'],
            params['noise_clip'] )
        vf = networks.VNet( env.observation_space.shape[0], params['net'] )
        qf1 = networks.QNet( env.observation_space.shape[0], 
            env.action_space.shape[0],
            params['net'] )
        qf2 = networks.QNet( env.observation_space.shape[0], 
            env.action_space.shape[0],
            params['net'] )
        return algo.TD3(
            pf = pf,
            qf1 = qf1,
            qf2 = qf2,
            pretrain_pf = pretrain_pf,
            **params['td3'],
            **params['general_setting']
        )

    if params['agent'] == 'ddpg':
        pf = policies.MLPPolicy( env.observation_space.shape[0], 
            env.action_space.shape[0],
            params['net'] )
        qf = networks.QNet( env.observation_space.shape[0], 
            env.action_space.shape[0],
             params['net'] )
        return algo.DDPG(
            pf = pf,
            qf = qf,
            pretrain_pf = pretrain_pf,
            **params['ddpg'],
            **params['general_setting']
        )

    raise Exception("specified algorithm is not implemented")
