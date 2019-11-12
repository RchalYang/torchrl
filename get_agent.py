"""
Legacy. Left for Reference
"""

import gym

import torch.optim as optim

import torchrl.networks as networks
import torchrl.replay_buffers as replay_buffers
import torchrl.policies as policies

from torchrl.algo import *
from torchrl.collector import BaseCollector

def get_agent( params):

    env = params['general_setting']['env']
    # params['general_setting']['collector'] = BaseCollector(
    #     env
    # ) 

    if len(env.observation_space.shape) == 3:
        params['net']['base_type']=networks.CNNBase
        if params['env']['frame_stack']:    
            buffer_param = params['replay_buffer'] 
            efficient_buffer = replay_buffers.MemoryEfficientReplayBuffer(int(buffer_param['size']))
            params['general_setting']['replay_buffer'] = efficient_buffer
    else:
        params['net']['base_type']=networks.MLPBase

    if params['agent'] == 'sac':
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

        return SAC(
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
        vf = networks.Net( 
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
        pretrain_pf = policies.UniformPolicyContinuous(env.action_space.shape[0])

        return TwinSAC(
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
        pretrain_pf = policies.UniformPolicyContinuous(env.action_space.shape[0])

        return TD3(
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
        pretrain_pf = policies.UniformPolicyContinuous(env.action_space.shape[0])
            
        return DDPG(
            pf = pf,
            qf = qf,
            pretrain_pf = pretrain_pf,
            **params['ddpg'],
            **params['general_setting']
        )
    
    if params['agent'] == 'dqn':
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
        params["general_setting"]["optimizer_class"] = optim.RMSprop
        return DQN(
            pf = pf,
            qf = qf,
            pretrain_pf = pretrain_pf,
            **params["dqn"],
            **params["general_setting"]
        )

    if params['agent'] == 'bootstrapped dqn':
        qf = networks.BootstrappedNet(
            input_shape = env.observation_space.shape,
            output_shape = env.action_space.n,
            head_num = params['bootstrapped dqn']['head_num'],
            **params['net']
        )
        pf = policies.BootstrappedDQNDiscretePolicy(
            qf = qf,
            head_num = params['bootstrapped dqn']['head_num'],
            action_shape = env.action_space.n,
            **params['policy']
        )
        pretrain_pf = policies.UniformPolicyDiscrete(
            action_num = env.action_space.n
        )
        params["general_setting"]["optimizer_class"] = optim.RMSprop
        return BootstrappedDQN (
            pf = pf,
            qf = qf,
            pretrain_pf = pretrain_pf,
            **params["bootstrapped dqn"],
            **params["general_setting"]
        )

    if params['agent'] == 'qrdqn':
        qf = networks.Net(
            input_shape = env.observation_space.shape,
            output_shape = env.action_space.n * params["qrdqn"]["quantile_num"],
            **params['net']
        )
        pf = policies.EpsilonGreedyQRDQNDiscretePolicy(
            qf = qf,
            action_shape = env.action_space.n,
            **params['policy']
        )
        pretrain_pf = policies.UniformPolicyDiscrete(
            action_num = env.action_space.n
        )
        return QRDQN (
            pf = pf,
            qf = qf,
            pretrain_pf = pretrain_pf,
            **params["qrdqn"],
            **params["general_setting"]
        )

    # On Policy Methods
    act_space = env.action_space
    params[params['agent']]['continuous'] = isinstance(act_space, gym.spaces.Box)

    buffer_param = params['replay_buffer'] 
    buffer = replay_buffers.OnPolicyReplayBuffer(int(buffer_param['size']))
    params['general_setting']['replay_buffer'] = buffer

    if params[params['agent']]['continuous']:
        pf = policies.GuassianContPolicy(
            input_shape = env.observation_space.shape,
            output_shape = 2 * env.action_space.shape[0],
            **params['net']
        )
    else:
        print(params['policy'])
        print(params['net'])
        # print(**params['policy'])
        pf = policies.CategoricalDisPolicy(
            input_shape = env.observation_space.shape,
            output_shape = env.action_space.n,
            **params['net'],
            **params['policy']
        )

    if params['agent'] == 'reinforce':
        return Reinforce(
            pf = pf,
            **params["reinforce"],
            **params["general_setting"]
        )

    # Actor-Critic Frameworks
    vf = networks.Net( 
        input_shape = env.observation_space.shape,
        output_shape = 1,
        **params['net'] )
        
    if params['agent'] == 'a2c':
        return A2C(
            pf = pf,
            vf = vf,
            **params["a2c"],
            **params["general_setting"]
        )

    if params['agent'] == 'ppo':
        return PPO(
            pf = pf,
            vf = vf,
            **params["ppo"],
            **params["general_setting"]
        )

    raise Exception("specified algorithm is not implemented")