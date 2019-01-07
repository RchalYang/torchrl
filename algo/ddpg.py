import numpy as np
import copy

import torch.optim as optim
import pytorch_util as ptu
import torch
from torch import nn as nn

from algo.rl_algo import RLAlgo
import math

class DDPG(RLAlgo):
    """
    SAC
    """

    def __init__(
            self,
            pf, qf,
            pretrain_pf,
            plr,qlr,
            optimizer_class=optim.Adam,
            
            **kwargs
    ):
        super(DDPG, self).__init__(**kwargs)
        self.pf = pf
        self.target_pf = copy.deepcopy(pf)
        self.pretrain_pf = pretrain_pf
        self.qf = qf
        self.target_qf = copy.deepcopy(qf)
        self.to(self.device)

        self.plr = plr
        self.qlr = qlr

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qlr,
        )

        self.qf_criterion = nn.MSELoss()

    def update(self, batch):
        self.training_update_num += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        rewards = torch.Tensor(rewards).to( self.device )
        terminals = torch.Tensor(terminals).to( self.device )
        obs = torch.Tensor(obs).to( self.device )
        actions = torch.Tensor(actions).to( self.device )
        next_obs = torch.Tensor(next_obs).to( self.device )

        """
        Policy Loss.
        """
        
        new_actions = self.pf(obs)
        new_q_pred = self.qf(obs, new_actions)
        
        policy_loss = -new_q_pred.mean()

        """
        QF Loss
        """
        target_actions = self.target_pf(next_obs)
        target_q_values = self.target_qf(next_obs, target_actions)

        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_pred = self.qf(obs, actions)
        qf_loss = self.qf_criterion( q_pred, q_target.detach())

        """
        Update Networks
        """

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self._update_target_networks()

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rewards.mean().item()
        info['Traning/policy_loss'] = policy_loss.item()
        info['Traning/qf_loss'] = qf_loss.item()

        info['new_actions/mean'] = new_actions.mean().item()
        info['new_actions/std'] = new_actions.std().item()
        info['new_actions/max'] = new_actions.max().item()
        info['new_actions/min'] = new_actions.min().item()

        return info

    @property
    def networks(self):
        return [
            self.pf,
            self.qf,
            self.target_pf,
            self.target_qf,
            self.pretrain_pf
        ]
    
    @property
    def target_networks(self):
        return [
            ( self.pf, self.target_pf ),
            ( self.qf, self.target_qf )
        ]
