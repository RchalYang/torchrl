import numpy as np
import copy

import torch.optim as optim
import pytorch_util as ptu
import torch
from torch import nn as nn

from algo.rl_algo import RLAlgo
import math

class TD3(RLAlgo):
    def __init__(self,
        pf, qf1, qf2,
        pretrain_pf,
        plr, qlr,
        optimizer_class = optim.Adam,

        policy_update_delay = 2,
        **kwargs
    ):
        super(TD3, self).__init__(**kwargs)

        self.pf = pf
        self.target_pf = copy.deepcopy(pf)
        self.pretrain_pf = pretrain_pf

        self.qf1 = qf1
        self.target_qf1 = copy.deepcopy(qf1)
        self.qf2 = qf2
        self.target_qf2 = copy.deepcopy(qf2)
        self.to(self.device)

        self.plr = plr
        self.qlr = qlr

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=self.qlr,
        )

        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=self.qlr,
        )

        self.qf_criterion = nn.MSELoss()

        self.policy_update_delay = policy_update_delay

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
        QF Loss
        """
        _, _, target_actions, _ = self.target_pf.explore(next_obs)
        target_q_values = torch.min( 
                self.target_qf1(next_obs, target_actions) ,
                self.target_qf2(next_obs, target_actions)
        )
                        

        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        qf1_loss = self.qf_criterion( q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion( q2_pred, q_target.detach())


        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()
        
        # Information For Logger
        info = {}

        info['Reward_Mean'] = rewards.mean().item()

        info['Traning/qf1_loss'] = qf1_loss.item()
        info['Traning/qf2_loss'] = qf2_loss.item()


        if self.training_update_num % self.policy_update_delay:
            """
            Policy Loss.
            """
            
            _, _, new_actions, _ = self.pf.explore(obs)
            new_q_pred_1 = self.qf1(obs, new_actions)
            
            policy_loss = -new_q_pred_1.mean()

            """
            Update Networks
            """

            self.pf_optimizer.zero_grad()
            policy_loss.backward()
            self.pf_optimizer.step()

            self._update_target_networks()

            info['Traning/policy_loss'] = policy_loss.item()

            info['new_actions/mean'] = new_actions.mean().item()
            info['new_actions/std'] = new_actions.std().item()
            info['new_actions/max'] = new_actions.max().item()
            info['new_actions/min'] = new_actions.min().item()

        return info

    @property
    def networks(self):
        return [
            self.pf,
            self.qf1,
            self.qf2,
            self.target_pf,
            self.target_qf1,
            self.target_qf2,
            self.pretrain_pf
        ]
    
    @property
    def target_networks(self):
        return [
            ( self.pf, self.target_pf ),
            ( self.qf1, self.target_qf1 ),
            ( self.qf2, self.target_qf2 ),
        ]