import numpy as np
import copy

import torch
import torch.optim as optim
from torch import nn as nn
from torch.distributions import Normal

from .off_rl_algo import OffRLAlgo

class DDPG(OffRLAlgo):
    """
    DDPG
    """

    def __init__(
            self,
            pf, qf,
            plr,qlr,
            norm_std_explore,
            optimizer_class=optim.Adam,
            **kwargs
    ):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            pf: (todo): write your description
            qf: (int): write your description
            plr: (float): write your description
            qlr: (str): write your description
            norm_std_explore: (todo): write your description
            optimizer_class: (todo): write your description
            optim: (todo): write your description
            Adam: (todo): write your description
        """
        super(DDPG, self).__init__(**kwargs)
        self.pf = pf
        self.target_pf = copy.deepcopy(pf)
        self.qf = qf
        self.target_qf = copy.deepcopy(qf)
        self.to(self.device)

        self.plr = plr
        self.qlr = qlr

        self.norm_std_explore = norm_std_explore

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qlr,
        )

        self.qf_criterion = nn.MSELoss()

    # def get_actions(self, ob):
    #     out = self.pf.explore( torch.Tensor( ob ).to(self.device).unsqueeze(0) )
    #     action = out["action"]
    #     action = action.detach().cpu()
        
    #     action += Normal(
    #              torch.zeros( action.size()),
    #              self.norm_std_explore * torch.ones( action.size())
    #     ).sample()

    #     action = action.numpy()

    #     return action

    def update(self, batch):
        """
        Updates the update of a batch.

        Args:
            self: (todo): write your description
            batch: (todo): write your description
        """
        self.training_update_num += 1
        
        obs = batch['obs']
        actions = batch['acts']
        next_obs = batch['next_obs']
        rewards = batch['rewards']
        terminals = batch['terminals']

        rewards = torch.Tensor(rewards).to( self.device )
        terminals = torch.Tensor(terminals).to( self.device )
        obs = torch.Tensor(obs).to( self.device )
        actions = torch.Tensor(actions).to( self.device )
        next_obs = torch.Tensor(next_obs).to( self.device )

        """
        Policy Loss.
        """
        
        new_actions = self.pf(obs)
        new_q_pred = self.qf([obs, new_actions])
        
        policy_loss = -new_q_pred.mean()

        """
        QF Loss
        """
        target_actions = self.target_pf(next_obs)
        target_q_values = self.target_qf([next_obs, target_actions])

        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_pred = self.qf([obs, actions])
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
        info['Training/policy_loss'] = policy_loss.item()
        info['Training/qf_loss'] = qf_loss.item()

        info['new_actions/mean'] = new_actions.mean().item()
        info['new_actions/std'] = new_actions.std().item()
        info['new_actions/max'] = new_actions.max().item()
        info['new_actions/min'] = new_actions.min().item()

        return info

    @property
    def networks(self):
        """
        A list of the : class : network.

        Args:
            self: (todo): write your description
        """
        return [
            self.pf,
            self.qf,
            self.target_pf,
            self.target_qf
        ]
    
    @property
    def snapshot_networks(self):
        """
        Returns a list of networks

        Args:
            self: (todo): write your description
        """
        return [
            ["pf", self.pf],
            ["qf", self.qf],
        ]

    @property
    def target_networks(self):
        """
        A list of all networks of the target.

        Args:
            self: (todo): write your description
        """
        return [
            ( self.pf, self.target_pf ),
            ( self.qf, self.target_qf )
        ]
