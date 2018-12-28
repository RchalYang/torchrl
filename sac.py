from collections import OrderedDict

import numpy as np
import copy

import torch.optim as optim
import pytorch_util as ptu
import torch
from torch import nn as nn


class SAC():
    """
    SAC
    """

    def __init__(
            self,
            pf, vf, qf,
            plr,vlr,qlr,

            target_hard_update_period=1000,
            tau=1e-2,
            
            use_soft_update=False,
            optimizer_class=optim.Adam,

            discount = 0.99,
            device = 'cpu',
            policy_std_reg_weight=1e-3,
            policy_mean_reg_weight=1e-3,
            max_grad_norm = 0.5,
            norm=True
    ):

        self.pf = pf

        self.qf = qf

        self.vf = vf
        self.target_vf = copy.deepcopy(vf).to(device)

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.plr = plr
        self.vlr = vlr
        self.qlr = qlr

        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        
        self.discount = discount

        self.device = device

        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_mean_reg_weight = policy_mean_reg_weight

        
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qlr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=self.vlr,
        )
        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )
        self.training_update_num = 0

        self.max_grad_norm = max_grad_norm
        self.norm = norm

    def update(self, batch):
        # batch = self.get_batch()
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
        Policy operations.
        """
        #print(terminals.shape)

        mean, std, new_actions, log_probs, ent = self.pf.explore(obs, return_log_probs=True )
        # log_probs = self.pf.get_log_probs( mean, std, new_actions, z )
        #print( log_probs.shape )
        #print( z.shape ) 
        q_pred = self.qf(obs, actions)
        v_pred = self.vf(obs)

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self.qf_criterion( q_pred, q_target.detach())

        """
        VF Loss
        """
        q_new_actions = self.qf(obs, new_actions)
        v_target = q_new_actions - log_probs
        vf_loss = self.vf_criterion( v_pred, v_target.detach())

        """
        Policy Loss
        """
        # log_policy_target = q_new_actions - v_pred
        # policy_loss = (
        #     log_probs * ( log_probs - log_policy_target).detach()
        # ).mean()

        policy_loss = (log_probs - q_new_actions).mean()

        std_reg_loss = self.policy_std_reg_weight * (std**2).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()

        policy_loss += std_reg_loss + mean_reg_loss
        

        """
        Update Networks
        """

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self._update_target_networks()

        info = {}
        info['policy_loss'] = policy_loss.item()
        info['vf_loss'] = vf_loss.item()
        info['qf_loss'] = qf_loss.item()
        """
        Save some statistics for eval using just one batch.
        """
        return info

    def _update_target_networks(self):
        if self.use_soft_update:
            ptu.soft_update_from_to(self.vf, self.target_vf, self.tau)
        else:
            if self.training_update_num % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.vf, self.target_vf)
                print("hard update")


    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    @property
    def networks(self):
        return [
            self.pf,
            self.qf,
            self.vf,
            self.target_vf
        ]

    # def pretrain(self):
    #     if (
    #         self.num_paths_for_normalization == 0
    #         or (self.obs_normalizer is None and self.action_normalizer is None)
    #     ):
    #         return

    #     pretrain_paths = []
    #     random_policy = RandomPolicy(self.env.action_space)
    #     while len(pretrain_paths) < self.num_paths_for_normalization:
    #         path = rollout(self.env, random_policy, self.max_path_length)
    #         pretrain_paths.append(path)
    #     ob_mean, ob_std, ac_mean, ac_std = (
    #         compute_normalization(pretrain_paths)
    #     )
    #     if self.obs_normalizer is not None:
    #         self.obs_normalizer.set_mean(ob_mean)
    #         self.obs_normalizer.set_std(ob_std)
    #         self.target_qf.obs_normalizer = self.obs_normalizer
    #         self.target_policy.obs_normalizer = self.obs_normalizer
    #     if self.action_normalizer is not None:
    #         self.action_normalizer.set_mean(ac_mean)
    #         self.action_normalizer.set_std(ac_std)
    #         self.target_qf.action_normalizer = self.action_normalizer
    #         self.target_policy.action_normalizer = self.action_normalizer


# def compute_normalization(paths):
#     obs = np.vstack([path["observations"] for path in paths])
#     ob_mean = np.mean(obs, axis=0)
#     ob_std = np.std(obs, axis=0)
#     actions = np.vstack([path["actions"] for path in paths])
#     ac_mean = np.mean(actions, axis=0)
#     ac_std = np.std(actions, axis=0)
#     return ob_mean, ob_std, ac_mean, ac_std
