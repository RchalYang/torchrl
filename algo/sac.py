import time
import numpy as np
import copy

import torch.optim as optim
import pytorch_util as ptu
import torch
from torch import nn as nn

from algo.rl_algo import RLAlgo
import math

class SAC(RLAlgo):
    """
    SAC
    """

    def __init__(
            self,
            pf, vf, qf,
            pretrain_pf,
            plr,vlr,qlr,
            optimizer_class=optim.Adam,
            
            policy_std_reg_weight=1e-3,
            policy_mean_reg_weight=1e-3,

            reparameterization = True,
            **kwargs
    ):
        super(SAC, self).__init__(**kwargs)
        self.pf = pf
        self.pretrain_pf = pretrain_pf
        self.qf = qf
        self.vf = vf
        self.target_vf = copy.deepcopy(vf)
        self.to(self.device)

        self.plr = plr
        self.vlr = vlr
        self.qlr = qlr

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

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_mean_reg_weight = policy_mean_reg_weight

        self.reparameterization = reparameterization

    def pretrain(self):
        
        self.env.reset()
        self.env.train()

        self.current_step = 0
        ob = self.env.reset()
        print(self.pretrain_frames)
        print(self.epoch_frames)
        pretrain_epochs = math.ceil( self.pretrain_frames / self.epoch_frames)

        for pretrain_epoch in range( pretrain_epochs ):
            
            start = time.time()
            for step in range( self.epoch_frames):
            
                action = self.pretrain_pf( torch.Tensor( ob ).to(self.device).unsqueeze(0) )
                action = action.detach().cpu().numpy()
                next_ob, reward, done, _ = self.env.step(action)
                self.replay_buffer.add_sample( ob, action, reward, done, next_ob )

                if step > max( self.min_pool, self.batch_size ):
                    for _ in range( self.opt_times ):
                        batch = self.replay_buffer.random_batch( self.batch_size)
                        infos = self.update( batch )
                        self.logger.add_update_info( infos )

                ob = next_ob
                self.current_step += 1
                if done or self.current_step >= self.max_episode_frames:
                    ob = self.env.reset()
                    self.current_step = 0
            
            self.eval()

            total_frames = (pretrain_epoch + 1) * self.epoch_frames
            
            infos = {}
            infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
            
            self.logger.add_epoch_info(pretrain_epoch, total_frames, time.time() - start, infos )
            # self.logger.flush()

    def update(self, batch):
        self.training_update_num += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        
        if ( (terminals==1).any() ):
            exit()
         
        rewards = torch.Tensor(rewards).to( self.device )
        terminals = torch.Tensor(terminals).to( self.device )
        obs = torch.Tensor(obs).to( self.device )
        actions = torch.Tensor(actions).to( self.device )
        next_obs = torch.Tensor(next_obs).to( self.device )

        """
        Policy operations.
        """

        mean, log_std, new_actions, log_probs, ent = self.pf.explore(obs, return_log_probs=True )
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
        if not self.reparameterization:
            log_policy_target = q_new_actions - v_pred
            policy_loss = (
                log_probs * ( log_probs - log_policy_target).detach()
            ).mean()
        else:
            policy_loss = (log_probs - q_new_actions).mean()

        std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
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

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rewards.mean().item()
        info['Traning/policy_loss'] = policy_loss.item()
        info['Traning/vf_loss'] = vf_loss.item()
        info['Traning/qf_loss'] = qf_loss.item()

        info['log_std/mean'] = log_std.mean().item()
        info['log_std/std'] = log_std.std().item()
        info['log_std/max'] = log_std.max().item()
        info['log_std/min'] = log_std.min().item()

        info['log_probs/mean'] = log_std.mean().item()
        info['log_probs/std'] = log_std.std().item()
        info['log_probs/max'] = log_std.max().item()
        info['log_probs/min'] = log_std.min().item()

        info['mean/mean'] = mean.mean().item()
        info['mean/std'] = mean.std().item()
        info['mean/max'] = mean.max().item()
        info['mean/min'] = mean.min().item()

        return info

    @property
    def networks(self):
        return [
            self.pf,
            self.qf,
            self.vf,
            self.target_vf
        ]
    
    @property
    def target_networks(self):
        return [
            ( self.vf, self.target_vf )
        ]
