import numpy as np
import copy
import torch
import torch.optim as optim
from torch import nn as nn
from torch.distributions import Normal
from .off_policy_trainer import OffPolicyTrainer


class DDPGTrainer(OffPolicyTrainer):
    """
    DDPG
    """
    def __init__(
        self,
        plr: float,
        qlr: float,
        **kwargs
    ) -> None:
        super(DDPGTrainer, self).__init__(**kwargs)

        self.plr = plr
        self.qlr = qlr

        self.pf_optimizer = self.optimizer_class(
            self.agent.pf.parameters(),
            lr=self.plr,
        )

        self.qf_optimizer = self.optimizer_class(
            self.agent.qf.parameters(),
            lr=self.qlr,
        )

        self.qf_criterion = nn.MSELoss()

    def update(
        self,
        batch: dict
    ) -> dict:
        self.training_update_num += 1

        obs = batch['obs']
        actions = batch['acts']
        next_obs = batch['next_obs']
        rewards = batch['rewards']
        terminals = batch['terminals']

        """
        Policy Loss.
        """

        new_actions = self.agent.pf(obs)
        new_q_pred = self.agent.predict_q(
            obs, new_actions
        )

        policy_loss = -new_q_pred.mean()

        """
        QF Loss
        """
        target_actions = self.agent.target_pf(next_obs)
        target_q_values = self.agent.predict_q(
            next_obs, target_actions, use_target=True
        )
        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_pred = self.agent.predict_q(obs, actions)
        assert q_pred.shape == q_target.shape
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """
        Update Networks
        """

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        if self.grad_clip:
            pf_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.pf.parameters(), self.grad_clip)
        self.pf_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        if self.grad_clip:
            qf_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.qf.parameters(), self.grad_clip)
        self.qf_optimizer.step()

        self._update_target_networks()

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rewards.mean().item()
        info['Training/policy_loss'] = policy_loss.item()
        info['Training/qf_loss'] = qf_loss.item()
        if self.grad_clip is not None:
            info['Training/pf_grad_norm'] = pf_grad_norm.item()
            info['Training/qf_grad_norm'] = qf_grad_norm.item()

        info['new_actions/mean'] = new_actions.mean().item()
        info['new_actions/std'] = new_actions.std().item()
        info['new_actions/max'] = new_actions.max().item()
        info['new_actions/min'] = new_actions.min().item()

        return info

    @property
    def optimizers(self):
        return [
            ("pf_optim", self.pf_optimizer),
            ("qf_optim", self.qf_optimizer)
        ]
