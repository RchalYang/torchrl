import time
import numpy as np

import torch
from torch import nn as nn

import torchrl.trainer.utils as atu
from .dqn import DQNTrainer


class QRDQNTrainer(DQNTrainer):
    def __init__(
            self,
            quantile_num=100,
            **kwargs):
        super(QRDQNTrainer, self).__init__(**kwargs)

        self.quantile_num = quantile_num
        self.quantile_coefficient = torch.Tensor(
            (2 * np.arange(quantile_num) + 1) / (2.0 * quantile_num)
        ).view(1, -1).to(self.device)
        self.qf_criterion = atu.quantile_regression_loss

    def update(self, batch):
        self.training_update_num += 1

        obs = batch['obs']
        actions = batch['acts']
        next_obs = batch['next_obs']
        rewards = batch['rewards']
        terminals = batch['terminals']

        if not self.replay_buffer.on_gpu:
            rewards = torch.Tensor(rewards).to(self.device)
            terminals = torch.Tensor(terminals).to(self.device)
            obs = torch.Tensor(obs).to(self.device)
            actions = torch.Tensor(actions).to(self.device)
            next_obs = torch.Tensor(next_obs).to(self.device)

        batch_size = obs.shape[0]

        q_pred = self.agent.predict_q(obs)
        q_pred = q_pred.view(batch_size, -1, self.quantile_num)
        q_s_a = q_pred.gather(
            1, actions.unsqueeze(1).unsqueeze(2).repeat(
                [1, 1, self.quantile_num]).long()
        )
        q_s_a = q_s_a.squeeze(1)

        next_q_pred = self.agent.predict_q(next_obs, use_target=True)
        next_q_pred = next_q_pred.view(
            batch_size, -1, self.quantile_num
        )
        target_action = next_q_pred.detach().mean(dim=2).max(
            dim=1, keepdim=True
        )[1]

        target_q_s_a = rewards + self.discount * \
            (1 - terminals) * next_q_pred.gather(
                1, target_action.unsqueeze(2).repeat(
                    [1, 1, self.quantile_num]
                )
            ).squeeze(1)

        qf_loss = self.qf_criterion(
            self.quantile_coefficient,
            q_s_a,
            target_q_s_a.detach()
        )

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        if self.grad_clip is not None:
            qf_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.qf.parameters(), self.grad_clip
            )
        self.qf_optimizer.step()

        self._update_target_networks()

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rewards.mean().item()
        info['Training/qf_loss'] = qf_loss.item()
        info['Training/epsilon'] = self.pf.epsilon
        info['Training/q_s_a'] = q_s_a.mean().item()
        info['Training/qf_norm'] = qf_grad_norm.item()

        return info
