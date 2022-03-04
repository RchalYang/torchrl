import numpy as np
import copy
import torch
import torch.optim as optim
from torch import nn as nn
from .off_policy_trainer import OffPolicyTrainer


class DQNTrainer(OffPolicyTrainer):
    def __init__(
        self,
        qlr,
        optimizer_class=optim.Adam,
        optimizer_info={},
        **kwargs
    ):
        super(DQNTrainer, self).__init__(**kwargs)

        self.qlr = qlr
        self.qf_optimizer = optimizer_class(
            self.agent.qf.parameters(),
            lr=self.qlr,
            **optimizer_info
        )

        self.qf_criterion = nn.MSELoss()

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

        q_pred = self.agent.predict_q(obs)
        q_s_a = q_pred.gather(-1, actions.long())
        with torch.no_grad():
            next_q_pred = self.agent.predict_q(next_obs, use_target=True)

        target_q_s_a = rewards + self.discount * \
            (1 - terminals) * next_q_pred.max(-1, keepdim=True)[0]
        assert q_s_a.shape == target_q_s_a.shape
        qf_loss = self.qf_criterion(q_s_a, target_q_s_a.detach())

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        if self.grad_clip is not None:
            qf_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.qf.parameters(), self.grad_clip
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
