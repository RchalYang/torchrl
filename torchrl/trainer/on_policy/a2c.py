import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from .on_policy_trainer import OnPolicyTrainer


class A2CTrainer(OnPolicyTrainer):
    """
    Actor Critic
    """
    def __init__(
        self,
        plr: float = 3e-4,
        vlr: float = 3e-4,
        optimizer_class: object = optim.Adam,
        entropy_coeff: float = 0.001,
        **kwargs
    ) -> None:
        super(A2CTrainer, self).__init__(**kwargs)

        self.plr = plr
        self.vlr = vlr

        self.optimizer_class = optimizer_class
        self.pf_optimizer = optimizer_class(
            self.agent.pf.parameters(),
            lr=self.plr,
            eps=1e-5,
        )

        self.vf_optimizer = optimizer_class(
            self.agent.vf.parameters(),
            lr=self.vlr,
            eps=1e-5,
        )

        self.entropy_coeff = entropy_coeff

        self.vf_criterion = nn.MSELoss()

    def update(
        self,
        batch: object
    ) -> dict:
        self.training_update_num += 1

        info = {}

        obs = batch['obs']
        acts = batch['acts']
        advs = batch['advs']
        est_rets = batch['estimate_returns']

        out = self.agent.pf.update(obs, acts)
        log_probs = out['log_prob']
        ent = out['ent']

        # Normalize the advantage
        # advs = (advs - advs.mean()) / (advs.std() + 1e-5)

        assert log_probs.shape == advs.shape, \
            "log_prob shape: {}, adv shape: {}".format(
                log_probs.shape, advs.shape)

        # Policy Update
        self.pf_optimizer.zero_grad()
        policy_loss = -log_probs * advs
        policy_loss = policy_loss.mean() - self.entropy_coeff * ent.mean()
        policy_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.agent.pf.parameters(), self.grad_clip
            )
        self.pf_optimizer.step()

        # Value Update
        self.vf_optimizer.zero_grad()
        values = self.agent.predict_v(obs)
        vf_loss = self.vf_criterion(values, est_rets)
        vf_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.agent.vf.parameters(), self.grad_clip
            )
        self.vf_optimizer.step()

        info['Training/policy_loss'] = policy_loss.item()
        info['Training/vf_loss'] = vf_loss.item()

        info['v_pred/mean'] = values.mean().item()
        info['v_pred/std'] = values.std().item()
        info['v_pred/max'] = values.max().item()
        info['v_pred/min'] = values.min().item()

        if 'std' in out:
            # Log for continuous
            std = out["std"]
            info['std/mean'] = std.mean().item()
            info['std/std'] = std.std().item()
            info['std/max'] = std.max().item()
            info['std/min'] = std.min().item()

        info['ent'] = ent.mean().item()
        info['logprob/mean'] = log_probs.mean().item()
        info['logprob/std'] = log_probs.std().item()
        info['logprob/max'] = log_probs.max().item()
        info['logprob/min'] = log_probs.min().item()

        return info

    @property
    def optimizers(self):
        return [
            ("pf_optim", self.pf_optimizer),
            ("vf_optim", self.vf_optimizer)
        ]
