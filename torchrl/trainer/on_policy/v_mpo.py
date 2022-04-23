import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .a2c import A2CTrainer
import torchrl.trainer.utils as atu


class VMPOTrainer(A2CTrainer):
  """
  Actor Critic
  """

  def __init__(
      self,
      opt_epochs: int = 10,
      eta_eps: float = 0.02,
      alpha_eps: float = 0.1,
      **kwargs
  ):
    super(VMPOTrainer, self).__init__(**kwargs)

    self.eta_eps = eta_eps
    self.eta = torch.Tensor([1]).to(self.device)
    self.eta.requires_grad_()

    self.alpha_eps = alpha_eps
    self.alpha = torch.Tensor([0.1]).to(self.device)
    self.alpha.requires_grad_()

    self.param_optimizer = self.optimizer_class(
        [self.eta, self.alpha],
        lr=self.plr,
        eps=1e-5,
    )

    self.opt_epochs = opt_epochs
    self.sample_key = ["obs", "acts", "advs", "estimate_returns", "values"]

  def pre_update_process(self) -> None:
    super().pre_update_process()
    assert self.agent.with_target_pf
    atu.copy_model_params_from_to(
        self.agent.pf, self.agent.target_pf
    )

  def update_per_epoch(self) -> None:
    for _ in range(self.opt_epochs):
      for batch in self.replay_buffer.one_iteration(
          self.batch_size,
          self.sample_key,
          self.shuffle,
          device=self.device
      ):
        infos = self.update(batch)
        self.logger.add_update_info(infos)

  def update_actor(
      self,
      info: dict,
      obs: Tensor,
      actions: Tensor,
      advs: Tensor,
  ) -> dict:
    _, idx = torch.sort(advs, dim=0, descending=True)
    idx = idx.reshape(-1).long()
    idx, _ = idx.chunk(2, dim=0)

    obs = obs[idx, ...]
    actions = actions[idx, ...]
    advs = advs[idx, ...]

    out = self.agent.update(obs, actions)
    log_probs = out["log_prob"]
    dis = out["dis"]

    log_probs = log_probs

    with torch.no_grad():
      target_out = self.agent.update(obs, actions, use_target=True)
      target_log_probs = target_out["log_prob"]
      target_log_probs = target_log_probs
      target_dis = target_out["dis"]

    phis = F.softmax(advs / self.eta.detach(), dim=0)

    policy_loss = -phis * log_probs
    eta_loss = self.eta * self.eta_eps + \
        self.eta * torch.log(
            torch.mean(torch.exp(advs / self.eta))
        )

    kl = torch.distributions.kl.kl_divergence(
        dis, target_dis
    ).sum(-1, keepdim=True)

    alpha_loss = self.alpha * self.alpha_eps - self.alpha * kl.detach().mean()

    policy_loss += self.alpha.detach() * kl
    policy_loss = policy_loss.mean()
    loss = policy_loss + eta_loss + alpha_loss

    self.pf_optimizer.zero_grad()
    self.param_optimizer.zero_grad()

    loss.backward()

    if self.grad_clip is not None:
      pf_grad_norm = torch.nn.utils.clip_grad_norm_(
          self.agent.pf.parameters(), self.grad_clip
      )

    self.pf_optimizer.step()
    self.param_optimizer.step()

    with torch.no_grad():
      self.eta.copy_(torch.clamp(self.eta, min=1e-8))
      self.alpha.copy_(torch.clamp(self.alpha, min=1e-8))

    info["Training/policy_loss"] = policy_loss.item()
    info["Training/alpha_loss"] = alpha_loss.item()
    info["Training/alpha"] = self.alpha.item()
    info["Training/eta"] = self.eta.item()

    info["logprob/mean"] = log_probs.mean().item()
    info["logprob/std"] = log_probs.std().item()
    info["logprob/max"] = log_probs.max().item()
    info["logprob/min"] = log_probs.min().item()

    info["KL/mean"] = kl.detach().mean().item()
    info["KL/std"] = kl.detach().std().item()
    info["KL/max"] = kl.detach().max().item()
    info["KL/min"] = kl.detach().min().item()

    info["grad_norm/pf"] = pf_grad_norm.item()

  def update_critic(
      self,
      info: dict,
      obs: Tensor,
      est_rets: Tensor
  ) -> dict:
    values = self.agent.predict_v(obs)
    assert values.shape == est_rets.shape, \
        print(values.shape, est_rets.shape)
    vf_loss = self.vf_criterion(values, est_rets)

    self.vf_optimizer.zero_grad()
    vf_loss.backward()
    if self.grad_clip is not None:
      vf_grad_norm = torch.nn.utils.clip_grad_norm_(
          self.agent.vf.parameters(), self.grad_clip)
    self.vf_optimizer.step()

    info["Training/vf_loss"] = vf_loss.item()
    info["grad_norm/vf"] = vf_grad_norm.item()

  def update(
      self,
      batch: dict
  ) -> dict:
    self.training_update_num += 1
    info = {}

    obs = batch["obs"]
    actions = batch["acts"]
    advs = batch["advs"]
    old_values = batch["values"]
    est_rets = batch["estimate_returns"]

    info["advs/mean"] = advs.mean().item()
    info["advs/std"] = advs.std().item()
    info["advs/max"] = advs.max().item()
    info["advs/min"] = advs.min().item()

    self.update_critic(info, obs, est_rets)
    self.update_actor(info, obs, actions, advs)
    return info
