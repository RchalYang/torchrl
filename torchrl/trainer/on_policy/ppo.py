import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
from .a2c import A2CTrainer
import torchrl.trainer.utils as atu


class PPOTrainer(A2CTrainer):
  """
  Actor Critic
  """

  def __init__(
      self,
      clip_para: float = 0.2,
      opt_epochs: int = 10,
      clipped_value_loss: bool = False,
      **kwargs
  ):
    super(PPOTrainer, self).__init__(**kwargs)
    self.clip_para = clip_para
    self.opt_epochs = opt_epochs
    self.clipped_value_loss = clipped_value_loss
    self.sample_key = ["obs", "acts", "advs", "estimate_returns", "values"]

  def pre_update_process(self) -> None:
    super().pre_update_process()
    assert self.agent.with_target_pf
    atu.copy_model_params_from_to(
        self.agent.pf, self.agent.target_pf
    )

  def update_per_epoch(self):
    # atu.update_linear_schedule(
    #     self.pf_optimizer, self.current_epoch, self.num_epochs, self.plr)
    # atu.update_linear_schedule(
    #     self.vf_optimizer, self.current_epoch, self.num_epochs, self.vlr)

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
      advs: Tensor
  ):

    out = self.agent.update(obs, actions)
    log_probs = out["log_prob"]
    ent = out["ent"]
    log_std = out["log_std"]

    with torch.no_grad():
      target_out = self.agent.update(obs, actions, use_target=True)
      target_log_probs = target_out["log_prob"]

    ratio = torch.exp(log_probs - target_log_probs.detach())
    assert ratio.shape == advs.shape, print(ratio.shape, advs.shape)
    surrogate_loss_pre_clip = ratio * advs
    surrogate_loss_clip = torch.clamp(
        ratio,
        1.0 - self.clip_para,
        1.0 + self.clip_para
    ) * advs

    policy_loss = -torch.mean(
        torch.min(surrogate_loss_clip, surrogate_loss_pre_clip)
    )
    policy_loss = policy_loss - self.entropy_coeff * ent.mean()

    self.pf_optimizer.zero_grad()
    policy_loss.backward()
    if self.grad_clip is not None:
      pf_grad_norm = torch.nn.utils.clip_grad_norm_(
          self.agent.pf.parameters(), self.grad_clip
      )
    self.pf_optimizer.step()

    info["Training/policy_loss"] = policy_loss.item()

    info["logprob/mean"] = log_probs.mean().item()
    info["logprob/std"] = log_probs.std().item()
    info["logprob/max"] = log_probs.max().item()
    info["logprob/min"] = log_probs.min().item()

    if "std" in out:
      # Log for continuous
      std = out["std"]
      info["std/mean"] = std.mean().item()
      info["std/std"] = std.std().item()
      info["std/max"] = std.max().item()
      info["std/min"] = std.min().item()

    info["ratio/max"] = ratio.max().item()
    info["ratio/min"] = ratio.min().item()
    info["grad_norm/pf"] = pf_grad_norm.item()

  def update_critic(
      self,
      info: dict,
      obs: Tensor,
      old_values: Tensor,
      est_rets: Tensor
  ):
    values = self.agent.predict_v(obs)
    assert values.shape == est_rets.shape, \
        print(values.shape, est_rets.shape)

    if self.clipped_value_loss:
      values_clipped = old_values + \
          (values - old_values).clamp(-self.clip_para, self.clip_para)
      vf_loss = (values - est_rets).pow(2)
      vf_loss_clipped = (
          values_clipped - est_rets).pow(2)
      vf_loss = 0.5 * torch.max(vf_loss,
                                vf_loss_clipped).mean()
    else:
      vf_loss = self.vf_criterion(values, est_rets)

    self.vf_optimizer.zero_grad()
    vf_loss.backward()
    if self.grad_clip is not None:
      vf_grad_norm = torch.nn.utils.clip_grad_norm_(
          self.agent.vf.parameters(), self.grad_clip
      )
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

    self.update_critic(info, obs, old_values, est_rets)
    self.update_actor(info, obs, actions, advs)

    return info
