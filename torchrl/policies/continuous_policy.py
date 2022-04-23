"""Policies for continuous action space."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torchrl import networks
from .distribution import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class UniformPolicyContinuous(nn.Module):
  """Uniform Sampling policy."""

  def __init__(self, action_dim):
    super().__init__()
    self.continuous = True
    self.action_dim = action_dim

  def forward(self, _):
    return torch.Tensor(np.random.uniform(-1., 1., self.action_dim))

  def explore(self, _):
    return {
        "action": torch.Tensor(np.random.uniform(
            -1., 1., self.action_dim))
    }


class DetContPolicy(networks.Net):
  """Deterministic Policy."""

  def __init__(self, tanh_action=False, **kwargs):
    super().__init__(**kwargs)
    self.continuous = True
    self.tanh_action = tanh_action

  def forward(self, x):
    if self.tanh_action:
      return torch.tanh(super().forward(x))
    else:
      return super().forward(x)

  def eval_act(self, x):
    with torch.no_grad():
      act = self.forward(x).squeeze(0).detach()
    return act

  def explore(self, x):
    return {
        "action": self.forward(x).squeeze(0)
    }


class GuassianContPolicyBase():
  """Base Interface for Gaussian Policies"""

  def eval_act(self, x):
    with torch.no_grad():
      mean, _, _ = self.forward(x)
    if self.tanh_action:
      mean = torch.tanh(mean)
    return mean.squeeze(0).detach()

  def explore(self, x, return_log_probs=False, return_pre_tanh=False):
    mean, std, log_std = self.forward(x)

    if self.tanh_action:
      dis = TanhNormal(mean, std)
    else:
      dis = Normal(mean, std)

    ent = dis.entropy().sum(-1, keepdim=True)

    dic = {
        "mean": mean,
        "log_std": log_std,
        "std": std,
        "ent": ent
    }

    if return_log_probs:
      if self.tanh_action:
        action, z = dis.rsample(return_pretanh_value=True)
        log_prob = dis.log_prob(
            action,
            pre_tanh_value=z
        )
        dic["pre_tanh"] = z.squeeze(0)
      else:
        action = dis.sample()
        log_prob = dis.log_prob(action)
      log_prob = log_prob.sum(dim=-1, keepdim=True)
      dic["log_prob"] = log_prob
    else:
      if self.tanh_action:
        if return_pre_tanh:
          action, z = dis.rsample(return_pretanh_value=True)
          dic["pre_tanh"] = z.squeeze(0)
        action = dis.rsample(return_pretanh_value=False)
      else:
        action = dis.sample()

    dic["action"] = action.squeeze(0)
    return dic

  def update(self, obs, actions):
    mean, std, log_std = self.forward(obs)

    if self.tanh_action:
      dis = TanhNormal(mean, std)
    else:
      dis = Normal(mean, std)

    log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
    ent = dis.entropy().sum(-1, keepdim=True)

    out = {
        "mean": mean,
        "dis": Normal(mean, std),
        "log_std": log_std,
        "std": std,
        "log_prob": log_prob,
        "ent": ent
    }
    return out


class FixGuassianContPolicy(networks.Net, GuassianContPolicyBase):
  """Gaussian Continuous Policy with Fixed Variance."""

  def __init__(
      self,
      norm_std_explore: float = 0.1,
      tanh_action: bool = False,
      **kwargs
  ):
    super().__init__(**kwargs)
    self.continuous = True
    self.tanh_action = tanh_action
    self.norm_std_explore = norm_std_explore
    self.norm_std_log = np.log(norm_std_explore)

  def forward(self, x):
    mean = super().forward(x)
    std = torch.ones_like(mean) * self.norm_std_explore
    log_std = torch.ones_like(mean) * self.norm_std_log
    return mean, std, log_std


class GuassianContPolicy(networks.Net, GuassianContPolicyBase):
  """Gaussian Continuous Policy with Learned Variance."""

  def __init__(self, tanh_action=False, **kwargs):
    super().__init__(**kwargs)
    self.continuous = True
    self.tanh_action = tanh_action

  def forward(self, x):
    x = super().forward(x)
    mean, log_std = x.chunk(2, dim=-1)
    log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(log_std)
    return mean, std, log_std


class GuassianContPolicyBasicBias(networks.Net, GuassianContPolicyBase):
  """Gaussian Continuous Policy with Learned Variance (Single Vec)."""

  def __init__(
      self,
      action_dim: int,
      tanh_action: bool = False,
      log_init: float = 0.125,
      **kwargs
  ):
    super().__init__(output_dim=action_dim, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(action_dim) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x):
    mean = super().forward(x)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd
