"""Wrappers for Environment with Continuous Action Space."""
import gym
import torch
import numpy as np
from .base_wrapper import BaseWrapper


class NormAct(gym.ActionWrapper, BaseWrapper):
  """
  Normalized Action      => [ -1, 1 ]
  """

  def __init__(self, env):
    super().__init__(env)
    ub = np.ones(self.env.action_space.shape)
    self.action_space = gym.spaces.Box(-1 * ub, ub)
    self.lb = torch.Tensor(self.env.action_space.low, device=self.env.device)
    self.ub = torch.Tensor(self.env.action_space.high, device=self.env.device)
    self.range = 0.5 * (self.ub - self.lb)

  def action(self, action):
    scaled_action = self.lb + (action + 1.) * self.range
    return torch.clamp(scaled_action, self.lb, self.ub)
