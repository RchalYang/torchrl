"""Base Policy Class"""
from torch import nn


class BasePolicy(nn.Module):
  def __init__(
      self,
      action_dim: int
  ):
    super().__init__()
    self.action_dim = action_dim

  def forward(self, x):
    pass

  def explore(self, x):
    pass

  def eval(self, x):
    pass
