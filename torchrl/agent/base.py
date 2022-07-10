import torch
import torch.nn as nn
from torch import Tensor
import os.path as osp


class RLAgent(nn.Module):
  """
  Base RL Algorithm Framework
  """

  def __init__(
      self,
      agent_device="cpu"
  ):
    super().__init__()
    # device specification
    self.agent_device = agent_device

  def snapshot(self, prefix, epoch):
    agent_file_name = f"agent_{epoch}.pth"
    model_path = osp.join(prefix, agent_file_name)
    torch.save(self.state_dict(), model_path)

  def resume(self, prefix, epoch):
    self.to(self.agent_device)
    agent_file_name = f"agent_{epoch}.pth"
    agent_path = osp.join(prefix, agent_file_name)
    self.load_state_dict(
        torch.load(
            agent_path,
            map_location=self.agent_device
        )
    )

  def forward(self, _):
    pass

  def explore(
      self,
      x: Tensor,
      h: Tensor = None,
      detach: bool = True
  ) -> dict:
    pass

  def update(
      self,
      obs: Tensor,
      actions: Tensor,
      h: Tensor = None,
      use_target: bool = False
  ) -> dict:
    pass

  def eval_act(
      self,
      x: Tensor,
      h: Tensor = None,
  ) -> Tensor:
    pass

  def predict_v(
      self,
      x: Tensor,
      h: Tensor = None,
      use_target: bool = False
  ) -> Tensor:
    pass

  def predict_q(
      self,
      x: Tensor,
      h: Tensor = None,
      use_target: bool = False
  ) -> Tensor:
    pass

  @property
  def target_networks(self) -> list:
    pass
