import copy
import torch.nn as nn
from torch import Tensor
from torchrl.agent.base import RLAgent


class QAgent(RLAgent):

  """
  Actor Critic
  """

  def __init__(
      self,
      pf: nn.Module,
      qf: nn.Module,
      with_target_qf: bool = False,
      **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.pf = pf
    self.qf = qf

    self.with_target_qf = with_target_qf

    if self.with_target_qf:
      self.target_qf = copy.deepcopy(self.qf)

    self.to(self.agent_device)

  def explore(
      self,
      x: Tensor,
      #   return_numpy: bool = True
  ) -> dict:
    out_dict = self.pf.explore(x)
    # if return_numpy:
    #   for key in out_dict.items():
    #     out_dict[key] = out_dict[key].detach().cpu().numpy()
    return out_dict

  def eval_act(
      self,
      x: Tensor,
      return_numpy: bool = True
  ) -> Tensor:
    action = self.pf.eval_act(x, return_numpy=return_numpy)
    return action

  def predict_q(
      self,
      x: Tensor,
      use_target: bool = False
  ) -> Tensor:
    if use_target:
      return self.target_qf(x)
    return self.qf(x)

  @property
  def target_networks(self) -> list:
    target_list = []
    if self.with_target_qf:
      target_list.append([self.pf, self.target_qf])
    return target_list


# class BootstrappedQAgent(QAgent):
