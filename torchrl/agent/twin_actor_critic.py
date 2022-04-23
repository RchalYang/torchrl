import copy
import torch.nn as nn
from torch import Tensor
from torchrl.agent.base import RLAgent


class TwinActorCriticAgent(RLAgent):
  """
  Actor Critic
  """

  def __init__(
      self,
      pf: nn.Module,
      qf1: nn.Module,
      qf2: nn.Module,
      with_target_pf: bool = False,
      with_target_qf: bool = False,
      **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.pf = pf
    self.qf1 = qf1
    self.qf2 = qf2

    self.with_target_pf = with_target_pf
    self.with_target_qf = with_target_qf

    if self.with_target_pf:
      self.target_pf = copy.deepcopy(self.pf)
    if self.with_target_qf:
      self.target_qf1 = copy.deepcopy(self.qf1)
      self.target_qf2 = copy.deepcopy(self.qf2)

    self.to(self.agent_device)

  def explore(
      self,
      x: Tensor,
      # return_numpy: bool = True
  ) -> dict:
    out_dict = self.pf.explore(x)
    # if return_numpy:
    #   for key in out_dict.items():
    #     out_dict[key] = out_dict[key].detach().cpu().numpy()
    return out_dict

  def update(
      self,
      obs: Tensor,
      actions: Tensor,
      use_target: bool = False
  ) -> dict:
    if use_target:
      assert self.with_target_pf
      return self.target_pf.update(obs, actions)
    return self.pf.update(obs, actions)

  def eval_act(
      self,
      x: Tensor,
      return_numpy: bool = True
  ) -> Tensor:
    action = self.pf.eval_act(x, return_numpy=return_numpy)
    return action

  def predict_q1(
      self,
      x: Tensor,
      use_target: bool = False
  ) -> Tensor:
    if use_target:
      return self.target_qf1(x)
    return self.qf1(x)

  def predict_q2(
      self,
      x: Tensor,
      use_target: bool = False
  ) -> Tensor:
    if use_target:
      return self.target_qf2(x)
    return self.qf2(x)

  @property
  def target_networks(self) -> list:
    target_list = []
    if self.with_target_pf:
      target_list.append([self.pf, self.target_pf])
    if self.with_target_qf:
      target_list.append([self.qf1, self.target_qf1])
      target_list.append([self.qf2, self.target_qf2])
    return target_list
