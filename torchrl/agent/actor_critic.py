"""Actor Critic Agents."""
import copy
from torch import nn
from torch import Tensor
from torchrl.agent.base import RLAgent


class ActorCriticAgent(RLAgent):
  """
  Actor Critic without Value or Q function
  """

  def __init__(
      self,
      pf: nn.Module,
      with_target_pf: bool = False,
      **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.pf = pf
    self.pf_use_lstm = self.pf.use_lstm
    self.with_target_pf = with_target_pf
    if self.with_target_pf:
      self.target_pf = copy.deepcopy(self.pf)
    self.to(self.agent_device)

  def explore(
      self,
      x: Tensor,
      h: Tensor = None,
      detach: bool = True
  ) -> dict:
    out_dict = self.pf.explore(x)
    if detach:
      for key in out_dict.keys():
        if out_dict[key] is not None:
          out_dict[key] = out_dict[key].detach()
    return out_dict

  def update(
      self,
      obs: Tensor,
      actions: Tensor,
      h: Tensor = None,
      use_target: bool = False
  ) -> dict:
    if use_target:
      assert self.with_target_pf
      return self.target_pf.update(obs, actions, h=h)
    return self.pf.update(obs, actions, h=h)

  def eval_act(
      self,
      x: Tensor,
      h: Tensor = None,
  ) -> Tensor:
    action = self.pf.eval_act(x, h=h)
    return action

  @property
  def target_networks(self) -> list:
    target_list = []
    if self.with_target_pf:
      target_list.append([self.pf, self.target_pf])
    return target_list


class ActorCriticVAgent(ActorCriticAgent):
  """
  Actor Critic with V function
  """

  def __init__(
      self,
      vf: nn.Module,
      with_target_vf: bool = False,
      **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.vf = vf
    self.vf_use_lstm = self.vf.use_lstm
    self.with_target_vf = with_target_vf
    if self.with_target_vf:
      self.target_vf = copy.deepcopy(self.vf)
    self.to(self.agent_device)

  def predict_v(
      self,
      x: Tensor,
      h: Tensor = None,
      use_target: bool = False
  ) -> Tensor:
    if use_target:
      return self.target_vf(x, h=h)
    return self.vf(x, h=h)

  @property
  def target_networks(self) -> list:
    target_list = []
    if self.with_target_pf:
      target_list.append([self.pf, self.target_pf])
    if self.with_target_vf:
      target_list.append([self.vf, self.target_vf])
    return target_list


class ActorCriticQAgent(ActorCriticAgent):
  """
  Actor Critic with Q function
  """

  def __init__(
      self,
      qf: nn.Module,
      with_target_qf: bool = False,
      **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.qf = qf
    self.qf_use_lstm = self.qf.use_lstm
    self.with_target_qf = with_target_qf
    if self.with_target_qf:
      self.target_qf = copy.deepcopy(self.qf)
    self.to(self.agent_device)

  def predict_q(
      self,
      obs: Tensor,
      act: Tensor,
      h: Tensor = None,
      use_target: bool = False
  ) -> Tensor:
    if use_target:
      assert self.with_target_qf
      return self.target_qf([obs, act], h=h)
    return self.qf([obs, act], h=h)

  @property
  def target_networks(self) -> list:
    target_list = []
    if self.with_target_pf:
      target_list.append([self.pf, self.target_pf])
    if self.with_target_qf:
      target_list.append([self.qf, self.target_qf])
    return target_list


class TwinActorCriticQAgent(ActorCriticAgent):
  """
  Twin Actor Critic with Q function
  """

  def __init__(
      self,
      qf1: nn.Module,
      qf2: nn.Module,
      with_target_qf: bool = False,
      **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.qf1 = qf1
    self.qf2 = qf2
    self.qf_use_lstm = self.qf1.use_lstm
    self.with_target_qf = with_target_qf
    if self.with_target_qf:
      self.target_qf1 = copy.deepcopy(self.qf1)
      self.target_qf2 = copy.deepcopy(self.qf2)
    self.to(self.agent_device)

  def predict_q1(
      self,
      x: Tensor,
      h: Tensor = None,
      use_target: bool = False
  ) -> Tensor:
    if use_target:
      return self.target_qf1(x, h=h)
    return self.qf1(x, h=h)

  def predict_q2(
      self,
      x: Tensor,
      h: Tensor = None,
      use_target: bool = False
  ) -> Tensor:
    if use_target:
      return self.target_qf2(x, h=h)
    return self.qf2(x, h=h)

  @property
  def target_networks(self) -> list:
    target_list = []
    if self.with_target_pf:
      target_list.append([self.pf, self.target_pf])
    if self.with_target_qf:
      target_list.append([self.qf1, self.target_qf1])
      target_list.append([self.qf2, self.target_qf2])
    return target_list
