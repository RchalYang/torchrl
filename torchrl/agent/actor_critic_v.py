import copy
import torch.nn as nn
from torch import Tensor
from torchrl.agent.base import RLAgent


class ActorCriticVAgent(RLAgent):
    """
    Actor Critic
    """
    def __init__(
        self,
        pf: nn.Module,
        vf: nn.Module,
        with_target_pf: bool = False,
        with_target_vf: bool = False,
        **kwargs
    ) -> None:
        super(ActorCriticVAgent, self).__init__(**kwargs)
        self.pf = pf
        self.vf = vf

        self.with_target_pf = with_target_pf
        self.with_target_vf = with_target_vf

        if self.with_target_pf:
            self.target_pf = copy.deepcopy(self.pf)
        if self.with_target_vf:
            self.target_vf = copy.deepcopy(self.vf)

        self.to(self.agent_device)

    def explore(
        self,
        x: Tensor,
        return_numpy: bool = True
    ) -> dict:
        out_dict = self.pf.explore(x)
        if return_numpy:
            for key in out_dict.items():
                out_dict[key] = out_dict[key].detach().cpu().numpy()
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

    def eval(
        self,
        x: Tensor,
        return_numpy: bool = True
    ) -> Tensor:
        action = self.pf.eval(x, return_numpy=return_numpy)
        return action

    def predict_v(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.vf(x)

    @property
    def target_networks(self) -> list:
        target_list = []
        if self.with_target_pf:
            target_list.append([self.pf, self.target_pf])
        if self.with_target_vf:
            target_list.append([self.vf, self.target_vf])
        return target_list
