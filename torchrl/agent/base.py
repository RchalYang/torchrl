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
        agent_device='cpu'
    ):
        # device specification
        self.agent_device = agent_device

    def snapshot(self, prefix, epoch):
        agent_file_name = "agent_{}.pth".format(epoch)
        model_path = osp.join(prefix, agent_file_name)
        torch.save(self.state_dict(), model_path)
        # for name, network in self.snapshot_networks:
        #     model_file_name = "model_{}_{}.pth".format(name, epoch)
        #     model_path = osp.join(prefix, model_file_name)
        #     torch.save(network.state_dict(), model_path)

    def resume(self, prefix, epoch):
        self.to(self.agent_device)
        agent_file_name = "agent_{}.pth".format(epoch)
        agent_path = osp.join(prefix, agent_file_name)
        self.load_state_dict(
            torch.load(
                agent_path,
                map_location=self.agent_device
            )
        )

    def explore(
        self,
        x: Tensor,
        return_numpy: bool = True
    ) -> dict:
        pass

    def update(
        self,
        obs: Tensor,
        actions: Tensor,
        use_target: bool = False
    ) -> dict:
        pass

    def eval(
        self,
        x: Tensor,
        return_numpy: bool = True
    ) -> Tensor:
        pass

    def predict_v(
        self,
        x: Tensor,
    ) -> Tensor:
        pass

    def predict_q(
        self,
        x: Tensor,
        use_target: bool = False
    ) -> Tensor:
        pass

    @property
    def target_networks(self) -> list:
        pass
