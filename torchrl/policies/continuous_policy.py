import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np

import torchrl.networks as networks
from .distribution import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class UniformPolicyContinuous(nn.Module):
    def __init__(self, action_shape):
        """
        Initialize the initial shape.

        Args:
            self: (todo): write your description
            action_shape: (todo): write your description
        """
        super().__init__()
        self.action_shape = action_shape

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return torch.Tensor(np.random.uniform(-1., 1., self.action_shape))

    def explore(self, x):
        """
        Return a random variate x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return {
            "action": torch.Tensor(np.random.uniform(
                -1., 1., self.action_shape))
        }


class DetContPolicy(networks.Net):
    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return torch.tanh(super().forward(x))

    def eval_act(self, x):
        """
        Evaluate the given function.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        with torch.no_grad():
            return self.forward(x).squeeze(0).detach().cpu().numpy()

    def explore(self, x):
        """
        Explore the forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return {
            "action": self.forward(x).squeeze(0)
        }


class FixGuassianContPolicy(networks.Net):
    def __init__(self, norm_std_explore, **kwargs):
        """
        Initialize the underlying underlying norm.

        Args:
            self: (todo): write your description
            norm_std_explore: (todo): write your description
        """
        super().__init__(**kwargs)
        self.norm_std_explore = norm_std_explore

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return torch.tanh(super().forward(x))

    def eval_act(self, x):
        """
        Evaluate the given function.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        with torch.no_grad():
            return self.forward(x).squeeze(0).detach().cpu().numpy()

    def explore(self, x):
        """
        Perform an action on x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        action = self.forward(x).squeeze(0)
        action += Normal(
            torch.zeros(action.size()),
            self.norm_std_explore * torch.ones(action.size())
        ).sample().to(action.device)

        return {
            "action": action
        }


class GuassianContPolicyBase():
    def eval_act(self, x):
        """
        Evaluate the distribution.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        with torch.no_grad():
            mean, _, _ = self.forward(x)
        if self.tanh_action:
            mean = torch.tanh(mean)
        return mean.squeeze(0).detach().cpu().numpy()

    def explore(self, x, return_log_probs=False, return_pre_tanh=False):
        """
        Explore an objective function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            return_log_probs: (bool): write your description
            return_pre_tanh: (bool): write your description
        """
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
        """
        Update actions.

        Args:
            self: (todo): write your description
            obs: (array): write your description
            actions: (todo): write your description
        """
        mean, std, log_std = self.forward(obs)

        if self.tanh_action:
            dis = TanhNormal(mean, std)
        else:
            dis = Normal(mean, std)

        log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
        ent = dis.entropy().sum(-1, keepdim=True)

        out = {
            "mean": mean,
            "log_std": log_std,
            "std": std,
            "log_prob": log_prob,
            "ent": ent
        }
        return out


class GuassianContPolicy(networks.Net, GuassianContPolicyBase):
    def __init__(self, tanh_action=False, **kwargs):
        """
        Initialize the underlying action.

        Args:
            self: (todo): write your description
            tanh_action: (todo): write your description
        """
        super().__init__(**kwargs)
        self.tanh_action = tanh_action

    def forward(self, x):
        """
        Perform of the log - likelihood.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = super().forward(x)

        mean, log_std = x.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        return mean, std, log_std


class GuassianContPolicyBasicBias(networks.Net, GuassianContPolicyBase):
    def __init__(self, output_shape, tanh_action=False, **kwargs):
        """
        Initialize the layer.

        Args:
            self: (todo): write your description
            output_shape: (str): write your description
            tanh_action: (todo): write your description
        """
        super().__init__(output_shape=output_shape, **kwargs)
        self.logstd = nn.Parameter(torch.zeros(output_shape))
        self.tanh_action = tanh_action

    def forward(self, x):
        """
        Forward computation of the model.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        mean = super().forward(x)

        logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(logstd)
        std = std.unsqueeze(0).expand_as(mean)
        return mean, std, logstd
