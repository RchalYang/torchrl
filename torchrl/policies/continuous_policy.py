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
        super().__init__()
        self.action_shape = action_shape

    def forward(self, x ):
        return torch.Tensor(np.random.uniform(-1., 1., self.action_shape))

    def explore(self, x):
        {"action":self.forward(x)}
        return {"action":torch.Tensor(np.random.uniform(-1., 1., self.action_shape))}


class DetContPolicy(networks.Net):
    def forward(self, x):
        return torch.tanh(super().forward(x))    

    def eval_act( self, x ):
        with torch.no_grad():
            return self.forward(x).squeeze(0).detach().cpu().numpy()
    
    def explore( self, x ):
        return {"action":self.forward(x).squeeze(0)}

class FixGuassianContPolicy(networks.Net):
    def __init__(self, norm_std_explore, **kwargs):
        super().__init__(**kwargs)
        self.norm_std_explore = norm_std_explore

    def forward(self, x):
        return torch.tanh(super().forward(x))    

    def eval_act( self, x ):
        with torch.no_grad():
            return self.forward(x).squeeze(0).detach().cpu().numpy()
    
    def explore( self, x ):
        action = self.forward(x).squeeze(0)
        action += Normal(
                 torch.zeros( action.size()),
                 self.norm_std_explore * torch.ones( action.size())
        ).sample().to(action.device)

        return {"action":action}

# class GuassianContPolicy(networks.Net):

#     def forward(self, x):
#         x = super().forward(x)
        
#         mean, log_std = x.chunk(2, dim=-1)

#         log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
#         std = torch.exp(log_std)
        
#         return mean, std, log_std
    
#     def eval_act( self, x ):
#         with torch.no_grad():
#             mean, std, log_std = self.forward(x)
#         return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()
    
#     def explore( self, x, return_log_probs = False, return_pre_tanh = False ):
        
#         mean, std, log_std = self.forward(x)

#         dis = TanhNormal(mean, std)

#         ent = dis.entropy().sum(1, keepdim=True) 
        
#         dic = {
#             "mean": mean,
#             "log_std": log_std,
#             "ent":ent
#         }

#         if return_log_probs:
#             action, z = dis.rsample(return_pretanh_value=True)
#             log_prob = dis.log_prob(
#                 action,
#                 pre_tanh_value=z
#             )
#             log_prob = log_prob.sum(dim=1, keepdim=True)
#             dic["pre_tanh"] = z.squeeze(0)
#             dic["log_prob"] = log_prob
#         else:
#             if return_pre_tanh:
#                 action, z = dis.rsample(return_pretanh_value=True)
#                 dic["pre_tanh"] = z.squeeze(0)
#             action = dis.rsample(return_pretanh_value=False)

#         dic["action"] = action.squeeze(0)
#         return dic

#     def update(self, obs, actions):
#         mean, std, log_std = self.forward(obs)
#         dis = TanhNormal(mean, std)

#         log_prob = dis.log_prob(actions).sum(1, keepdim=True)
#         ent = dis.entropy().sum(1, keepdim=True) 
        
#         out = {
#             "mean": mean,
#             "log_std": log_std,
#             "log_prob": log_prob,
#             "ent": ent
#         }
#         return out

class GuassianContPolicy(networks.Net):

    def forward(self, x):
        x = super().forward(x)
        
        mean, log_std = x.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        
        return mean, std, log_std
    
    def eval_act( self, x ):
        with torch.no_grad():
            mean, std, log_std = self.forward(x)
        # return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()
        return mean.squeeze(0).detach().cpu().numpy()
    
    def explore( self, x, return_log_probs = False, return_pre_tanh = False ):
        
        mean, std, log_std = self.forward(x)

        dis = Normal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True) 
        
        dic = {
            "mean": mean,
            "log_std": log_std,
            "ent":ent
        }

        if return_log_probs:
            action = dis.sample()
            log_prob = dis.log_prob(action)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            # dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            # if return_pre_tanh:
                # action, z = dis.rsample(return_pretanh_value=True)
                # dic["pre_tanh"] = z.squeeze(0)
            action = dis.sample()

        dic["action"] = action.squeeze(0)
        return dic

    def update(self, obs, actions):
        mean, std, log_std = self.forward(obs)
        dis = Normal(mean, std)

        log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
        ent = dis.entropy().sum(-1, keepdim=True)
        
        out = {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "ent": ent
        }
        return out

class GuassianContPolicyBasicBias(networks.Net):

    def __init__(self, output_shape, **kwargs):
        super().__init__(output_shape=output_shape, **kwargs)
        self.logstd = nn.Parameter(torch.zeros(output_shape))

    def forward(self, x):
        mean = super().forward(x)

        logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(logstd)
        std = std.unsqueeze(0).expand_as(mean)
        return mean, std, logstd
    
    def eval_act( self, x ):
        with torch.no_grad():
            mean, std, log_std = self.forward(x)
        # return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()
        return mean.squeeze(0).detach().cpu().numpy()
    
    def explore(self, x, return_log_probs=False, return_pre_tanh=False):

        mean, std, log_std = self.forward(x)

        dis = Normal(mean, std)
        # dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True) 
        
        dic = {
            "mean": mean,
            "log_std": log_std,
            "ent": ent
        }

        if return_log_probs:
            action = dis.sample()
            log_prob = dis.log_prob(action)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            # dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            # if return_pre_tanh:
                # action, z = dis.rsample(return_pretanh_value=True)
                # dic["pre_tanh"] = z.squeeze(0)
            action = dis.sample()

        dic["action"] = action.squeeze(0)
        return dic

        # if return_log_probs:
        #     action, z = dis.rsample(return_pretanh_value=True)
        #     log_prob = dis.log_prob(
        #         action,
        #         pre_tanh_value=z
        #     )
        #     log_prob = log_prob.sum(dim=1, keepdim=True)
        #     dic["pre_tanh"] = z.squeeze(0)
        #     dic["log_prob"] = log_prob
        # else:
        #     if return_pre_tanh:
        #         action, z = dis.rsample(return_pretanh_value=True)
        #         dic["pre_tanh"] = z.squeeze(0)
        #     action = dis.rsample(return_pretanh_value=False)

        # dic["action"] = action.squeeze(0)
        # return dic

    def update(self, obs, actions):
        mean, std, log_std = self.forward(obs)
        # dis = TanhNormal(mean, std)
        dis = Normal(mean, std)
        # dis = TanhNormal(mean, std)

        log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
        ent = dis.entropy().sum(-1, keepdim=True) 
        
        out = {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "ent": ent
        }
        return out
