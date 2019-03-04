import torch
import torch.nn as nn

import numpy as np
from .distribution import TanhNormal
import networks

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

    def eval( self, x ):
        with torch.no_grad():
            return self.forward(x).squeeze(0).detach().cpu().numpy()
    
    def explore( self, x ):
        return {"action":self.forward(x).squeeze(0)}

class GuassianContPolicy(networks.Net):

    def forward(self, x):
        x = super().forward(x)
        
        mean, log_std = x.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        
        return mean, std, log_std
    
    def eval( self, x ):
        with torch.no_grad():
            mean, std, log_std = self.forward(x)
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()
    
    def explore( self, x, return_log_probs = False ):
        
        mean, std, log_std = self.forward(x)

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(1, keepdim=True) 
        
        dic = {
            "mean": mean,
            "log_std": log_std,
            "ent":ent
        }

        if return_log_probs:
            action, z = dis.rsample( return_pretanh_value = True )
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)
            dic["log_prob"] = log_prob
        else:
            action = dis.rsample( return_pretanh_value = False )

        dic["action"] = action.squeeze(0)
        return dic
    
    # def get_log_probs(self, mean, std, action, pre_tanh_value = None):
        
    #     dis = TanhNormal (mean, std )

    #     log_probs = dis.log_prob( action, pre_tanh_value ).sum(1,keepdim=True)

    #     return log_probs 


    def update(self, obs, actions):
        mean, std, log_std = self.forward(obs)
        dis = TanhNormal(mean, std)

        log_prob = dis.log_prob(actions).sum(1, keepdim=True)
        ent = dis.entropy().sum(1, keepdim=True) 
        
        out = {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "ent": ent
        }
        return out
