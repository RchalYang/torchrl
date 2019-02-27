import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from distribution import TanhNormal
from networks import MLPBase
import networks

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class UniformPolicy(nn.Module):
    def __init__(self, action_shape):
        super().__init__()
        self.action_shape = action_shape

    def __call__(self,x ):
        return torch.Tensor(np.random.uniform(-1., 1., self.action_shape))

    def explore(self, x):
        {"action":self.forward(x)}
        return {"action":torch.Tensor(np.random.uniform(-1., 1., self.action_shape))}

class UniformPolicyDiscrete(nn.Module):
    def __init__(self, action_num):
        super().__init__()
        self.action_num = action_num

    def __call__(self,x ):
        return np.random.randint(self.action_num)

    def explore(self, x):
        return {"action":np.random.randint(self.action_num)}


class DetContPolicy(networks.Net):
    def forward(self, x):
        return torch.tanh(super().forward(x))    

    def eval( self, x ):
        with torch.no_grad():
            return self.forward(x).detach().cpu().numpy()
    
    def explore( self, x ):
        return {"action":self.forward(x)}

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
        return torch.tanh(mean).detach().cpu().numpy()
    
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

        dic["action"] = action
        return dic
    
    def get_log_probs(self, mean, std, action, pre_tanh_value = None):
        
        dis = TanhNormal (mean, std )

        log_probs = dis.log_prob( action, pre_tanh_value ).sum(1,keepdim=True)

        return log_probs 

class EpsilonGreedyDQNDiscretePolicy():
    """
    wrapper over QNet
    """
    def __init__(self, qf, start_epsilon, end_epsilon, decay_frames, action_shape):
        self.qf = qf
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_frames = decay_frames
        self.count = 0
        self.action_shape = action_shape
    
    def explore(self, x):
        self.count += 1
        r = np.random.rand()
        if self.count < self.decay_frames:
            self.epsilon =  self.start_epsilon - ( self.start_epsilon - self.end_epsilon ) \
                * ( self.count / self.decay_frames )
        else:
            self.epsilon = self.end_epsilon
        
        if r < self.epsilon:
            return {
                "action":np.random.randint(0, self.action_shape )
            }
    
        output = self.qf(x)
        action = output.max(dim=-1)[1].detach().item()
        return {
            "q_value": output,
            "action":action
        }
    
    def eval(self, x):
        output = self.qf(x)
        action = output.max(dim=-1)[1].detach().item()
        return action

class BootstrappedDQNDiscretePolicy():
    """
    wrapper over QNet
    """
    def __init__(self, qf, head_num, action_shape):
        self.qf = qf
        self.head_num = head_num
        self.action_shape = action_shape
        self.idx = 0

    def sample_head(self):
        self.idx = np.random.randint(self.head_num)

    def explore(self, x):
        output = self.qf( x, [ self.idx ] )
        action = output[0].max(dim=-1)[1].detach().item()
        return {
            "q_value": output[0],
            "action":action
        }
    
    def eval(self, x):
        output = self.qf( x, range(self.head_num) )
        output = torch.mean( torch.cat(output, dim=0 ), dim=0 )
        action = output.max(dim=-1)[1].detach().item()
        return action

