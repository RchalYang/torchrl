import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from distribution import TanhNormal
from networks import MLPBase

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class UniformPolicy(nn.Module):
    def __init__(self, action_shape):
        super().__init__()
        self.action_shape = action_shape

    def __call__(self,x ):
        return torch.Tensor(np.random.uniform(-1., 1., self.action_shape))

    def explore(self, x):
        return None, None, torch.Tensor(np.random.uniform(-1., 1., self.action_shape)), None

class MLPPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_shapes, **kwargs ):
        
        super().__init__()

        self.base = MLPBase( obs_shape, hidden_shapes, **kwargs )

        self.action = nn.Linear( hidden_shapes[-1], action_space )
        self.action.weight.data.uniform_(-3e-3, 3e-3)
        self.action.bias.data.uniform_(-1e-3, 1e-3)
        
    def forward(self, x):
        
        h = self.base(x)
        mean = self.action( h )
        mean = F.tanh(mean)
        return mean
    
    def eval( self, x ):
        with torch.no_grad():
            return self.forward(x)
    
    def explore( self, x ):
        return None, None, self.forward(x), None

class MLPGuassianPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_shapes, **kwargs ):
        
        super().__init__()

        self.base = MLPBase( obs_shape, hidden_shapes, **kwargs )

        self.action = nn.Linear( hidden_shapes[-1], action_space )
        self.action.weight.data.uniform_(-3e-3, 3e-3)
        self.action.bias.data.uniform_(-1e-3, 1e-3)
        
        self.last_fc_log_std = nn.Linear( hidden_shapes[-1], action_space)
        self.last_fc_log_std.weight.data.uniform_(-1e-3, 1e-3)
        self.last_fc_log_std.bias.data.uniform_(-1e-3, 1e-3)
        
    def forward(self, x):
        
        h = self.base(x)

        mean = self.action( h )
        
        log_std = self.last_fc_log_std( h )

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        
        return mean, std, log_std
    
    def eval( self, x ):
        with torch.no_grad():
            mean, std, log_std = self.forward(x)
        return torch.tanh(mean)
    
    def explore( self, x, return_log_probs = False ):
        
        mean, std, log_std = self.forward(x)

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(1, keepdim=True) 
        
        if return_log_probs:
            action, z = dis.rsample( return_pretanh_value = True )
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)
            return mean, log_std, action, log_prob, ent

        else:
            action = dis.rsample( return_pretanh_value = False )
            return mean, log_std, action, ent
    
    def get_log_probs(self, mean, std, action, pre_tanh_value = None):
        
        dis = TanhNormal (mean, std )

        log_probs = dis.log_prob( action, pre_tanh_value ).sum(1,keepdim=True)

        return log_probs 
