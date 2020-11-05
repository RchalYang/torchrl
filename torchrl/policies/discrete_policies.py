import torch
import torch.nn as nn
from torch.distributions import Categorical

import numpy as np
import torchrl.networks as networks

class UniformPolicyDiscrete(nn.Module):
    def __init__(self, action_num):
        """
        Initializes the initialisation.

        Args:
            self: (todo): write your description
            action_num: (int): write your description
        """
        super().__init__()
        self.action_num = action_num

    def forward(self,x ):
        """
        Forward computation on the value

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return np.random.randint(self.action_num)

    def explore(self, x):
        """
        Return a random variates.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return {"action":np.random.randint(self.action_num)}

class EpsilonGreedyDQNDiscretePolicy():
    """
    wrapper over QNet
    """
    def __init__(self, qf, start_epsilon, end_epsilon, decay_frames, action_shape):
        """
        Initialize the socket.

        Args:
            self: (todo): write your description
            qf: (int): write your description
            start_epsilon: (float): write your description
            end_epsilon: (float): write your description
            decay_frames: (todo): write your description
            action_shape: (todo): write your description
        """
        self.qf = qf
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_frames = decay_frames
        self.count = 0
        self.action_shape = action_shape
        self.epsilon = self.start_epsilon
    
    def q_to_a(self, q):
        """
        Return the q - wise quaternion.

        Args:
            self: (todo): write your description
            q: (str): write your description
        """
        return q.max(dim=-1)[1].detach().item()

    def explore(self, x):
        """
        Explore the given action.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
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
        action = self.q_to_a(output)
        return {
            "q_value": output,
            "action":action
        }
    
    def eval_act(self, x):
        """
        Evaluate the given action.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        output = self.qf(x)
        action = self.q_to_a(output)
        return action

class EpsilonGreedyQRDQNDiscretePolicy(EpsilonGreedyDQNDiscretePolicy):
    """
    wrapper over DRQNet
    """
    def __init__(self, quantile_num, **kwargs):
        """
        Initialize num quantile.

        Args:
            self: (todo): write your description
            quantile_num: (int): write your description
        """
        super(EpsilonGreedyQRDQNDiscretePolicy,self).__init__( **kwargs)
        self.quantile_num = quantile_num

    def q_to_a(self, q):
        """
        Return q q q from q q q to - space

        Args:
            self: (todo): write your description
            q: (array): write your description
        """
        q = q.view(-1, self.action_shape, self.quantile_num)
        return q.mean(dim=-1).max(dim=-1)[1].detach().item()

class BootstrappedDQNDiscretePolicy():
    """
    wrapper over Bootstrapped QNet
    """
    def __init__(self, qf, head_num, action_shape):
        """
        Initialize the state of qf state.

        Args:
            self: (todo): write your description
            qf: (int): write your description
            head_num: (int): write your description
            action_shape: (todo): write your description
        """
        self.qf = qf
        self.head_num = head_num
        self.action_shape = action_shape
        self.idx = 0

    def sample_head(self):
        """
        Return the head of the head.

        Args:
            self: (todo): write your description
        """
        self.idx = np.random.randint(self.head_num)

    def explore(self, x):
        """
        Evaluate of x of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        output = self.qf( x, [ self.idx ] )
        action = output[0].max(dim=-1)[1].detach().item()
        return {
            "q_value": output[0],
            "action":action
        }
    
    def eval_act(self, x):
        """
        Evaluate the given action.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        output = self.qf( x, range(self.head_num) )
        output = torch.mean( torch.cat(output, dim=0 ), dim=0 )
        action = output.max(dim=-1)[1].detach().item()
        return action

class CategoricalDisPolicy(networks.Net):
    """
    Discrete Policy
    """
    def __init__( self, output_shape, **kwargs):
        """
        Initialize an action.

        Args:
            self: (todo): write your description
            output_shape: (str): write your description
        """
        super( CategoricalDisPolicy, self).__init__( output_shape = output_shape, **kwargs )
        self.action_shape = output_shape

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        logits = super().forward(x)
        return torch.softmax(logits, dim=1)

    def explore(self, x, return_log_probs = False):
        """
        Evaluate the given x

        Args:
            self: (todo): write your description
            x: (todo): write your description
            return_log_probs: (bool): write your description
        """

        output = self.forward(x)
        dis = Categorical(output)
        action = dis.sample()

        out = {
            "dis": output,
            "action": action
        }

        if return_log_probs:
            out['log_prob'] = dis.log_prob(action)

        return out

    def eval_act(self, x):
        """
        Evaluate the model.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        output = self.forward(x)
        return output.max(dim=-1)[1].detach().item()

    def update(self, obs, actions):
        """
        Updates actions.

        Args:
            self: (todo): write your description
            obs: (array): write your description
            actions: (todo): write your description
        """
        output = self.forward(obs)
        dis = Categorical(output)

        log_prob = dis.log_prob(actions).unsqueeze(1)
        ent = dis.entropy()

        out = {
            "dis": output,
            "log_prob": log_prob,
            "ent": ent
        }
        return out
