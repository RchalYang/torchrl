import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def constant_bias_init(tensor, constant = 0.1):
    tensor.data.fill_( constant )

def basic_init(layer, weight_init = fanin_init, bias_init = constant_bias_init ):
    fanin_init(layer.weight)
    bias_init(layer.bias)

class MLPBase(nn.Module):
    def __init__(self, input_shape, hidden_shapes, activation_func=F.relu, init_func = basic_init ):
        super().__init__()
        
        self.activation_func = activation_func
        self.fcs = []
        for i, next_shape in enumerate( hidden_shapes ):
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("fc{}".format(i), fc)

            input_shape = next_shape
    
    def forward(self, x):

        out = x
        for fc in self.fcs:
            out = fc(out)
            out = self.activation_func(out)

        return out

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shapes, **kwargs ):
        
        super().__init__()

        self.base = MLPBase( input_shape, hidden_shapes, **kwargs )

        self.last = nn.Linear( hidden_shapes[-1], 1 )     
        self.last.weight.data.uniform_(-3e-3, 3e-3)
        self.last.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        out = self.base(x)
        out = self.last(out)
        return out

class ContinuousQNet(MLP):
    def __init__(self, obs_shape, action_space, hidden_shapes, **kwargs ):
        super(ContinuousQNet, self).__init__(
            obs_shape + action_space, 1,
            hidden_shapes, **kwargs)

    def forward(self, state, action):
        x = torch.cat( [state, action], 1 )
        return super().forward(x)

class DiscreteQNet(MLP):
    def __init__(self, obs_shape, action_space, hidden_shapes, **kwargs ):
        
        super(DiscreteQNet, self).__init__(
            obs_shape + action_space, 1,
            hidden_shapes, **kwargs)

        self.base = MLPBase( obs_shape, hidden_shapes, **kwargs )

        self.q_fun = nn.Linear( hidden_shapes[-1], action_space )     
        self.q_fun.weight.data.uniform_(-3e-3, 3e-3)
        self.q_fun.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, state, action):
        out = self.base(state)
        out = self.q_fun(out)

        return out

class VNet(nn.Module):
    def __init__(self, obs_shape, hidden_shapes, **kwargs ):
        
        super().__init__()
        
        self.base = MLPBase( obs_shape, hidden_shapes, **kwargs )

        self.v_fun = nn.Linear( hidden_shapes[-1], 1 )     
        self.v_fun.weight.data.uniform_(-3e-3, 3e-3)
        self.v_fun.bias.data.uniform_(-1e-3, 1e-3)
        
    def forward(self, x):
        out = self.base(x)
        value = self.v_fun(out)
        return value




# class CNNBase(NNBase):
#     def __init__(self, num_inputs, recurrent=False, hidden_size=512):
#         super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

#         init_ = lambda m: init(m,
#             nn.init.orthogonal_,
#             lambda x: nn.init.constant_(x, 0),
#             nn.init.calculate_gain('relu'))

#         self.main = nn.Sequential(
#             init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
#             nn.ReLU(),
#             init_(nn.Conv2d(32, 64, 4, stride=2)),
#             nn.ReLU(),
#             init_(nn.Conv2d(64, 32, 3, stride=1)),
#             nn.ReLU(),
#             Flatten(),
#             init_(nn.Linear(32 * 7 * 7, hidden_size)),
#             nn.ReLU()
#         )

#         init_ = lambda m: init(m,
#             nn.init.orthogonal_,
#             lambda x: nn.init.constant_(x, 0))

#         self.critic_linear = init_(nn.Linear(hidden_size, 1))

#         self.train()

#     def forward(self, inputs, rnn_hxs, masks):
#         x = self.main(inputs / 255.0)

#         if self.is_recurrent:
#             x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

#         return self.critic_linear(x), x, rnn_hxs


