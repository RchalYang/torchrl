import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def _uniform_init(tensor, param=3e-3):
    return tensor.data.uniform_(-param, param)

def _constant_bias_init(tensor, constant = 0.1):
    tensor.data.fill_( constant )

def layer_init(layer, weight_init = _fanin_init, bias_init = _constant_bias_init ):
    weight_init(layer.weight)
    bias_init(layer.bias)

def basic_init(layer):
    layer_init(layer, weight_init = _fanin_init, bias_init = _constant_bias_init)

def uniform_init(layer):
    layer_init(layer, weight_init = _uniform_init, bias_init = _uniform_init )

def identity(x):
    return x

def calc_next_shape(input_shape, conv_info):
	"""
	take input shape per-layer conv-info as input
	"""
    out_channels, kernel_size, stride, padding = conv_info
    h , w, c = input_shape
    # for padding, dilation, kernel_size, stride in conv_info:
    h = int((h + 2*padding[0] - ( kernel_size[0] - 1 ) - 1 ) / stride[0] + 1)
    w = int((w + 2*padding[1] - ( kernel_size[1] - 1 ) - 1 ) / stride[1] + 1)
	return (h,w, out_channels)


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
        
        self.output_shape = hidden_shapes[-1]
    
    def forward(self, x):

        out = x
        for fc in self.fcs:
            out = fc(out)
            out = self.activation_func(out)

        return out

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
class CNNBase(nn.Module):
    def __init__(self, input_shape, hidden_shapes, activation_func=F.relu, init_func = basic_init ):
        super().__init__()

        current_shape = input_shape
        in_channels = input_shape[-1]
        self.activation_func = activation_func
        self.convs = []
        for i, conv_info in enumerate( hidden_shapes ):
            out_channels, kernel_size, stride, padding = conv_info
            conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding )
            init_func(conv)
            self.convs.append(conv)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("conv{}".format(i), conv)

            in_channels = out_channels
            current_shape = calc_next_shape( current_shape, conv_info )
        
        self.output_shape = current_shape[0] * current_shape[1] * current_shape[2]
    
    def forward(self, x):

        out = x
        for conv in self.convs:
            out = conv(out)
            out = self.activation_func(out)
        
        batch_size = out.size()[0]
        return out.view(batch_size, -1)


class Net(nn.Module):
    def __init__(self, output_shape, 
            base_type, 
            append_hidden_shapes=[], append_hidden_init_func = basic_init,
            last_init_func = uniform_init,
            activation_func = F.relu,
             **kwargs ):
        
        super().__init__()

        self.base = base_type( hidden_shapes, activation_func, **kwargs )
    
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for i, next_shape in enumerate( append_hidden_shapes ):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), fc)

            append_input_shape = next_shape
    
        self.last = nn.Linear( append_hidden_shapes[-1], output_shape )     
        last_init_func( self.last )

    def forward(self, x):
        out = self.base(x)
        
        for append_fc in self.append_fcs:
            out = append_fc(out)
            out = self.activation_func(out)

        out = self.last(out)
        return out

class FlattenNet(Net):
    def forward(self, *input):
        out = torch.cat( input, dim = 1 )
        return super().forward(out)
