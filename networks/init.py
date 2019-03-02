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
