import numpy as np
import torch.nn as nn


def _fanin_init(tensor, alpha=0):
    """
    Initialize fanin tensor.

    Args:
        tensor: (todo): write your description
        alpha: (float): write your description
    """
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    # bound = 1. / np.sqrt(fan_in)
    bound = np.sqrt(1. / ((1 + alpha * alpha) * fan_in))
    return tensor.data.uniform_(-bound, bound)


def _uniform_init(tensor, param=3e-3):
    """
    Initialize a uniform parameter.

    Args:
        tensor: (todo): write your description
        param: (todo): write your description
    """
    return tensor.data.uniform_(-param, param)


def _constant_bias_init(tensor, constant=0.1):
    """
    Initialize a bias.

    Args:
        tensor: (todo): write your description
        constant: (todo): write your description
    """
    tensor.data.fill_(constant)


def layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init):
    """
    Initialize layer initialization.

    Args:
        layer: (todo): write your description
        weight_init: (todo): write your description
        _fanin_init: (int): write your description
        bias_init: (todo): write your description
        _constant_bias_init: (todo): write your description
    """
    weight_init(layer.weight)
    bias_init(layer.bias)


def basic_init(layer):
    """
    Basic init initialization.

    Args:
        layer: (todo): write your description
    """
    layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init)


def uniform_init(layer):
    """
    Initialize the weights.

    Args:
        layer: (todo): write your description
    """
    layer_init(layer, weight_init=_uniform_init, bias_init=_uniform_init)


def _orthogonal_init(tensor, gain=np.sqrt(2)):
    """
    Initialize the orthogonal tensor.

    Args:
        tensor: (todo): write your description
        gain: (array): write your description
        np: (array): write your description
        sqrt: (array): write your description
    """
    nn.init.orthogonal_(tensor, gain=gain)


def orthogonal_init(layer, scale=np.sqrt(2), constant=0):
    """
    Initialize orthogonal layer.

    Args:
        layer: (todo): write your description
        scale: (float): write your description
        np: (array): write your description
        sqrt: (array): write your description
        constant: (todo): write your description
    """
    layer_init(
        layer,
        weight_init=lambda x: _orthogonal_init(x, gain=scale),
        bias_init=lambda x: _constant_bias_init(x, 0))
