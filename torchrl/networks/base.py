import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchrl.networks.init as init


class MLPBase(nn.Module):
    def __init__(
            self, input_shape, hidden_shapes,
            activation_func=nn.ReLU,
            init_func=init.basic_init,
            last_activation_func=None):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_shape: (dict): write your description
            hidden_shapes: (int): write your description
            activation_func: (todo): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            init_func: (todo): write your description
            init: (str): write your description
            basic_init: (str): write your description
            last_activation_func: (todo): write your description
        """
        super().__init__()

        self.activation_func = activation_func
        self.fcs = []
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        input_shape = np.prod(input_shape)

        self.output_shape = input_shape
        for next_shape in hidden_shapes:
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            self.fcs.append(activation_func())
            input_shape = next_shape
            self.output_shape = next_shape

        self.fcs.append(self.last_activation_func())
        self.seq_fcs = nn.Sequential(*self.fcs)

    def forward(self, x):
        """
        Evaluate x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.seq_fcs(x)


def calc_next_shape(input_shape, conv_info):
    """
    take input shape per-layer conv-info as input
    """
    out_channels, kernel_size, stride, padding = conv_info
    _, h, w = input_shape
    # for padding, dilation, kernel_size, stride in conv_info:
    h = int((h + 2*padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = int((w + 2*padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return (out_channels, h, w)


class CNNBase(nn.Module):
    def __init__(
            self, input_shape,
            hidden_shapes,
            activation_func=F.relu,
            init_func=init.basic_init,
            last_activation_func=None):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_shape: (dict): write your description
            hidden_shapes: (int): write your description
            activation_func: (todo): write your description
            F: (int): write your description
            relu: (todo): write your description
            init_func: (todo): write your description
            init: (str): write your description
            basic_init: (str): write your description
            last_activation_func: (todo): write your description
        """
        super().__init__()

        current_shape = input_shape
        in_channels = input_shape[0]
        self.activation_func = activation_func
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        self.convs = []
        self.output_shape = current_shape[0] * \
            current_shape[1] * current_shape[2]

        for conv_info in hidden_shapes:
            out_channels, kernel_size, stride, padding = conv_info
            conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride, padding)
            init_func(conv)

            self.convs.append(conv)
            in_channels = out_channels
            current_shape = calc_next_shape(current_shape, conv_info)
            self.output_shape = current_shape[0] * \
                current_shape[1] * current_shape[2]

        self.seq_convs = nn.Sequential(*self.convs)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        out = self.seq_convs(x)
        batch_size = out.size()[0]
        return out.view(batch_size, -1)
