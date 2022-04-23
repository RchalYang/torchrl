"""Base Encoder for Policy / Value Function / Q Function."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchrl.networks import init


class MLPBase(nn.Module):
  """MLP encoder for vector input"""

  def __init__(
      self,
      input_dim: int,
      hidden_dims: list,
      activation_func=nn.ReLU,
      init_func=init.basic_init,
      add_ln=False,
      last_activation_func=None
  ):
    super().__init__()

    self.add_ln = add_ln
    self.activation_func = activation_func
    if last_activation_func is not None:
      self.last_activation_func = last_activation_func
    else:
      self.last_activation_func = activation_func

    self.fcs = []

    self.output_dim = input_dim
    for next_dim in hidden_dims:
      fc = nn.Linear(input_dim, next_dim)
      init_func(fc)
      self.fcs.append(fc)
      self.fcs.append(activation_func())
      if self.add_ln:
        self.fcs.append(nn.LayerNorm(next_dim))
      input_dim = next_dim
      self.output_dim = next_dim

    self.fcs.pop(-1)
    if self.add_ln:
      self.fcs.pop(-1)
    self.fcs.append(self.last_activation_func())
    self.seq_fcs = nn.Sequential(*self.fcs)

  def forward(self, x):
    return self.seq_fcs(x)


def calc_next_shape(input_shape, conv_info):
  """
  take input shape per-layer conv-info as input
  """
  out_channels, kernel_size, stride, padding = conv_info
  _, h, w = input_shape
  # for padding, dilation, kernel_size, stride in conv_info:
  h = int((h + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1)
  w = int((w + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1)
  return (out_channels, h, w)


class CNNBase(nn.Module):
  """Conv encoder for image input"""

  def __init__(
      self, input_shape,
      hidden_shapes,
      activation_func=nn.ReLU,
      init_func=init.basic_init,
      add_ln=False,
      last_activation_func=None
  ):
    super().__init__()

    current_shape = input_shape
    in_channels = input_shape[0]
    self.add_ln = add_ln
    self.activation_func = activation_func
    if last_activation_func is not None:
      self.last_activation_func = last_activation_func
    else:
      self.last_activation_func = activation_func
    self.convs = []
    self.output_dim = current_shape[0] * \
        current_shape[1] * current_shape[2]

    for conv_info in hidden_shapes:
      out_channels, kernel_size, stride, padding = conv_info
      conv = nn.Conv2d(
          in_channels, out_channels,
          kernel_size, stride, padding)
      init_func(conv)

      self.convs.append(conv)
      self.convs.append(activation_func())
      in_channels = out_channels
      current_shape = calc_next_shape(current_shape, conv_info)
      if self.add_ln:
        self.convs.append(nn.LayerNorm(current_shape[1:]))

      self.output_dim = current_shape[0] * \
          current_shape[1] * current_shape[2]

    self.convs.pop(-1)
    if self.add_ln:
      self.fcs.pop(-1)
    self.convs.append(self.last_activation_func())
    self.seq_convs = nn.Sequential(*self.convs)

  def forward(self, x):
    view_shape = x.size()[:-3] + torch.Size([-1])
    x = x.view(torch.Size(
        [np.prod(x.size()[:-3])]) + x.size()[-3:])
    out = self.seq_convs(x)
    return out.view(view_shape)
