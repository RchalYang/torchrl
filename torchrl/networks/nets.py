"""Network definition (Encoder is used here)"""
from typing import Callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchrl.networks import init


class ZeroNet(nn.Module):
  def forward(self, x):
    return torch.zeros(1)


class Net(nn.Module):
  """Simplest NN definition"""

  def __init__(
      self,
      output_dim: int,
      base_type: Callable,
      append_hidden_dims: list = None,
      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      lstm_layers: int = 0,
      add_ln: bool = False,
      **kwargs
  ):
    super().__init__()
    self.base = base_type(
        activation_func=activation_func,
        add_ln=add_ln,
        **kwargs
    )

    self.add_ln = add_ln
    self.activation_func = activation_func
    append_input_dim = self.base.output_dim

    self.use_lstm = lstm_layers > 0
    if self.use_lstm:
      self.lstm_layers = lstm_layers
      self.hidden_state_size = append_input_dim
      self.gru = nn.GRU(
          input_size=append_input_dim,
          hidden_size=append_input_dim,
          num_layers=lstm_layers
      )

    self.append_fcs = []
    if append_hidden_dims is None:
      append_hidden_dims = []
    for next_dim in append_hidden_dims:
      fc = nn.Linear(append_input_dim, next_dim)
      append_hidden_init_func(fc)
      self.append_fcs.append(fc)
      self.append_fcs.append(self.activation_func())
      if self.add_ln:
        self.append_fcs.append(nn.LayerNorm(next_dim))
      append_input_dim = next_dim
    self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    last = nn.Linear(append_input_dim, output_dim)
    net_last_init_func(last)
    self.last_linear = last
    # self.append_fcs.append(last)

  def forward(self, x, h=None):
    out = self.base(x)
    if self.use_lstm:
      out = out.unsqueeze(0)
      out, h = self.gru(out, h)
      out = out.squeeze(0)
    out = self.seq_append_fcs(out)
    out = self.last_linear(out)
    return out, h


class FlattenNet(Net):
  def forward(self, x, h=None):
    out = torch.cat(x, dim=-1)
    return super().forward(out, h)


class QNet(Net):
  """Q Network"""

  def forward(self, x, h=None):
    assert len(x) == 2, "Q Net only get observation and action"
    state, action = x
    x = torch.cat([state, action], dim=-1)
    return super().forward(x, h)


class BootstrappedNet(nn.Module):
  """Multi-Head bootstrapped netowrk."""

  def __init__(
      self,
      output_dim,
      base_type: nn.Module,
      num_heads: int = 10,
      append_hidden_dims: list = None,
      append_hidden_init_func=init.basic_init,
      net_last_init_func=init.uniform_init,
      activation_func=nn.ReLU,
      lstm_layers: int = 0,
      add_ln: bool = False,
      **kwargs
  ):
    super().__init__()
    self.base = base_type(
        activation_func=activation_func,
        add_ln=add_ln,
        ** kwargs)
    self.add_ln = add_ln
    self.activation_func = activation_func

    self.use_lstm = (lstm_layers > 0)
    self.bootstrapped_head_fcs = nn.ModuleList()
    if self.use_lstm:
      self.bootstrapped_head_lstms = nn.ModuleList()
    self.bootstrapped_lasts = nn.ModuleList()

    self.num_heads = num_heads
    for _ in range(num_heads):
      append_input_dim = self.base.output_dim
      append_fcs = []
      if append_hidden_dims is None:
        append_hidden_dims = []
      for next_dim in append_hidden_dims:
        fc = nn.Linear(append_input_dim, next_dim)
        append_hidden_init_func(fc)
        append_fcs.append(fc)
        append_fcs.append(self.activation_func())
        if self.add_ln:
          append_fcs.append(nn.LayerNorm(next_dim))
        # set attr for pytorch to track parameters( device )
        append_input_dim = next_dim
      seq_append_fcs = nn.Sequential(*append_fcs)
      self.bootstrapped_head_fcs.append(seq_append_fcs)

      if self.use_lstm:
        gru = nn.GRU(
            input_size=append_input_dim,
            hidden_size=append_input_dim,
            layers=lstm_layers
        )
        self.bootstrapped_head_lstms.append(gru)

      last = nn.Linear(append_input_dim, output_dim)
      net_last_init_func(last)
      self.bootstrapped_lasts.append(last)

  def forward(self, x, head_idxs, h=None):
    # Assume head_idxs with shape (batch_size, 1)
    feature = self.base(x)
    outputs = []
    hs = []
    for i in range(self.num_heads):
      output = self.bootstrapped_head_fcs[i](feature)
      if self.use_lstm:
        output, h = self.bootstrapped_head_lstms[i](output, h)
        hs.append(h.unsqueeze(-1))
      output = self.bootstrapped_lasts[i](output)
      outputs.append(output.unsqueeze(-1))
    outputs = torch.cat(outputs, dim=-1)
    outputs = torch.gather(
        outputs, head_idxs.unsqueeze(-2).repeat_interleave(
            1, outputs.shape[-2]
        ), dim=-1
    )
    if self.use_lstm:
      hs = torch.cat(hs, dim=-1)
      hs = torch.gather(
          hs, head_idxs.unsqueeze(-2).repeat_interleave(
              1, hs.shape[-2]
          ), dim=-1
      )
      return outputs, hs
    return outputs


class FlattenBootstrappedNet(BootstrappedNet):
  def forward(self, x, head_idxs, h=None):
    out = torch.cat(x, dim=-1)
    return super().forward(out, head_idxs, h=h)
