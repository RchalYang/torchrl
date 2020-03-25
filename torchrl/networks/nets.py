import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchrl.networks.init as init

class ZeroNet(nn.Module):
    def forward(self, x):
        return torch.zeros(1)

class Net(nn.Module):
    def __init__(
            self, output_shape,
            base_type,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):

        super().__init__()

        self.base = base_type(activation_func=activation_func, **kwargs )
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for i, next_shape in enumerate( append_hidden_shapes ):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), fc)
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

    def forward(self, x):
        out = self.base(x)

        for append_fc in self.append_fcs:
            out = append_fc(out)
            out = self.activation_func(out)

        out = self.last(out)
        return out


class FlattenNet(Net):
    def forward(self, input):
        out = torch.cat( input, dim = 1 )
        return super().forward(out)


class BootstrappedNet(nn.Module):
    def __init__(
            self, output_shape,
            base_type, head_num=10,
            append_hidden_shapes=[], append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):

        super().__init__()

        self.base = base_type(activation_func=activation_func, **kwargs )
        self.activation_func = activation_func

        self.bootstrapped_append_fc = []
        self.bootstrapped_last = []

        append_input_shape = self.base.output_shape

        for idx in range(head_num):
            append_input_shape = self.base.output_shape
            append_fcs = []
            for i, next_shape in enumerate(append_hidden_shapes):
                fc = nn.Linear(append_input_shape, next_shape)
                append_hidden_init_func(fc)
                append_fcs.append(fc)
                # set attr for pytorch to track parameters( device )
                self.__setattr__("head_{}_append_fc{}".format(idx,i), fc)
                append_input_shape = next_shape

            last = nn.Linear(append_input_shape, output_shape)
            net_last_init_func(last)
            self.bootstrapped_last.append(last)
            self.__setattr__("head_{}_last".format(idx), last)

            self.bootstrapped_append_fc.append(append_fcs)

    def forward(self, x, head_idxs):
        output = []
        feature = self.base(x)
        for idx in head_idxs:
            out = feature
            for append_fc in self.bootstrapped_append_fc[idx]:
                out = append_fc(out)
                out = self.activation_func(out)
            out = self.bootstrapped_last[idx](out)

            output.append(out)
        return output

class FlattenBootstrappedNet(BootstrappedNet):
    def forward(self, input, head_idxs):
        out = torch.cat(input, dim=1)
        return super().forward(out, head_idxs)
