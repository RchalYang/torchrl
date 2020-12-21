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
            self,
            output_shape,
            base_type,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=nn.ReLU,
            add_ln=False,
            **kwargs):
        super().__init__()
        self.base = base_type(
            activation_func=activation_func,
            add_ln=add_ln,
            **kwargs)

        self.add_ln = add_ln
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for next_shape in append_hidden_shapes:
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            self.append_fcs.append(self.activation_func())
            if self.add_ln:
                self.append_fcs.append(nn.LayerNorm(next_shape))
            append_input_shape = next_shape

        last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(last)

        self.append_fcs.append(last)
        self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    def forward(self, x):
        out = self.base(x)
        out = self.seq_append_fcs(out)
        return out


class FlattenNet(Net):
    def forward(self, input):
        out = torch.cat(input, dim=-1)
        return super().forward(out)


class QNet(Net):
    def forward(self, input):
        assert len(input) == 2, "Q Net only get observation and action"
        state, action = input
        x = torch.cat([state, action], dim=-1)
        out = self.base(x)
        out = self.seq_append_fcs(out)
        return out


class BootstrappedNet(nn.Module):
    def __init__(
            self,
            output_shape,
            base_type, head_num=10,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=nn.ReLU,
            add_ln=False,
            **kwargs):
        super().__init__()
        self.base = base_type(
            activation_func=activation_func,
            add_ln=add_ln
            **kwargs)
        self.add_ln = add_ln
        self.activation_func = activation_func

        self.bootstrapped_heads = []

        append_input_shape = self.base.output_shape

        for idx in range(head_num):
            append_input_shape = self.base.output_shape
            append_fcs = []
            for next_shape in append_hidden_shapes:
                fc = nn.Linear(append_input_shape, next_shape)
                append_hidden_init_func(fc)
                append_fcs.append(fc)
                append_fcs.append(self.activation_func())
                if self.add_ln:
                    append_fcs.append(nn.LayerNorm(next_shape))
                # set attr for pytorch to track parameters( device )
                append_input_shape = next_shape

            last = nn.Linear(append_input_shape, output_shape)
            net_last_init_func(last)
            append_fcs.append(last)
            head = nn.Sequential(*append_fcs)
            self.__setattr__(
                "head{}".format(idx),
                head)
            self.bootstrapped_heads.append(head)

    def forward(self, x, head_idxs):
        output = []
        feature = self.base(x)
        for idx in head_idxs:
            output.append(self.bootstrapped_heads[idx](feature))
        return output


class FlattenBootstrappedNet(BootstrappedNet):
    def forward(self, input, head_idxs):
        out = torch.cat(input, dim=-1)
        return super().forward(out, head_idxs)
