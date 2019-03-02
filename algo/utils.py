import torch
import numpy as np


def quantile_regression_loss( coefficient, source, target ):

    diff = target.unsqueeze(-1) - source.unsqueeze(1)
    loss = huber(diff) * ( coefficient - (diff.detach() < 0).float()).abs()
    loss = loss.mean()
    return loss

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
