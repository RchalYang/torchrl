import torch
import numpy as np


def quantile_regression_loss(coefficient, source, target):
    """
    Calculate loss loss.

    Args:
        coefficient: (todo): write your description
        source: (str): write your description
        target: (array): write your description
    """
    diff = target.unsqueeze(-1) - source.unsqueeze(1)
    loss = huber(diff) * (coefficient - (diff.detach() < 0).float()).abs()
    loss = loss.mean()
    return loss


def huber(x, k=1.0):
    """
    Huberberberberber )

    Args:
        x: (todo): write your description
        k: (int): write your description
    """
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def soft_update_from_to(source, target, tau):
    """
    Update the soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft soft

    Args:
        source: (todo): write your description
        target: (todo): write your description
        tau: (todo): write your description
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    """
    Copy all the model parameters from target to target.

    Args:
        source: (todo): write your description
        target: (todo): write your description
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
   