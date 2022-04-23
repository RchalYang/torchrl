"""Utility Module."""
from .args import get_args
from .args import get_params
from .logger import Logger
import torch
import numpy as np
import random
import time


def set_seed(env, seed, sim_device, rl_device, torch_deterministic=False):
  if seed == -1:
    seed = time.time()
  env.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if "cuda" in str(sim_device) or "cuda" in str(rl_device):
    torch.cuda.manual_seed_all(seed)
    if torch_deterministic:
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
