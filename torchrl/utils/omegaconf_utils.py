"""Convert Omega Config to Dict or List"""

from omegaconf import DictConfig, ListConfig
from typing import Dict
from pprint import pprint


def omegadict_to_dict(d: DictConfig) -> Dict:
  """Convert omegaconf DictConfig to python Dict in recursive manner."""
  dic = {}
  for key, value in d.items():
    if isinstance(value, DictConfig):
      value = omegadict_to_dict(value)
    if isinstance(value, ListConfig):
      value = omegalist_to_list(value)
    dic[key] = value
  return dic


def omegalist_to_list(d: ListConfig) -> Dict:
  """Convert omegaconf ListConfig to python list in recursive manner."""
  l = []
  for v in d:
    if isinstance(v, DictConfig):
      v = omegadict_to_dict(v)
    elif isinstance(v, ListConfig):
      v = omegalist_to_list(v)
    l.append(v)
  return l


def print_dict(dic):
  pprint(dic)
