"""On Policy Replay buffer."""
import torch
import numpy as np
from .base import BaseReplayBuffer


class OnPolicyReplayBufferBase:
  """Replay Buffer for On Policy algorithms."""

  def last_sample(self, sample_key, device=None):
    if device is None:
      device = self.device
    return_dict = {}
    for key in sample_key:
      return_dict[key] = self.__getattribute__("_" + key)[
          self._max_replay_buffer_size - self.num_envs:
      ].to(device)
    return return_dict

  def generalized_advantage_estimation(self, last_value, gamma, tau):
    """
    use GAE to process rewards
    """
    a = 0
    advs = torch.zeros_like(self._rewards).to(self.device)
    estimate_returns = torch.zeros_like(self._rewards).to(self.device)

    values = torch.cat([self._values, last_value], 0)
    # self.time_limit_filter: handled in sampling
    for t in reversed(range(len(self._rewards) // self.num_envs)):

      c_start, c_end = t * self.num_envs, (t + 1) * self.num_envs
      n_start, n_end = (t + 1) * self.num_envs, (t + 2) * self.num_envs

      delta = self._rewards[c_start: c_end] + \
          (1 - self._terminals[c_start: c_end]) * gamma * \
          values[n_start: n_end] - values[c_start: c_end]
      a = delta + (1 - self._terminals[c_start: c_end]) * gamma * tau * a

      advs[c_start: c_end] = a
      estimate_returns[c_start: c_end] = a + values[c_start: c_end]

    self._advs = advs
    self._estimate_returns = estimate_returns

  def discount_reward(self, last_value, gamma):
    """
    Compute the discounted reward to estimate return and advantages
    """
    advs = torch.zeros_like(self._rewards).to(self.device)
    estimate_returns = torch.zeros_like(self._rewards).to(self.device)

    r = last_value
    # self.time_limit_filter: handled in sampling
    for t in reversed(range(len(self._rewards) // self.num_envs)):
      c_start, c_end = t * self.num_envs, (t + 1) * self.num_envs

      r = self._rewards[c_start: c_end] + \
          (1 - self._terminals[c_start: c_end]) * gamma * r

      advs[c_start: c_end] = r - self._values[c_start: c_end]
      estimate_returns[c_start: c_end] = r

    self._advs = advs
    self._estimate_returns = estimate_returns

  def one_iteration(self, batch_size, sample_key, shuffle, device=None):
    if device is None:
      device = self.device
    assert batch_size % self.num_envs == 0, \
        "batch size should be dividable by num_envs"

    indices = np.arange(self._max_replay_buffer_size)
    if shuffle:
      indices = np.random.permutation(self._max_replay_buffer_size)
    indices = torch.Tensor(indices).to(self.device).long()

    pos = 0
    while pos < self._max_replay_buffer_size:
      return_dict = {}

      for key in sample_key:
        return_dict[key] = torch.index_select(
            self.__getattribute__("_" + key),
            0,
            indices[pos: pos + batch_size]
        ).to(device)

      yield return_dict
      pos += batch_size


class OnPolicyReplayBuffer(OnPolicyReplayBufferBase, BaseReplayBuffer):
  pass
