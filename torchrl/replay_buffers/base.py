"""Basic Replay Buffer"""
import torch


class BaseReplayBuffer():
  """
  Basic Replay Buffer
  """

  def __init__(
      self,
      max_replay_buffer_size: int,
      num_envs: int = 1,
      on_gpu: bool = False,
      device: str = "cpu",
      time_limit_filter: bool = False
  ):
    self.num_envs = num_envs
    self._max_replay_buffer_size = max_replay_buffer_size
    assert self._max_replay_buffer_size % self.num_envs == 0
    self._top = 0
    self._size = 0
    self.time_limit_filter = time_limit_filter
    self.on_gpu = on_gpu
    self.device = device
    if self.on_gpu:
      assert self.device.startswith("cuda")

  def add_sample(self, sample_dict):
    for key in sample_dict:
      if not hasattr(self, "_" + key):
        self.__setattr__(
            "_" + key,
            torch.zeros(
                (self._max_replay_buffer_size,) +
                sample_dict[key].shape[1:]
            ).to(self.device)
        )
      # print(key)
      # print(sample_dict[key].shape)
      # print(self.__getattribute__("_" + key).shape)
      self.__getattribute__("_" + key)[
          self._top: self._top + self.num_envs, ...
      ] = sample_dict[key].detach()
    self._advance()

  def terminate_episode(self):
    pass

  def _advance(self):
    self._top = (self._top + self.num_envs) % self._max_replay_buffer_size
    if self._size < self._max_replay_buffer_size:
      self._size += self.num_envs

  def random_batch(self, batch_size, sample_key, device=None):
    if device is None:
      device = self.device
    assert batch_size % self.num_envs == 0, \
        "batch size should be dividable by num_envs"
    size = self.num_steps_can_sample()
    indices = torch.randint(0, size, (batch_size,))
    return_dict = {}
    for key in sample_key:
      return_dict[key] = self.__getattribute__("_" + key)[indices].to(
          self.device
      )
    return return_dict

  def num_steps_can_sample(self):
    return self._size
