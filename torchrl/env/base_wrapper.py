"""Basic Wrapper for Environments."""
import gym
import numpy as np
import copy
import torch
from torchrl.utils.normalizer import TorchNormalizer, Normalizer


class BaseWrapper(gym.Wrapper):
  """Basic Env Wrapper"""

  def __init__(self, env):
    super().__init__(env)
    self._wrapped_env = env
    self.training = True

  def train(self):
    if isinstance(self._wrapped_env, BaseWrapper):
      self._wrapped_env.train()
    self.training = True

  def eval(self):
    if isinstance(self._wrapped_env, BaseWrapper):
      self._wrapped_env.eval()
    self.training = False

  def reset(self, **kwargs):
    return self._wrapped_env.reset()

  def __getattr__(self, attr):
    if attr == '_wrapped_env':
      raise AttributeError()
    return getattr(self._wrapped_env, attr)

  def copy_state(self, source_env):
    pass


class RewardShift(gym.RewardWrapper, BaseWrapper):
  def __init__(self, env, reward_scale=1):
    super().__init__(env)
    self._reward_scale = reward_scale

  def reward(self, reward):
    if self.training:
      return self._reward_scale * reward
    else:
      return reward


class NormObs(gym.ObservationWrapper, BaseWrapper):
  """
  Normalized Observation with Pytorch
  """

  def __init__(self, env, epsilon=1e-4, clipob=10.):
    super().__init__(env)
    self.count = epsilon
    self.clipob = clipob
    self._obs_normalizer = Normalizer(
        env.observation_space.shape, clip=clipob
    )

  def observation(self, observation):
    if self.training:
      self._obs_normalizer.update_estimate(observation)
    return self._obs_normalizer.filt(observation)

  def partial_reset(self, *args, **kwargs):
    return self.observation(self._wrapped_env.partial_reset(*args, **kwargs))


class TorchNormObs(NormObs):
  """
  Normalized Observation with Pytorch
  """

  def __init__(self, env, device, epsilon=1e-4, clipob=10.):
    super().__init__(
        env, epsilon=epsilon, clipob=clipob
    )
    self._obs_normalizer = TorchNormalizer(
        env.observation_space.shape, device, clip=clipob
    )


class NormRet(BaseWrapper):
  """
  Normalized Return => Optional, Use Momentum
  """

  def __init__(self, env, discount=0.99):
    super().__init__(env)
    self.ret = 0
    self.discount = discount
    self.epsilon = 1e-4
    self.ret_normalizer = TorchNormalizer(
        (1,), self._wrapped_env.device
    )

  def step(self, action):
    obs, rews, done, infos = self._wrapped_env.step(action)
    if self.training:
      self.ret = self.ret * self.discount + rews
      # if self.ret_rms:
      self.ret_normalizer.update_estimate(self.ret)
      rews = rews / torch.sqrt(self.ret_normalizer._var + self.epsilon)
      self.ret *= (1 - done)
    return obs, rews, done, infos

  def reset(self, **kwargs):
    self.ret = 0
    return self._wrapped_env.reset(**kwargs)


class TimeLimitAugment(BaseWrapper):
  """Check Trajectory is ended by time limit or not"""

  def step(self, action):
    obs, rew, done, info = self._wrapped_env.step(action)
    info['time_limit'] = self._wrapped_env._max_episode_steps == self._wrapped_env._elapsed_steps
    return obs, rew, done, info

  def reset(self, **kwargs):
    return self._wrapped_env.reset(**kwargs)


class TorchEnv(gym.ObservationWrapper, BaseWrapper):
  """Convert env using numpy to env using torch"""

  def __init__(self, env, device, dtype=torch.float32):
    super().__init__(env)
    self.device = device
    self.dtype = dtype

  def observation(self, observation):
    return torch.tensor(
        observation, device=self.device, dtype=self.dtype
    )

  def step(self, action):
    obs, rew, done, info = self._wrapped_env.step(
        action.detach().cpu().numpy())
    obs = self.observation(obs)
    done = torch.tensor(done, device=self.device, dtype=torch.bool)
    rew = torch.tensor(rew, device=self.device, dtype=self.dtype)
    for key in info.keys():
      info[key] = torch.tensor(
          info[key], device=self.device
      )
    return obs, rew, done, info

  def partial_reset(self, *args, **kwargs):
    return self.observation(self._wrapped_env.partial_reset(*args, **kwargs))
