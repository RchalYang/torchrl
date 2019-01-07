from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from gym import Wrapper
from gym.spaces import Box
import gym
import numpy as np
from gym import Env

class NormalizedContinuousEnv(Env):
    """
    Normalized Action      => [ -1, 1 ]
    Normalized Observation => Optional, Use Momentum
    """
    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_norm = False,
            obs_alpha = 0.001
    ):
        self._wrapped_env = env
        self._obs_norm = obs_norm
        if self._obs_norm:
            self._obs_alpha = obs_alpha
            self._obs_mean = np.zeros(env.observation_space.shape[0])
            self._obs_var = np.ones(env.observation_space.shape[0])

        self._reward_scale = reward_scale

        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)
        self.observation_space = self._wrapped_env.observation_space

        self.training = False
    
    def _update_obs_estimate(self, obs):
        
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(obs - self._obs_mean)

    def _apply_normalize_obs(self, raw_obs):
        if not self._obs_norm:
            return raw_obs
        if self.training:
            self._update_obs_estimate(raw_obs)
        return (raw_obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        raw_next_obs, reward, done, info = self._wrapped_env.step(scaled_action)
        next_obs = self._apply_normalize_obs(raw_next_obs)
        return next_obs, reward * self._reward_scale, done, info
    
    def reset(self):
        raw_obs = self._wrapped_env.reset()
        return self._apply_normalize_obs(raw_obs)

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
