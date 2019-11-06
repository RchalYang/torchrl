import gym
import numpy as np

from .base_wrapper import BaseWrapper

class NormObs(gym.ObservationWrapper, BaseWrapper):
    """
    Normalized Observation => Optional, Use Momentum
    """
    def __init__( self, env, obs_alpha = 0.001 ):
        super(NormObs,self).__init__(env)
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(env.observation_space.shape[0])
        self._obs_var = np.ones(env.observation_space.shape[0])

    def _update_obs_estimate(self, obs):
        
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(obs - self._obs_mean)

    def _apply_normalize_obs(self, raw_obs):
        if self.training:
            self._update_obs_estimate(raw_obs)
        return (raw_obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def observation(self, observation):
        return self._apply_normalize_obs(observation)

class NormAct(gym.ActionWrapper, BaseWrapper):
    """
    Normalized Action      => [ -1, 1 ]
    """
    def __init__(self, env):
        super(NormAct, self).__init__(env)
        ub = np.ones(self.env.action_space.shape)
        self.action_space = gym.spaces.Box(-1 * ub, ub)
    
    def action(self, action):
        lb = self.env.action_space.low
        ub = self.env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        return np.clip(scaled_action, lb, ub)

class RewardShift(gym.RewardWrapper, BaseWrapper):
    def __init__(self, env, reward_scale = 1):
        super(RewardShift, self).__init__(env)
        self._reward_scale = reward_scale
    
    def reward(self, reward):
        if self.training:
            return self._reward_scale * reward
        else:
            return reward
