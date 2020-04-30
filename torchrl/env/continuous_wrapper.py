import gym
import numpy as np

from .base_wrapper import BaseWrapper


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
