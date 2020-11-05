import gym
import numpy as np

from .base_wrapper import BaseWrapper


# Check Trajectory is ended by time limit or not
class TimeLimitAugment(gym.Wrapper):
    def step(self, action):
        """
        Perform a step.

        Args:
            self: (todo): write your description
            action: (int): write your description
        """
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['time_limit'] = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        """
        Reset the environment.

        Args:
            self: (todo): write your description
        """
        return self.env.reset(**kwargs)


class NormAct(gym.ActionWrapper, BaseWrapper):
    """
    Normalized Action      => [ -1, 1 ]
    """
    def __init__(self, env):
        """
        Initialize actions.

        Args:
            self: (todo): write your description
            env: (todo): write your description
        """
        super(NormAct, self).__init__(env)
        ub = np.ones(self.env.action_space.shape)
        self.action_space = gym.spaces.Box(-1 * ub, ub)

    def action(self, action):
        """
        Action : math function.

        Args:
            self: (todo): write your description
            action: (str): write your description
        """
        lb = self.env.action_space.low
        ub = self.env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        return np.clip(scaled_action, lb, ub)
