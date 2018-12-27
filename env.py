from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from gym import Wrapper
from gym.spaces import Box
import gym
import numpy as np
from gym import Env

# def make_env( env, norm_ob = False, norm_action = True, reward_scale = 1 ):


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, 'log_diagnostics'):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

# class RunningMeanStd(object):
#     # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#     def __init__(self, epsilon=1e-4, shape=()):
#         self.mean = np.zeros(shape, 'float64')
#         self.var = np.ones(shape, 'float64')
#         self.count = epsilon

#     def update(self, x):
#         batch_mean = np.mean(x, axis=0)
#         batch_var = np.var(x, axis=0)
#         batch_count = x.shape[0]
#         self.update_from_moments(batch_mean, batch_var, batch_count)

#     def update_from_moments(self, batch_mean, batch_var, batch_count):
#         self.mean, self.var, self.count = update_mean_var_count_from_moments(
#             self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

# def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
#     delta = batch_mean - mean
#     tot_count = count + batch_count

#     new_mean = mean + delta * batch_count / tot_count
#     m_a = var * count
#     m_b = batch_var * batch_count
#     M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
#     new_var = M2 / tot_count
#     new_count = tot_count

#     return new_mean, new_var, new_count

# class NormalizeObs(gym.ObservationWrapper):
#     """
#     A vectorized wrapper that normalizes the observations
#     and returns from an environment.
#     """

#     def __init__(self, venv, ob=True, ret=True, gamma=0.99, epsilon=1e-8):
#         super(NormalizeObs, self).__init__( venv)
#         self.venv = venv
#         self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
#         self.gamma = gamma
#         self.epsilon = epsilon

#     def observation(self, obs):
#         #print("filted:")
#         if self.ob_rms:
#             if self.training:
#                 self.ob_rms.update(obs)
#             obs = (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon)
#             return obs
#         else:
#             return obs

#     def train(self):
#         self.training = True

#     def eval(self):
#         self.training = False

class RewardScale(gym.Wrapper):
    def __init__(self, env, reward_scale = 1):
        super(RewardScale, self).__init__(env)
        self.reward_scale = reward_scale
        self.venv = env

    def step(self, action):
        next_ob, r, done, info = self.venv.step(action)
        return next_ob, self.reward_scale * r, done, info

class NormalizeObs(gym.ObservationWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, mean = None, var = None, epsilon=1e-8):
        super(NormalizeObs, self).__init__(venv)
        self.venv = venv
        self.ob_mean = mean
        self.ob_var = var
        self.epsilon = epsilon

    def observation(self, obs):
        #print("filted:")
        if self.ob_mean is not None and self.ob_var is not None:
            obs = (obs - self.ob_mean) / np.sqrt(self.ob_var + self.epsilon)
            return obs
        else:
            return obs

# class NormalizeObs(gym.ObservationWrapper):
#     def __init__(
#             self,
#             env,
#             obs_alpha=0.001,
#     ):
#         super(NormalizeObs, self).__init__(env)
#         self.venv = env
#         self._obs_alpha = obs_alpha
#         self._obs_mean = np.zeros(env.observation_space.shape[0])
#         self._obs_var = np.ones(env.observation_space.shape[0])
    
#     def _update_obs_estimate(self, obs):
#         # flat_obs = self.venv.observation_space.flatten(obs)
#         self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * obs
#         self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(obs - self._obs_mean)

#     def _apply_normalize_obs(self, obs):
#         self._update_obs_estimate(obs)
#         return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

#     def observation(self, obs):
#         return self._apply_normalize_obs(obs)


class NormalizedActions(gym.ActionWrapper):

    def __init__(
            self,
            env,
            obs_alpha=0.001,
    ):
        super(NormalizedActions, self).__init__(env)
        self.venv = env
        ub = np.ones(env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def action(self, action):
        low_bound   = self.venv.action_space.low
        upper_bound = self.venv.action_space.high
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        #print("action called")        

        return action

    def reverse_action(self, action):
        low_bound   = self.venv.action_space.low
        upper_bound = self.venv.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        #print("reverse_action called")        

        return action
