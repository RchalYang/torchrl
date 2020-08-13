import gym
import numpy as np
import copy


class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BaseWrapper, self).__init__(env)
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

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def copy_state(self, source_env):
        pass


class RewardShift(gym.RewardWrapper, BaseWrapper):
    def __init__(self, env, reward_scale=1):
        super(RewardShift, self).__init__(env)
        self._reward_scale = reward_scale

    def reward(self, reward):
        if self.training:
            return self._reward_scale * reward
        else:
            return reward


def update_mean_var_count(
        mean, var, count,
        batch_mean, batch_var, batch_count):
    """
    Imported From OpenAI Baseline
    """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormObs(gym.ObservationWrapper, BaseWrapper):
    """
    Normalized Observation => Optional, Use Momentum
    """
    def __init__(self, env, epsilon=1e-4, clipob=10.):
        super(NormObs, self).__init__(env)
        self.count = epsilon
        self.clipob = clipob
        self._obs_mean = np.zeros(env.observation_space.shape[0])
        self._obs_var = np.ones(env.observation_space.shape[0])

    def _update_obs_estimate(self, obs):
        if len(obs.shape) == 1:
            self._obs_mean, self._obs_var, self.count = update_mean_var_count(
                self._obs_mean, self._obs_var,
                self.count, obs, np.zeros_like(obs), 1)
        elif len(obs.shape) == 2:
            self._obs_mean, self._obs_var, self.count = update_mean_var_count(
                self._obs_mean, self._obs_var,
                self.count, np.mean(obs, axis=0),
                np.var(obs, axis=0), obs.shape[0])
        else:
            raise ValueError("Wrong shape for obs")

    def copy_state(self, source_env):
        self._obs_mean = copy.deepcopy(source_env._obs_mean)
        self._obs_var = copy.deepcopy(source_env._obs_var)

    def _apply_normalize_obs(self, raw_obs):
        if self.training:
            self._update_obs_estimate(raw_obs)
        if len(raw_obs.shape) == 1:
            return np.clip(
                (raw_obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8),
                -self.clipob, self.clipob)
        elif len(raw_obs.shape) == 2:
            return np.clip(
                (raw_obs - self._obs_mean[np.newaxis, ...]) / (
                    np.sqrt(self._obs_var[np.newaxis, ...]) + 1e-8),
                -self.clipob, self.clipob)

    def observation(self, observation):
        return self._apply_normalize_obs(observation)


class NormRet(BaseWrapper):
    def __init__(self, env, discount=0.99, epsilon=1e-4):
        super(NormRet, self).__init__(env)
        self._ret = 0
        self.count = 1e-4
        self.ret_mean = 0
        self.ret_var = 1
        self.discount = discount
        self.epsilon = 1e-4

    def step(self, act):
        obs, rews, done, infos = self.env.step(act)
        if self.training:
            self.ret = self.ret * self.discount + rews
            # if self.ret_rms:
            self.ret_mean, self.ret_var, self.count = update_mean_var_count(
                self.ret_mean, self.ret_var, self.count, self.ret, 0, 1)
            rews = rews / np.sqrt(self.ret_var + self.epsilon)
            self.ret *= (1-done)
        return obs, rews, done, infos

    def reset(self, **kwargs):
        self.ret = 0
        return self.env.reset(**kwargs)


# Check Trajectory is ended by time limit or not
class TimeLimitAugment(BaseWrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info['time_limit'] = done \
            and self.env._max_episode_steps == self.env._elapsed_steps
        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
