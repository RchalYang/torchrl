import numpy as np
from .base import BaseReplayBuffer
from .shared.base import SharedBaseReplayBuffer
from .vec import VecReplayBuffer


class OnPolicyReplayBufferBase:
    """
    Replay Buffer for On Policy algorithms
    """
    def last_sample(self, sample_key):
        """
        Return the last sample of a sample.

        Args:
            self: (todo): write your description
            sample_key: (str): write your description
        """
        return_dict = {}
        for key in sample_key:
            return_dict[key] = self.__getattribute__("_"+key)[
                self._max_replay_buffer_size - 1]
        return return_dict

    def generalized_advantage_estimation(self, last_value, gamma, tau):
        """
        use GAE to process rewards
        """
        A = 0
        advs = []
        estimate_returns = []
        values = np.concatenate([self._values, np.array([last_value])], 0)
        if self.time_limit_filter:
            for t in reversed(range(len(self._rewards))):
                delta = self._rewards[t] + \
                    (1 - self._terminals[t]) * gamma * values[t + 1] - \
                    values[t]
                A = delta + (1 - self._terminals[t]) * gamma * tau * A
                A = A * (1 - self._time_limits[t])
                advs.insert(0, A)
                estimate_returns.insert(0, A + values[t])
        else:
            for t in reversed(range(len(self._rewards))):
                delta = self._rewards[t] + \
                    (1 - self._terminals[t]) * gamma * values[t + 1] - \
                    values[t]
                A = delta + (1 - self._terminals[t]) * gamma * tau * A
                advs.insert(0, A)
                estimate_returns.insert(0, A + values[t])

        self._advs = np.array(advs)
        self._estimate_returns = np.array(estimate_returns)

    def discount_reward(self, last_value, gamma):
        """
        Compute the discounted reward to estimate return and advantages
        """
        advs = []
        estimate_returns = []

        R = last_value
        if self.time_limit_filter:
            for t in reversed(range(len(self._rewards))):
                R = (self._rewards[t] + \
                     (1 - self._terminals[t]) * gamma * R *
                     (1 - self._time_limits[t])) + \
                    self._time_limits[t] * self._values[t]
                advs.insert(0, R - self._values[t])
                estimate_returns.insert(0, R)
        else:
            for t in reversed(range(len(self._rewards))):
                R = self._rewards[t] + \
                    (1 - self._terminals[t]) * gamma * R * \
                    (1 - self._time_limits[t])
                advs.insert(0, R - self._values[t])
                estimate_returns.insert(0, R)

        self._advs = np.array(advs)
        self._estimate_returns = np.array(estimate_returns)

    def one_iteration(self, batch_size, sample_key, shuffle):
        """
        Return an iterator that returns batches of data.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
            sample_key: (str): write your description
            shuffle: (bool): write your description
        """
        indices = np.arange(self._max_replay_buffer_size)
        if shuffle:
            indices = np.random.permutation(self._max_replay_buffer_size)

        pos = 0
        while pos < self._max_replay_buffer_size:
            return_dict = {}
            for key in sample_key:
                return_dict[key] = self.__getattribute__("_"+key)[
                    indices[pos: pos+batch_size]]
                data_shape = (-1, ) + return_dict[key].shape[2:]
                return_dict[key] = return_dict[key].reshape(data_shape)

            yield return_dict
            pos += batch_size


class OnPolicyReplayBuffer(OnPolicyReplayBufferBase, BaseReplayBuffer):
    pass


class SharedOnPolicyReplayBuffer(SharedBaseReplayBuffer, OnPolicyReplayBufferBase):
    pass


class VecOnPolicyReplayBuffer(OnPolicyReplayBufferBase, VecReplayBuffer):
    def one_iteration(self, batch_size, sample_key, shuffle):
        """
        Return an iterator that returns batches of data.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
            sample_key: (str): write your description
            shuffle: (bool): write your description
        """
        indices = np.arange(self._max_replay_buffer_size)
        if shuffle:
            indices = np.random.permutation(self._max_replay_buffer_size)

        assert batch_size % self.env_nums == 0, \
            "batch size should be dividable by worker_nums"
        batch_size //= self.env_nums

        pos = 0
        while pos < self._max_replay_buffer_size:
            return_dict = {}
            # print(pos)
            for key in sample_key:
                return_dict[key] = self.__getattribute__("_"+key)[
                    indices[pos: pos+batch_size]]
                data_shape = (-1, ) + return_dict[key].shape[2:]
                return_dict[key] = return_dict[key].reshape(data_shape)

            yield return_dict
            pos += batch_size
