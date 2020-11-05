import numpy as np
from .base import BaseReplayBuffer


class VecReplayBuffer(BaseReplayBuffer):
    """
    Replay Buffer That Support Vector Env
    """
    def __init__(self, env_nums, **kwargs):
        """
        Initialize the buffer.

        Args:
            self: (todo): write your description
            env_nums: (int): write your description
        """
        super().__init__(**kwargs)
        self.env_nums = env_nums
        self._max_replay_buffer_size = self._max_replay_buffer_size // \
            self.env_nums

    def random_batch(self, batch_size, sample_key):
        """
        Return a batch of the given batch.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
            sample_key: (str): write your description
        """
        assert batch_size % self.env_nums == 0, \
            "batch size should be dividable by worker_nums"
        batch_size //= self.env_nums
        size = self.num_steps_can_sample()
        indices = np.random.randint(0, size, batch_size)
        return_dict = {}
        for key in sample_key:
            return_dict[key] = self.__getattribute__("_"+key)[indices].reshape(
                (batch_size * self.env_nums, -1))
        return return_dict
