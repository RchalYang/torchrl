import numpy as np


class BaseReplayBuffer():
    """
    Basic Replay Buffer
    """
    def __init__(
        self, max_replay_buffer_size, time_limit_filter=False,
    ):
        """
        Initialize the buffer.

        Args:
            self: (todo): write your description
            max_replay_buffer_size: (int): write your description
            time_limit_filter: (str): write your description
        """
        self.worker_nums = 1
        self.num_envs = 1
        self._max_replay_buffer_size = max_replay_buffer_size
        self._top = 0
        self._size = 0
        self.time_limit_filter = time_limit_filter

    def add_sample(self, sample_dict, env_rank=0, **kwargs):
        """
        Add sample to the sample.

        Args:
            self: (todo): write your description
            sample_dict: (dict): write your description
            env_rank: (int): write your description
        """
        for key in sample_dict:
            if not hasattr(self, "_" + key):
                self.__setattr__(
                    "_" + key,
                    np.zeros((self._max_replay_buffer_size, 1,) +
                             np.shape(sample_dict[key])))
            self.__getattribute__("_" + key)[self._top, 0] = sample_dict[key]
        self._advance()

    def terminate_episode(self):
        """
        Terminate the episode.

        Args:
            self: (todo): write your description
        """
        pass

    def _advance(self):
        """
        Advance the buffer.

        Args:
            self: (todo): write your description
        """
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size, sample_key):
        """
        Generate a random samples.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
            sample_key: (str): write your description
        """
        indices = np.random.randint(0, self._size, batch_size)
        return_dict = {}
        for key in sample_key:
            return_dict[key] = np.squeeze(
                self.__getattribute__("_"+key)[indices], axis=1)
        return return_dict

    def num_steps_can_sample(self):
        """
        Return the number of steps that have been modified.

        Args:
            self: (todo): write your description
        """
        return self._size
