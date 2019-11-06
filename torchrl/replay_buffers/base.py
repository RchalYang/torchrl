import numpy as np

class BaseReplayBuffer():
    """
    Basic Replay Buffer
    """
    def __init__(self, 
        max_replay_buffer_size 
    ):
        self._max_replay_buffer_size = max_replay_buffer_size
        self._top = 0
        self._size = 0

    def add_sample(self, sample_dict, env_rank, **kwargs):
        for key in sample_dict:
            if not hasattr( self, "_" + key ):
                self.__setattr__( "_" + key,
                    np.zeros( (self._max_replay_buffer_size,) + np.shape(sample_dict[key]) ) )
            self.__getattribute__( "_" + key )[self._top] = sample_dict[key] 
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size, sample_key):
        indices = np.random.randint(0, self._size, batch_size)
        return_dict = {}
        for key in sample_key:
            return_dict[key] = self.__getattribute__("_"+key) [indices]
        return return_dict

    def num_steps_can_sample(self):
        return self._size
