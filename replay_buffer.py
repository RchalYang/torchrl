import torch
import numpy as np

class SimpleReplayBuffer():
    def __init__(
            self, max_replay_buffer_size #, observation_dim, action_dim,
    ):
        self._max_replay_buffer_size = max_replay_buffer_size
        self._top = 0
        self._size = 0

    def add_sample(self, sample_dict, **kwargs):
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

class MemoryEfficientReplayBuffer(SimpleReplayBuffer):
    def add_sample(self, sample_dict, **kwargs):
        for key in sample_dict:
            if not hasattr( self, "_" + key ):
                self.__setattr__( "_" + key,
                    [ None for _ in range(self._max_replay_buffer_size) ] ) 
            self.__getattribute__( "_" + key )[self._top] = sample_dict[key] 
        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def encode_batchs(self, key, batch_indices):
        pointer = self.__getattribute__("_"+key)
        data = []
        for idx in batch_indices:
            data.append( pointer[idx] )
        return np.array( data, dtype = np.float )

    def random_batch(self, batch_size, sample_key):
        indices = np.random.randint(0, self._size, batch_size)
        return_dict = {}
        for key in sample_key:
            # return_dict[key] = self.__getattribute__("_"+key) [indices]
            return_dict[key] = self.encode_batchs(key, indices)

        return return_dict
