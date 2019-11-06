import numpy as np
from .base import BaseReplayBuffer

class OnPolicyReplayBuffer(BaseReplayBuffer):
    """
    Replay Buffer for On Policy algorithms
    """

    def last_sample(self, sample_key):
        return_dict = {}
        for key in sample_key:
            return_dict[key] = self.__getattribute__("_"+key)[ self._max_replay_buffer_size - 1 ]
        return return_dict

    def generalized_advantage_estimation(self, last_value, gamma, tau):
        """
        use GAE to process rewards
        P.S: only one round no need to process done info
        """
        A = 0
        advs = []
        estimate_returns = []

        values= np.concatenate( [self._values, np.array([[last_value]])], 0 )

        for t in reversed(range(len(self._rewards))):
            delta = self._rewards[t] + ( 1 - self._terminals[t] ) * gamma * values[t + 1] - values[t]
            A = delta + ( 1 - self._terminals[t] ) * gamma * tau * A
            advs.insert( 0, A )
            estimate_returns.insert( 0, A + values[t] )

        self._advs = np.array(advs)
        self._estimate_returns = np.array(estimate_returns)

    def discount_reward(self, last_value, gamma):
        """
        Compute the discounted reward to estimate return and advantages
        """
        advs = []
        estimate_returns = []

        R = last_value
        for t in reversed(range(len(self._rewards))):
            R = self._rewards[t] + ( 1 - self._terminals[t] ) * gamma * R
            advs.insert( 0, R - self._values[t] )
            estimate_returns.insert( 0, R )

        self._advs = np.array(advs)
        self._estimate_returns = np.array(estimate_returns)

    def one_iteration(self, batch_size, sample_key, shuffle):
        
        indices = np.arange(self._max_replay_buffer_size)
        if shuffle:
            indices = np.random.permutation( self._max_replay_buffer_size )

        pos = 0
        while pos < self._max_replay_buffer_size:
            return_dict = {}
            for key in sample_key:
                return_dict[key] = self.__getattribute__("_"+key)[ indices[pos: pos+batch_size] ]
            yield return_dict
            pos += batch_size