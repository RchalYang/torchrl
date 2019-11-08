import numpy as np

import torch

from torchrl.algo.rl_algo import RLAlgo

class OnRLAlgo(RLAlgo):
    """
    Base RL Algorithm Framework
    """
    def __init__(
        self, 
        # continuous,
        shuffle = True,
        tau = None,
        gae = True,
        **kwargs
    ):
        super(OnRLAlgo, self).__init__(**kwargs )
        self.sample_key = [ "obs", "acts", "advs", "estimate_returns" ]
        self.shuffle = shuffle
        # self.continuous = continuous
        self.tau = tau
        self.gae = gae

    def update_per_epoch(self):
        sample = self.replay_buffer.last_sample( ['next_obs', 'terminals' ] )
        # last_value = 0
        # print(sample['terminals'])
        # if not sample['terminals']:
        last_ob = torch.Tensor( sample['next_obs'] ).to(self.device)
        if self.collector.worker_nums == 1:
            last_ob = last_ob.unsqueeze(0)
        last_value = self.vf( last_ob ).cpu().numpy()
        last_value = last_value * (1 - sample['terminals'])
        
        if self.gae:
            self.replay_buffer.generalized_advantage_estimation(last_value, self.discount, self.tau)
        else:
            self.replay_buffer.discount_reward(last_value, self.discount)
        
        # print(self.replay_buffer._advs)
        # exit()
        self.replay_buffer._advs = ( self.replay_buffer._advs - self.replay_buffer._advs.mean() ) / \
            ( self.replay_buffer._advs.std() + 1e-8 )
            
        for batch in self.replay_buffer.one_iteration(self.batch_size, self.sample_key, self.shuffle):
            infos = self.update( batch )
            self.logger.add_update_info( infos )

    @property
    def networks(self):
        return [
            self.pf,
            self.vf
        ]
