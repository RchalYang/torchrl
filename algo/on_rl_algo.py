import numpy as np

import torch

from .rl_algo import RLAlgo

class OnRLAlgo(RLAlgo):
    """
    Base RL Algorithm Framework
    """
    def __init__(self, **kwargs):
        super(OnRLAlgo, self).__init__(self, **kwargs )
        self.sample_key = [ "obs", "actions", "advs", "estimate_returns" ]

    def take_actions(self, ob, action_func):
        
        action = action_func( ob )

        value = self.vf( torch.Tensor( ob ).to(self.device).unsqueeze(0) )
        value = value.item()

        if type(action) is not int:
            if np.isnan(action).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, reward, done, info = self.env.step(action)

        sample_dict = {
            "obs":ob,
            "next_obs": next_ob,
            "actions":action,
            "values": [value],
            "rewards": [reward],
            "terminals": [done]
        }

        self.replay_buffer.add_sample( sample_dict )

        return next_ob, done, reward, info

    def update_per_epoch(self):

        sample = self.replay_buffer.last_sample( ['obs', 'terminals' ] )
        last_value = 0
        if not sample['terminals']:
            last_ob = torch.Tensor( sample['obs'] ).to(self.device).unsqueeze(0) 
            last_value = self.vf( last_ob ).item()
        
        if self.gae:
            self.replay_buffer.generalized_advantage_estimation(last_value)
        else:
            self.replay_buffer.discount_reward(last_value)

        for batch in self.replay_buffer.one_iteration(self.batch_size, self.sample_key):
            # batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
            infos = self.update( batch )
            self.logger.add_update_info( infos )

    @property
    def networks(self):
        return [
            self.pf,
            self.vf
        ]
    