import numpy as np

import torch

from algo.rl_algo import RLAlgo

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

    def add_sample(self, ob, act, next_ob, reward, done):
        value = self.vf(torch.Tensor(ob).to(self.device).unsqueeze(0))
        value = value.item()

        sample_dict = {
            "obs":ob,
            "next_obs": next_ob,
            "acts":act,
            "values": [value],
            "rewards": [reward],
            "terminals": [done]
        }

        if done or self.current_step >= self.max_episode_frames:
            if not done and self.current_step >= self.max_episode_frames:                 
                last_ob = torch.Tensor( next_ob ).to(self.device).unsqueeze(0) 
                last_value = self.vf( last_ob ).item()
                
                sample_dict["terminals"] = [True]
                sample_dict["rewards"] = [ reward + self.discount * last_value ]

            next_ob = self.env.reset()
            self.finish_episode()
            self.start_episode()
            self.current_step = 0

        self.replay_buffer.add_sample( sample_dict )

        return next_ob

    def update_per_epoch(self):
        sample = self.replay_buffer.last_sample( ['next_obs', 'terminals' ] )
        last_value = 0
        if not sample['terminals']:
            last_ob = torch.Tensor( sample['next_obs'] ).to(self.device).unsqueeze(0) 
            last_value = self.vf( last_ob ).item()
        
        if self.gae:
            self.replay_buffer.generalized_advantage_estimation(last_value, self.discount, self.tau)
        else:
            self.replay_buffer.discount_reward(last_value, self.discount)

        # print(self.replay_buffer._advs.mean())
        # print(self.replay_buffer._advs.std())
        # print(self.replay_buffer._advs)

        self.replay_buffer._advs = ( self.replay_buffer._advs - self.replay_buffer._advs.mean() ) / \
            ( self.replay_buffer._advs.std() + 1e-8 )
            
        # print(self.replay_buffer._advs.shape)
        # print(self.replay_buffer._advs)
        # exit()
        # advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for batch in self.replay_buffer.one_iteration(self.batch_size, self.sample_key, self.shuffle):
            infos = self.update( batch )
            self.logger.add_update_info( infos )

    @property
    def networks(self):
        return [
            self.pf,
            self.vf
        ]
