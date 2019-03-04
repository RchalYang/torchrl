import numpy as np

import torch

from .rl_algo import RLAlgo

class OnRLAlgo(RLAlgo):
    """
    Base RL Algorithm Framework
    """
    def __init__(
        self, 
        continuous,
        shuffle = True,
        tau = None,
        gae = True,
        **kwargs
    ):
        super(OnRLAlgo, self).__init__(**kwargs )
        self.sample_key = [ "obs", "actions", "advs", "estimate_returns" ]
        self.shuffle = shuffle
        self.continuous = continuous
        self.tau = tau
        self.gae = gae

    def take_actions(self, ob, action_func):
        
        action = action_func( ob )

        value = self.vf( torch.Tensor( ob ).to(self.device).unsqueeze(0) )
        value = value.item()

        if not self.continuous:
            action = action[0]

        if type(action) is not int:
            if np.isnan(action).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, reward, done, info = self.env.step(action)

        self.current_step += 1

        sample_dict = {
            "obs":ob,
            "next_obs": next_ob,
            "actions":action,
            "values": [value],
            "rewards": [reward],
            "terminals": [done]
        }

        if done or self.current_step >= self.max_episode_frames:
            if self.current_step >= self.max_episode_frames:
                
                last_ob = torch.Tensor( sample['obs'] ).to(self.device).unsqueeze(0) 
                last_value = self.vf( last_ob ).item()
                
                sample_dict["terminals"] = [True]
                sample_dict["rewards"] = [ reward + self.discount * last_value ]
                
            next_ob = self.env.reset()
            self.finish_episode()
            self.start_episode()
            self.current_step = 0

        self.replay_buffer.add_sample( sample_dict )

        return next_ob, done, reward, info

    def update_per_epoch(self):

        # sample = self.replay_buffer.last_sample( ['obs', 'terminals' ] )
        last_value = 0
        # if not sample['terminals']:
        #     last_ob = torch.Tensor( sample['obs'] ).to(self.device).unsqueeze(0) 
        #     last_value = self.vf( last_ob ).item()
        
        if self.gae:
            self.replay_buffer.generalized_advantage_estimation(last_value, self.discount, self.tau)
        else:
            self.replay_buffer.discount_reward(last_value, self.discount)
            
        for batch in self.replay_buffer.one_iteration(self.batch_size, self.sample_key, self.shuffle):
            # batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
            infos = self.update( batch )
            self.logger.add_update_info( infos )

    @property
    def networks(self):
        return [
            self.pf,
            self.vf
        ]
    
