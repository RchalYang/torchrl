import time
import numpy as np
import math

import torch

from .rl_algo import RLAlgo

class OffRLAlgo(RLAlgo):
    """
    Base RL Algorithm Framework
    """
    def __init__(self,
        
        pretrain_frames=0,
        
        min_pool = 0,

        target_hard_update_period = 1000,
        use_soft_update = True,
        tau = 0.001,
        opt_times = 1,

        **kwargs
    ):
        super(OffRLAlgo, self).__init__(**kwargs)

        # environment relevant information
        self.pretrain_frames = pretrain_frames
        
        # target_network update information
        self.target_hard_update_period = target_hard_update_period
        self.use_soft_update = use_soft_update
        self.tau = tau

        # training information
        self.opt_times = opt_times
        self.min_pool = min_pool

    def get_pretrain_actions(self, ob):
        out = self.pretrain_pf.explore( torch.Tensor( ob ).to(self.device).unsqueeze(0) )
        action = out["action"]
        action = action.detach().cpu().numpy()
        return action

    def update_per_timestep(self):
        if self.replay_buffer.num_steps_can_sample() > max( self.min_pool, self.batch_size ):
            for _ in range( self.opt_times ):
                batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
                infos = self.update( batch )
                self.logger.add_update_info( infos )

    def pretrain(self):
        
        self.env.reset()
        self.env.train()

        self.current_step = 0
        ob = self.env.reset()

        self.start_episode()

        pretrain_epochs = math.ceil( self.pretrain_frames / self.epoch_frames)

        for pretrain_epoch in range( pretrain_epochs ):
            
            start = time.time()
            for step in range( self.epoch_frames):
            
                next_ob, done, reward, info = self.take_actions( ob, self.get_pretrain_actions )

                self.update_per_timestep()

                ob = next_ob
                self.current_step += 1
                if done or self.current_step >= self.max_episode_frames:
                    ob = self.env.reset()
                    self.finish_episode()
                    self.start_episode()
                    self.current_step = 0
                
                total_frames = pretrain_epoch * self.epoch_frames + step
                if total_frames > self.pretrain_frames:
                    break
            
            self.update_per_epoch()
                    
            finish_epoch_info = self.finish_epoch()

            eval_infos = self.eval()

            total_frames = (pretrain_epoch + 1) * self.epoch_frames
            
            infos = {}
            infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
            infos.update(eval_infos)
            infos.update(finish_epoch_info)

            self.logger.add_epoch_info(pretrain_epoch, total_frames, time.time() - start, infos )
        
        self.logger.log("Finished Pretrain")
