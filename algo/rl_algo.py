import copy
import time
from collections import deque
import numpy as np

import torch

import algo.utils as atu

class RLAlgo():
    """
    Base RL Algorithm Framework
    """
    def __init__(self,
        env = None,
        replay_buffer = None,
        logger = None,

        discount=0.99,

        num_epochs = 3000,
        epoch_frames = 1000,
        max_episode_frames = 999,
        
        batch_size = 128,
        
        device = 'cpu',
        
        eval_episodes = 1,

    ):

        self.env = env
        self.replay_buffer = replay_buffer
        
        # device specification
        self.device = device

        # environment relevant information
        self.discount = discount
        self.num_epochs = num_epochs
        self.epoch_frames = epoch_frames
        self.max_episode_frames = max_episode_frames

        # training information
        self.batch_size = batch_size

        # Logger & relevant setting
        self.logger = logger

        self.training_update_num = 0
        self.episode_rewards = deque(maxlen=10)
        self.eval_episodes = eval_episodes
        
        self.sample_key = [ "obs", "next_obs", "actions", "rewards", "terminals" ]

    def get_actions(self, ob):
        out = self.pf.explore( torch.Tensor( ob ).to(self.device).unsqueeze(0) )
        action = out["action"]
        action = action.detach().cpu().numpy()
        return action

    def take_actions(self, ob, action_func):
        
        action = action_func( ob )

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
            "rewards": [reward],
            "terminals": [done]
        }

        if done or self.current_step >= self.max_episode_frames:
            next_ob = self.env.reset()
            self.finish_episode()
            self.start_episode()
            self.current_step = 0

        self.replay_buffer.add_sample( sample_dict )

        return next_ob, done, reward, info

    def start_episode(self):
        self.current_step = 0

    def start_epoch(self):
        pass

    def finish_episode(self):
        pass

    def finish_epoch(self):
        return {}

    def pretrain(self):
        pass

    def update_per_timestep(self):
        pass
    
    def update_per_epoch(self):
        pass
        
    def train(self):
        self.pretrain()

        ob = self.env.reset()

        self.start_episode()
        
        for epoch in range( self.num_epochs ):
            
            start = time.time()
                    
            self.start_epoch()
            
            for frame in range(self.epoch_frames):
                # Sample actions
                next_ob, done, reward, info = self.take_actions( ob, self.get_actions )
                
                ob = next_ob
                
                self.update_per_timestep()

            self.update_per_epoch()
                    
            finish_epoch_info = self.finish_epoch()

            eval_infos = self.eval()

            total_frames = (epoch + 1) * self.epoch_frames
            if hasattr(self, "pretrain_frames"):
                total_frames += self.pretrain_frames
            
            infos = {}
            infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
            infos.update(eval_infos)
            infos.update(finish_epoch_info)
            
            self.logger.add_epoch_info(epoch, total_frames, time.time() - start, infos )
            
    def eval(self):

        eval_env = copy.deepcopy(self.env)
        eval_env.eval()
        eval_env._reward_scale = 1

        eval_infos = {}
        eval_rews = []

        done = False
        for _ in range(self.eval_episodes):

            eval_ob = eval_env.reset()
            rew = 0
            while not done:
                act = self.pf.eval( torch.Tensor( eval_ob ).to(self.device).unsqueeze(0) )
                eval_ob, r, done, _ = eval_env.step( act )
                rew += r

            eval_rews.append(rew)
            self.episode_rewards.append(rew)

            done = False

        eval_infos["Eval_Rewards_Average"] = np.mean( eval_rews )
        return eval_infos

    def update(self, batch):
        raise NotImplementedError

    def _update_target_networks(self):
        if self.use_soft_update:
            for net, target_net in self.target_networks:
                atu.soft_update_from_to(net, target_net, self.tau)
        else:
            if self.training_update_num % self.target_hard_update_period == 0:
                for net, target_net in self.target_networks:
                    atu.copy_model_params_from_to(net, target_net)

    @property
    def networks(self):
        return [
        ]
    
    def target_networks(self):
        return [
        ]
    
    def to(self, device):
        for net in self.networks:
            net.to(device)
