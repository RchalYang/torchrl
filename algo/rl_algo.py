import copy
import time
from collections import deque
import numpy as np
import math
import torch

import pytorch_util as ptu

class RLAlgo():
    """
    Base RL Algorithm Framework
    """
    def __init__(self,
        env = None,
        replay_buffer = None,
        logger = None,

        discount=0.99,
        pretrain_frames=0,
        num_epochs = 3000,
        epoch_frames = 1000,
        max_episode_frames = 999,
        
        batch_size = 128,
        min_pool = 0,

        target_hard_update_period = 1000,
        use_soft_update = True,
        tau = 0.001,
        opt_times = 1,

        device = 'cpu',
        
        eval_episodes = 1,

    ):

        self.env = env
        self.replay_buffer = replay_buffer
        
        # device specification
        self.device = device

        # environment relevant information
        self.discount = discount
        self.pretrain_frames = pretrain_frames
        self.num_epochs = num_epochs
        self.epoch_frames = epoch_frames
        self.max_episode_frames = max_episode_frames

        # target_network update information
        
        self.target_hard_update_period = target_hard_update_period
        self.use_soft_update = use_soft_update
        self.tau = tau

        # training information
        self.opt_times = opt_times
        self.batch_size = batch_size
        self.min_pool = min_pool

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

    def get_pretrain_actions(self, ob):
        out = self.pretrain_pf.explore( torch.Tensor( ob ).to(self.device).unsqueeze(0) )
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

        sample_dict = {
            "obs":ob,
            "next_obs": next_ob,
            "actions":action,
            "rewards": [reward],
            "terminals": [done]
        }

        self.replay_buffer.add_sample( sample_dict )

        return next_ob, done, reward, info

    def start_episode(self):
        pass

    def start_epoch(self):
        pass

    def finish_episode(self):
        pass

    def finish_epoch(self):
        return {}

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

                if self.replay_buffer.num_steps_can_sample() > max( self.min_pool, self.batch_size ):
                    for _ in range( self.opt_times ):
                        batch = self.replay_buffer.random_batch( self.batch_size, self.sample_key)
                        infos = self.update( batch )
                        self.logger.add_update_info( infos )

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
            
            finish_epoch_info = self.finish_epoch()
            eval_infos = self.eval()

            total_frames = (pretrain_epoch + 1) * self.epoch_frames
            
            infos = {}
            infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
            infos.update(eval_infos)
            infos.update(finish_epoch_info)

            self.logger.add_epoch_info(pretrain_epoch, total_frames, time.time() - start, infos )

    def train(self):
        self.pretrain()
        self.logger.log("Finished Pretrain")

        self.current_step = 0
        ob = self.env.reset()

        self.start_episode()
        
        for epoch in range( self.num_epochs ):
            
            start = time.time()
            for frame in range(self.epoch_frames):
                # Sample actions
                next_ob, done, reward, info = self.take_actions( ob, self.get_actions )

                if self.replay_buffer.num_steps_can_sample() > max( self.min_pool, self.batch_size ):
                    for _ in range( self.opt_times ):
                        batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
                        infos = self.update( batch )
                        self.logger.add_update_info( infos )
                
                ob = next_ob
                self.current_step += 1
                if done or self.current_step >= self.max_episode_frames:
                    ob = self.env.reset()
                    self.finish_episode()
                    self.start_episode()
                    self.current_step = 0
            
            finish_epoch_info = self.finish_epoch()
            eval_infos = self.eval()

            total_frames = (epoch + 1) * self.epoch_frames + self.pretrain_frames
            
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
                ptu.soft_update_from_to(net, target_net, self.tau)
        else:
            if self.training_update_num % self.target_hard_update_period == 0:
                for net, target_net in self.target_networks:
                    ptu.copy_model_params_from_to(net, target_net)

    @property
    def networks(self):
        return [
        ]
    
    def target_networks(self):
        return [
        ]
    
    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
