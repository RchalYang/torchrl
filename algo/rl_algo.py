import copy
import time
from collections import deque
import numpy as np

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


    def pretrain(self):
        pass
    
    def eval(self):

        eval_env = copy.deepcopy(self.env)
        eval_env.eval()
        eval_env._reward_scale = 1

        done = False

        for _ in range(self.eval_episodes):

            eval_ob = eval_env.reset()
            rew = 0
            while not done:
                act = self.pf.eval( torch.Tensor( eval_ob ).to(self.device).unsqueeze(0) )
                eval_ob, r, done, _ = eval_env.step( act.detach().cpu().numpy() )
                rew += r
            self.episode_rewards.append(rew)
            done = False

    def update(self, batch):
        raise NotImplementedError

    def train(self):
        self.pretrain()
        self.logger.log("Finished Pretrain")

        self.current_step = 0
        ob = self.env.reset()

        for epoch in range( self.num_epochs ):
            
            start = time.time()
            for frame in range(self.epoch_frames):
                # Sample actions
                with torch.no_grad():
                    _, _, action, _ = self.pf.explore( torch.Tensor( ob ).to(self.device).unsqueeze(0) )
                action = action.detach().cpu().numpy()
                action = action[0]

                # Nan detected, stop training
                if np.isnan(action).any():
                    print("NaN detected. BOOM")
                    exit()
                # Obser reward and next obs
                next_ob, reward, done, _ = self.env.step(action)

                self.replay_buffer.add_sample(ob, action, reward, done, next_ob )
                if self.replay_buffer.num_steps_can_sample() > self.min_pool:
                    for _ in range( self.opt_times ):
                        batch = self.replay_buffer.random_batch(self.batch_size)
                        infos = self.update( batch )

                        self.logger.add_update_info( infos )
                
                ob = next_ob
                self.current_step += 1
                if done or self.current_step >= self.max_episode_frames:
                    ob = self.env.reset()
                    self.current_step = 0
            
            self.eval()

            total_frames = (epoch + 1) * self.epoch_frames + self.pretrain_frames
            
            infos = {}
            infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
            
            self.logger.add_epoch_info(epoch, total_frames, time.time() - start, infos )
            
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
