import copy
import time
from collections import deque
import numpy as np

import torch

import torchrl.algo.utils as atu

import gym

import os
import os.path as osp

class RLAlgo():
    """
    Base RL Algorithm Framework
    """
    def __init__(self,
        env = None,
        replay_buffer = None,
        collector = None,
        logger = None,
        continuous = None,
        discount=0.99,
        num_epochs = 3000,
        epoch_frames = 1000,
        max_episode_frames = 999,
        batch_size = 128,
        device = 'cpu',
        train_render = False,
        eval_episodes = 1,
        eval_render = False,
        save_interval = 100,
        save_dir = None
    ):

        self.env = env

        self.continuous = isinstance(self.env.action_space, gym.spaces.Box)

        self.replay_buffer = replay_buffer
        self.collector = collector        
        # device specification
        self.device = device

        # environment relevant information
        self.discount = discount
        self.num_epochs = num_epochs
        self.epoch_frames = epoch_frames
        self.max_episode_frames = max_episode_frames

        self.train_render = train_render
        self.eval_render = eval_render

        # training information
        self.batch_size = batch_size
        self.training_update_num = 0
        self.sample_key = None

        # Logger & relevant setting
        self.logger = logger

        
        self.episode_rewards = deque(maxlen=10)
        self.training_episode_rewards = deque(maxlen=10)
        self.eval_episodes = eval_episodes

        self.save_interval = save_interval
        self.save_dir = save_dir
        if not osp.exists( self.save_dir ):
            os.mkdir( self.save_dir )

    def start_epoch(self):
        pass

    def finish_epoch(self):
        return {}

    def pretrain(self):
        pass
    
    def update_per_epoch(self):
        pass

    def snapshot(self, prefix, epoch):
        model_file_name="model_{}.pth".format(epoch)
        model_path=osp.join(prefix, model_file_name)
        torch.save(self.pf.state_dict(), model_path)

    def train(self):
        self.pretrain()
        total_frames = 0
        if hasattr(self, "pretrain_frames"):
            total_frames = self.pretrain_frames

        self.start_epoch()

        for epoch in range( self.num_epochs ):

            start = time.time()

            self.start_epoch()

            training_epoch_info =  self.collector.train_one_epoch()
            for reward in training_epoch_info["train_rewards"]:
                self.training_episode_rewards.append(reward)

            self.update_per_epoch()

            finish_epoch_info = self.finish_epoch()

            eval_infos = self.collector.eval_one_epoch()

            total_frames += self.collector.worker_nums * self.epoch_frames

            infos = {}

            for reward in eval_infos["eval_rewards"]:
                self.episode_rewards.append(reward)
            del eval_infos["eval_rewards"]

            infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Running_Training_Average_Rewards"] = np.mean(self.training_episode_rewards)
            infos.update(eval_infos)
            infos.update(finish_epoch_info)

            self.logger.add_epoch_info(epoch, total_frames, time.time() - start, infos )
            if epoch % self.save_interval == 0:
                self.snapshot(self.save_dir, epoch)
        self.snapshot(self.save_dir, "finish")
        self.collector.terminate()
        self.logger.csv_file.close()

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
        
    @property
    def target_networks(self):
        return [
        ]
    
    def to(self, device):
        for net in self.networks:
            net.to(device)
