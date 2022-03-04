import time
from collections import deque
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torchrl.trainer.utils as atu
import torch.optim as optim

import gym
import os.path as osp
import pathlib
import pickle
from torchrl.agent import RLAgent
from torchrl.replay_buffers.base import BaseReplayBuffer


class RLTrainer():
    """
    Base RL Algorithm Framework
    """
    def __init__(
        self,
        agent: RLAgent = None,
        env: object = None,
        replay_buffer: BaseReplayBuffer = None,
        collector: object = None,
        logger: object = None,
        grad_clip: float = None,
        discount: float = 0.99,
        num_epochs: int = 3000,
        batch_size: int = 128,
        device: str = 'cpu',
        save_interval: int = 100,
        eval_interval: int = 1,
        save_dir: str = None,
        episodes_reward_record_length: int = 30,
        training_episode_rewards_record_length: int = 30,
        optimizer_class: optim.Optimizer = optim.Adam,
    ) -> None:

        self.env = env
        self.agent = agent

        self.continuous = isinstance(self.env.action_space, gym.spaces.Box)

        self.replay_buffer = replay_buffer
        self.collector = collector
        # device specification
        self.device = device

        # environment relevant information
        self.discount = discount
        self.num_epochs = num_epochs
        self.epoch_frames = self.collector.epoch_frames

        # training information
        self.batch_size = batch_size
        self.training_update_num = 0
        self.sample_key = None
        self.grad_clip = grad_clip

        # Logger & relevant setting
        self.logger = logger

        self.episode_rewards = deque(maxlen=episodes_reward_record_length)
        self.training_episode_rewards = deque(
            maxlen=training_episode_rewards_record_length
        )

        self.save_interval = save_interval
        self.save_dir = save_dir

        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.best_eval = None
        self.eval_interval = eval_interval

        self.explore_time = 0
        self.train_time = 0
        self.start = time.time()

        self.optimizer_class = optimizer_class

    def start_epoch(self) -> None:
        pass

    def finish_epoch(self) -> dict:
        return {}

    def pretrain(self) -> None:
        pass

    def pre_update_process(self) -> None:
        pass

    def update_per_epoch(self) -> None:
        pass

    def train(self) -> None:
        self.pretrain()
        total_frames = 0
        if hasattr(self, "pretrain_frames"):
            total_frames = self.pretrain_frames
        self.start_epoch()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            self.start_epoch()

            explore_start_time = time.time()
            training_epoch_info = self.collector.train_one_epoch()
            for reward in training_epoch_info["train_rewards"]:
                self.training_episode_rewards.append(reward)

            self.explore_time += time.time() - explore_start_time

            train_start_time = time.time()
            self.pre_update_process()
            self.update_per_epoch()
            self.train_time += time.time() - train_start_time

            finish_epoch_info = self.finish_epoch()

            total_frames += self.epoch_frames

            if epoch % self.eval_interval == 0:
                eval_start_time = time.time()
                eval_infos = self.collector.eval_one_epoch()
                eval_time = time.time() - eval_start_time

                infos = {}

                for reward in eval_infos["eval_rewards"]:
                    self.episode_rewards.append(reward)

                if self.best_eval is None or \
                   (np.mean(eval_infos["eval_rewards"]) > self.best_eval):
                    self.best_eval = np.mean(eval_infos["eval_rewards"])
                    self.snapshot(self.save_dir, 'best')
                del eval_infos["eval_rewards"]

                infos["Running_Average_Rewards"] = np.mean(
                    self.episode_rewards)
                infos["Train_Epoch_Reward"] = \
                    training_epoch_info["train_epoch_reward"]
                infos["Running_Training_Average_Rewards"] = np.mean(
                    self.training_episode_rewards)
                infos["Explore_Time"] = self.explore_time
                infos["Train___Time"] = self.train_time
                infos["Eval____Time"] = eval_time
                self.explore_time = 0
                self.train_time = 0
                infos.update(eval_infos)
                infos.update(finish_epoch_info)

                self.logger.add_epoch_info(
                    epoch, total_frames, time.time() - self.start, infos)
                self.start = time.time()

            if epoch % self.save_interval == 0:
                self.snapshot(self.save_dir, str(epoch))

        self.snapshot(self.save_dir, "finish")
        self.collector.terminate()
        self.logger.finish()

    def update(
        self,
        batch: object
    ) -> None:
        raise NotImplementedError

    def _update_target_networks(self) -> None:
        if self.use_soft_update:
            for net, target_net in self.agent.target_networks:
                atu.soft_update_from_to(net, target_net, self.tau)
        else:
            if self.training_update_num % self.target_hard_update_period == 0:
                for net, target_net in self.agent.target_networks:
                    atu.copy_model_params_from_to(net, target_net)

    def snapshot(
        self,
        prefix: str,
        epoch: str
    ):
        # Save Env
        if hasattr(self.env, "_obs_normalizer") and \
           self.env._obs_normalizer is not None:
            normalizer_file_name = "_obs_normalizer_{}.pkl".format(epoch)
            normalizer_path = osp.join(prefix, normalizer_file_name)
            with open(normalizer_path, "wb") as f:
                pickle.dump(self.env._obs_normalizer, f)

        # Save Optimizier
        for name, optim in self.optimizers:
            optim_file_name = "optim_{}_{}.pth".format(name, epoch)
            optim_path = osp.join(prefix, optim_file_name)
            torch.save(optim.state_dict(), optim_path)

        # Save Agent
        self.agent.snapshot(prefix, epoch)

    def resume(
        self,
        prefix: str,
        epoch: str
    ) -> None:
        # Load Env
        if hasattr(self.env, "_obs_normalizer") and \
           self.env._obs_normalizer is not None:
            normalizer_file_name = "_obs_normalizer_{}.pkl".format(epoch)
            normalizer_path = osp.join(prefix, normalizer_file_name)
            with open(normalizer_path, "rb") as f:
                self.env._obs_normalizer = pickle.load(f)

        # Load Optim
        for name, optim in self.optimizers:
            optim_file_name = "optim_{}_{}.pth".format(name, epoch)
            optim_path = osp.join(prefix, optim_file_name)
            optim.load_state_dict(
                torch.load(
                    optim_path,
                    map_location=self.device
                )
            )
        self.agent.resume(prefix, epoch)

    @property
    def optimizers(self):
        return [
        ]
