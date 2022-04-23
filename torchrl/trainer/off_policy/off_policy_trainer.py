import time
import numpy as np
import math
import torch
from torchrl.trainer.rl_trainer import RLTrainer


class OffPolicyTrainer(RLTrainer):
  """
  Base RL Algorithm Framework
  """

  def __init__(
      self,
      pretrain_epochs: int = 0,
      # min_pool: int = 0,
      target_hard_update_period: int = 1000,
      use_soft_update: bool = True,
      tau: float = 0.001,
      opt_times: int = 1,
      **kwargs
  ) -> None:
    super(OffPolicyTrainer, self).__init__(**kwargs)

    # environment relevant information
    self.pretrain_epochs = pretrain_epochs

    # target_network update information
    self.target_hard_update_period = target_hard_update_period
    self.use_soft_update = use_soft_update
    self.tau = tau

    # training information
    self.opt_times = opt_times
    # self.min_pool = min_pool

    self.sample_key = ["obs", "next_obs", "acts", "rewards", "terminals"]

  def update_per_epoch(self) -> None:
    for _ in range(self.opt_times):
      batch = self.replay_buffer.random_batch(
          self.batch_size,
          self.sample_key,
          device=self.device
      )
      infos = self.update(batch)
      self.logger.add_update_info(infos)

  def pretrain(self) -> None:
    total_frames = 0

    self.pretrain_frames = self.pretrain_epochs * self.epoch_frames
    for pretrain_epoch in range(self.pretrain_epochs):
      start = time.time()

      self.start_epoch()

      training_epoch_info = self.collector.train_one_epoch()
      for reward in training_epoch_info["train_rewards"]:
        self.training_episode_rewards.append(reward)

      finish_epoch_info = self.finish_epoch()

      total_frames += self.epoch_frames

      infos = {}

      infos["Train_Epoch_Reward"] = \
          training_epoch_info["train_epoch_reward"]
      infos["Running_Training_Average_Rewards"] = np.mean(
          self.training_episode_rewards)
      infos.update(finish_epoch_info)

      self.logger.add_epoch_info(
          pretrain_epoch, total_frames,
          time.time() - start, infos, csv_write=False)

    self.logger.log("Finished Pretrain")
