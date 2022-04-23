import numpy as np
import torch
from torchrl.trainer.rl_trainer import RLTrainer
from torchrl.utils.normalizer import TorchNormalizer


class OnPolicyTrainer(RLTrainer):
  """
  Base RL Algorithm Framework
  """

  def __init__(
      self,
      shuffle: bool = True,
      tau: float = None,
      gae: bool = True,
      **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.sample_key = ["obs", "acts", "advs", "estimate_returns"]
    self.shuffle = shuffle
    self.tau = tau
    self.gae = gae

    # self.advantage_normalizer = TorchNormalizer(
    #     (1,), self.device
    # )

  def process_epoch_samples(self) -> None:
    sample = self.replay_buffer.last_sample(
        ["next_obs", "terminals", "time_limits"]
    )

    last_ob = sample["next_obs"].to(self.device)
    # if not self.replay_buffer.on_gpu:
      # last_ob = torch.tensor(last_ob).to(self.device)
    last_value = self.agent.predict_v(last_ob).to(
        self.replay_buffer.device).detach()
    # if not self.replay_buffer.on_gpu:
      # last_value = last_value.detach().cpu().numpy()
    last_value = last_value * (1 - sample["terminals"])

    if self.gae:
      self.replay_buffer.generalized_advantage_estimation(
          last_value,
          self.discount,
          self.tau
      )
    else:
      self.replay_buffer.discount_reward(
          last_value, self.discount
      )

  def pre_update_process(self) -> None:
    self.process_epoch_samples()
    # self.advantage_normalizer.update(
    #     self.replay_buffer._advs
    # )
    # self.replay_buffer._advs = self.advantage_normalizer.filt(
    #     self.replay_buffer._advs
    # )
    self.replay_buffer._advs = (
        self.replay_buffer._advs - self.replay_buffer._advs.mean()
    ) / (self.replay_buffer._advs.std() + 1e-5)

  def update_per_epoch(self) -> None:
    for batch in self.replay_buffer.one_iteration(
        self.batch_size,
        self.sample_key,
        self.shuffle,
        device=self.device
    ):
      infos = self.update(batch)
      self.logger.add_update_info(infos)
