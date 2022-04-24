"""Base Collector: Implement Basic Operations."""
import gym
import torch
import copy
import sys
import numpy as np
from collections import deque
from torchrl.env.base_wrapper import BaseWrapper
from torchrl.env.vecenv import VecEnv
from torchrl.agent import RLAgent
from torchrl.replay_buffers import BaseReplayBuffer


class BaseCollector:
  """Base Collector: Implement Basic Operations."""

  def __init__(
      self,
      env: BaseWrapper,
      eval_env: BaseWrapper,
      agent: RLAgent,
      replay_buffer: BaseReplayBuffer,
      epoch_frames: int,
      train_render=False,
      eval_interval: int = 1,
      eval_episodes: int = 1,
      eval_render: bool = False,
      rl_device: str = "cpu",
      sim_device: str = "cpu",
      max_episode_frames: int = 999,
  ):
    """
    Base Collector.

    Env: Environment Wrapper.

    """
    self.agent = agent
    self.replay_buffer = replay_buffer

    self.env = env
    self.env.train()
    self.continuous = isinstance(self.env.action_space, gym.spaces.Box)

    if eval_env is not None:
      self.eval_env = eval_env
    else:
      self.eval_env = copy.deepcopy(env)

    if hasattr(env, "_obs_normalizer"):
      self.eval_env._obs_normalizer = env._obs_normalizer
    self.eval_env._reward_scale = 1

    # device specification
    self.rl_device = rl_device
    self.sim_device = sim_device

    # Training
    self.epoch_frames = epoch_frames
    self.sample_epoch_frames = epoch_frames
    self.sample_epoch_frames //= self.env.env_nums
    self.max_episode_frames = max_episode_frames
    self.train_render = train_render

    # Evaluation
    self.eval_interval = eval_interval
    self.enable_eval = (eval_interval > 0)
    self.eval_episodes = eval_episodes
    self.eval_render = eval_render

    # Initialization
    self.current_ob = self.env.reset()
    self.current_step = torch.zeros((self.env.env_nums, 1), device=sim_device)
    self.train_rew = torch.zeros_like(self.current_step, device=sim_device)

    self.agent.to(self.rl_device)

  def start_episode(self, flag):
    pass

  def finish_episode(self, flag):
    pass

  def take_actions(self):
    out = self.agent.explore(
        self.current_ob.to(self.rl_device)
    )
    act = out["action"]

    if not self.continuous:
      act = act[..., 0]
    elif torch.isnan(act).any():
      print("NaN detected. BOOM")
      sys.exit()

    next_ob, reward, done, info = self.env.step(act)
    if self.train_render:
      self.env.render()
    self.current_step += 1
    self.train_rew += reward

    sample_dict = {
        "obs": self.current_ob,
        "next_obs": next_ob,
        "acts": act,
        "rewards": reward,
        "terminals": done,
        "time_limits": info["time_limit"] if "time_limit" in info
          else torch.zeros_like(reward, device=self.sim_device),
    }

    if torch.any(done) or \
            torch.any(self.current_step >= self.max_episode_frames):
      flag = (self.current_step >= self.max_episode_frames) | done

      next_ob = self.env.partial_reset(flag.squeeze(-1))

      self.current_step[flag] = 0
      self.train_rews += list(self.train_rew[flag])
      self.train_rew[flag] = 0

      self.finish_episode(flag)
      self.start_episode(flag)

    self.replay_buffer.add_sample(sample_dict)

    self.current_ob = next_ob

    return reward

  def terminate(self):
    self.env.close()
    if self.eval_env is not self.env:
      self.eval_env.close()

  def train_one_epoch(self):
    self.train_rews = []
    self.train_epoch_reward = 0
    self.env.train()

    for _ in range(self.sample_epoch_frames):
      # Sample actions
      reward = self.take_actions()
      self.train_epoch_reward += reward

    return {
        "train_rewards": self.train_rews,
        "train_epoch_reward": self.train_epoch_reward,
    }

  def eval_one_epoch(self):
    eval_infos = {}
    eval_rews = []

    done = False
    # if hasattr(self.env, "_obs_normalizer"):
    #   self.eval_env._obs_normalizer = copy.deepcopy(self.env._obs_normalizer)
    print(self.eval_env._obs_normalizer._mean)
    print(self.eval_env._obs_normalizer._var)
    print(self.eval_env._obs_normalizer._count)
    self.eval_env.eval()

    traj_lens = []
    for _ in range(self.eval_episodes):
      done = torch.zeros(
          (self.eval_env.env_nums, 1),
          device=self.sim_device, dtype=torch.bool
      )
      epi_done = torch.zeros(
          (self.eval_env.env_nums, 1),
          device=self.sim_device, dtype=torch.bool
      )

      eval_obs = self.eval_env.reset()
      rews = torch.zeros_like(done, device=self.sim_device)
      traj_len = torch.zeros_like(rews, device=self.sim_device)

      while not torch.all(epi_done):
        act = self.agent.eval_act(
            eval_obs.to(self.rl_device)
        ).to(self.sim_device)
        if self.continuous and torch.isnan(act).any():
          print("NaN detected. BOOM")
          print(self.agent.pf.forward(eval_obs.to(self.rl_device)))
          sys.exit()
        try:
          eval_obs, r, done, _ = self.eval_env.step(act)
          rews = rews + (~epi_done).float() * r
          traj_len = traj_len + (~epi_done).float()

          epi_done = epi_done | done
          if torch.any(done):
            eval_obs = self.eval_env.partial_reset(
                done.squeeze(-1)
            )

          if self.eval_render:
            self.eval_env.render()
        except Exception as e:
          print(e)
          print(act)
          sys.exit()
      eval_rews += list(rews.cpu().numpy())
      traj_lens += list(traj_len.cpu().numpy())
    eval_infos["eval_rewards"] = eval_rews
    eval_infos["eval_traj_length"] = np.mean(traj_lens)
    return eval_infos
