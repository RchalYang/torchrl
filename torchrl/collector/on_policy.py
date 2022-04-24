"""Collectors For OnPolicy RL methods."""
import torch
import sys
import numpy as np
from .base import BaseCollector


class OnPolicyCollector(BaseCollector):
  """On Policy Collector."""

  def __init__(
      self,
      discount: float = 0.99,
      **kwargs
  ):
    super().__init__(**kwargs)
    self.discount = discount

  def take_actions(self):
    ob_tensor = self.current_ob.clone().to(self.rl_device)
    out = self.agent.explore(ob_tensor)
    acts = out["action"].to(self.sim_device)
    values = self.agent.predict_v(ob_tensor)

    if self.continuous:
      if torch.isnan(acts).any():
        print("NaN detected. BOOM")
        print(ob_tensor)
        print(self.agent.explore(ob_tensor))
        sys.exit()

    next_obs, rewards, dones, infos = self.env.step(acts)
    if self.train_render:
      self.env.render()
    self.current_step += 1

    sample_dict = {
        "obs": self.current_ob,
        "next_obs": next_obs,
        "acts": acts,
        "values": values,
        "rewards": rewards,
        "terminals": dones,
        "time_limits":
            infos["time_limit"] if "time_limit" in infos else
            torch.zeros_like(dones, device=self.sim_device)
    }
    self.train_rew += rewards

    if torch.any(dones) or \
       torch.any(self.current_step >= self.max_episode_frames):
      surpass_flag = self.current_step >= self.max_episode_frames
      flag = (self.current_step >= self.max_episode_frames) | dones

      last_ob = next_obs.to(self.rl_device)
      last_value = self.agent.predict_v(last_ob).detach().to(self.sim_device)
      sample_dict["terminals"] = flag
      sample_dict["rewards"] = rewards + \
          self.discount * last_value * surpass_flag

      next_obs = self.env.partial_reset(
          flag.squeeze(-1)
      )
      self.current_step[flag] = 0

      self.train_rews += list(self.train_rew[dones])
      self.train_rew[dones] = 0

    self.replay_buffer.add_sample(sample_dict)

    self.current_ob = next_obs

    return torch.sum(rewards)
