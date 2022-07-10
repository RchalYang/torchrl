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
    self.pf_use_lstm = self.agent.pf_use_lstm
    if self.pf_use_lstm:
      self.hidden_states = torch.zeros(
          self.agent.pf.lstm_layers,
          self.env.env_nums,
          self.agent.pf.hidden_state_size,
          dtype=torch.float, device=self.rl_device
      )
    else:
      self.hidden_states = None

  def take_actions(self):
    ob_tensor = self.current_ob.clone().to(self.rl_device)
    out = self.agent.explore(
        ob_tensor,
        h=self.hidden_states
    )
    acts = out["action"].to(self.sim_device)
    values, _ = self.agent.predict_v(
        ob_tensor,
        h=self.hidden_states
    )
    next_hidden_states = out["hidden_state"]

    if not self.continuous:
      acts = acts[..., 0]
    else:
      if torch.isnan(acts).any():
        print("NaN detected. BOOM")
        print(ob_tensor)
        print(self.agent.explore(ob_tensor))
        sys.exit()

    next_obs, rewards, dones, infos = self.env.step(acts)
    if self.train_render:
      self.env.render()
    self.current_step += 1
    self.train_rew += rewards

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
    if self.pf_use_lstm:
      # hidden state shape (# num layers, # batch size, # hidden dims)
      sample_dict["hidden_states"] = self.hidden_states.transpose(0, 1)

    if torch.any(dones) or \
       torch.any(self.current_step >= self.max_episode_frames):
      surpass_flag = self.current_step >= self.max_episode_frames
      flag = (self.current_step >= self.max_episode_frames) | dones

      last_ob = next_obs.to(self.rl_device)
      last_value, _ = self.agent.predict_v(
          last_ob, h=next_hidden_states
      )
      last_value = last_value.detach().to(self.sim_device)
      sample_dict["terminals"] = flag
      sample_dict["rewards"] = rewards + \
          self.discount * last_value * surpass_flag

      next_obs = self.env.partial_reset(
          flag.squeeze(-1)
      )
      self.current_step[flag] = 0

      self.train_rews += list(self.train_rew[dones])
      self.train_rew[dones] = 0

      if self.pf_use_lstm:
        next_hidden_states[:, flag.squeeze(), :] = 0

    self.replay_buffer.add_sample(sample_dict)

    self.current_ob = next_obs
    self.hidden_states = next_hidden_states

    return torch.sum(rewards)
