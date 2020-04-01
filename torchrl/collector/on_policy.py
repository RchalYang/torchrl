import torch
import numpy as np
import copy
from .base import BaseCollector
from torchrl.env import VecEnv

class OnPlicyCollectorBase(BaseCollector):
    def __init__(self, vf, discount=0.99, **kwargs):
        self.vf = vf
        super().__init__(**kwargs)
        self.discount = discount
        self.env_info.vf = vf
        self.env_info.discount = discount
        self.env_info.is_vec = isinstance(self.env, VecEnv)

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf = funcs["pf"]
        vf = funcs["vf"]

        ob = ob_info["ob"]

        ob_tensor = torch.Tensor(ob).to(env_info.device).unsqueeze(0)
        # if env_info.is_vec:
        # ob_tensor = ob_tensor.unsqueeze(0)

        out = pf.explore(ob_tensor)
        act = out["action"]
        act = act.detach().cpu().numpy()

        value = vf(ob_tensor)
        value = value.cpu().item()

        if not env_info.continuous:
            act = act[0]
        # print(act)
        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, reward, done, info = env_info.env.step(act)
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            "values": [value],
            "rewards": [reward],
            "terminals": [done],
            "time_limits": [True if "time_limit" in info else False]
        }

        if done or env_info.current_step >= env_info.max_episode_frames:
            if not done and env_info.current_step >= env_info.max_episode_frames:
                last_ob = torch.Tensor(next_ob).to(env_info.device).unsqueeze(0)
                last_value = env_info.vf(last_ob).item()

                sample_dict["terminals"] = [True]
                sample_dict["rewards"] = [reward + env_info.discount * last_value]

            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode()

        replay_buffer.add_sample(sample_dict, env_info.env_rank)

        return next_ob, done, reward, info

    @property
    def funcs(self):
        return {
            "pf": self.pf,
            "vf": self.vf
        }


class VecOnPlicyCollector(OnPlicyCollectorBase):

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf = funcs["pf"]
        vf = funcs["vf"]

        obs = ob_info["ob"]

        ob_tensor = torch.Tensor(obs).to(env_info.device)
        # if env_info.is_vec:
        # ob_tensor = ob_tensor.unsqueeze(0)

        out = pf.explore(ob_tensor)
        acts = out["action"]
        acts = acts.detach().cpu().numpy()

        values = vf(ob_tensor)
        values = values.detach().cpu().numpy()

        # if not env_info.continuous:
        #     act = act[0]
        # print(act)
        if type(acts) is not int:
            if np.isnan(acts).any():
                print("NaN detected. BOOM")
                exit()

        next_obs, rewards, dones, infos = env_info.env.step(acts)
        # dones = 
        # rewards = rewards[..., np.newaxis]

        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1
        # print(infos)
        # print(infos[0]["reward_fwd"])
        # if env_info.is_vec:
        sample_dict = {
            "obs": obs,
            "next_obs": next_obs,
            "acts": acts,
            "values": values,
            "rewards": rewards[..., np.newaxis],
            "terminals": dones[..., np.newaxis],
            # "time_limits": [True if "time_limit" in info else False]
            "time_limits": np.zeros_like(values)
        }
        # print(sample_dict)
        # if done or env_info.current_step >= env_info.max_episode_frames:
        #     if not done and env_info.current_step >= env_info.max_episode_frames:
        #         last_ob = torch.Tensor(next_ob).to(env_info.device).unsqueeze(0)
        #         last_value = env_info.vf(last_ob).item()

        #         sample_dict["terminals"] = [True]
        #         sample_dict["rewards"] = [reward + env_info.discount * last_value]

        #     next_ob = env_info.env.reset()
        #     env_info.finish_episode()
        #     env_info.start_episode()
        if np.any(dones):
            env_info.env.partial_reset(dones)

        replay_buffer.add_sample(sample_dict, env_info.env_rank)

        return next_obs, dones, rewards, infos

    def train_one_epoch(self):
        train_rews = []
        train_epoch_reward = 0
        self.env.train()
        for i in range(self.epoch_frames):
            # print(i)
            # Sample actions
            next_ob, done, reward, _ = self.__class__.take_actions(self.funcs,
                self.env_info, self.c_ob, self.replay_buffer)
            self.c_ob["ob"] = next_ob
            # print(self.c_ob)
            self.train_rew += reward
            train_epoch_reward += reward
        #     if done:
        #         self.training_episode_rewards.append(self.train_rew)
        #         train_rews.append(self.train_rew)
        #         self.train_rew = 0
        # return {"train_rews": []}
        return {
            'train_rewards': [],
            'train_epoch_reward': 0
        }

    def eval_one_epoch(self):

        eval_infos = {}
        eval_rews = []

        done = False

        self.eval_env = copy.copy(self.env)
        self.eval_env.eval()
        # print(self.eval_env._obs_mean)
        traj_lens = []
        for _ in range(self.eval_episodes):
            for idx in range(self.eval_env.env_nums):
                eval_ob = self.eval_env.envs[idx].reset()
                rew = 0
                traj_len = 0
                while not np.all(done):
                    act = self.pf.eval_act(torch.Tensor(eval_ob).to(
                        self.device).unsqueeze(0))
                    eval_ob, r, done, _ = self.eval_env.envs[idx].step(act)
                    rew += r
                    traj_len += 1
                    if self.eval_render:
                        self.eval_env.envs[idx].render()

                eval_rews.append(rew)
                traj_lens.append(traj_len)

                done = False

        eval_infos["eval_rewards"] = eval_rews
        eval_infos["eval_traj_length"] = np.mean(traj_lens)
        return eval_infos
