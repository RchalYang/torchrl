
import torch
import torch.multiprocessing as mp
import copy
import numpy as np
import gym
from collections import deque

from .base import BaseCollector
from .base import EnvInfo

from torchrl.replay_buffers.shared import SharedBaseReplayBuffer

def train_worker_process(cls, shared_pf, env_info, replay_buffer, shared_que ):
    pf = copy.deepcopy(shared_pf)
    c_ob = env_info.env.reset()
    train_rew = 0
    while True:

        pf.load_state_dict(shared_pf.state_dict())

        train_rews = []
        train_epoch_reward = 0    

        for _ in range(env_info.epoch_frames):
            # Sample actions
            next_ob, done, reward, _ = cls.take_actions(pf, env_info, c_ob, cls.get_actions, replay_buffer )
            c_ob = next_ob
            # print(self.c_ob)
            train_rew += reward
            train_epoch_reward += reward
            if done:
                # self.training_episode_rewards.append(self.train_rew)
                train_rews.append(train_rew)
                train_rew = 0

        shared_que.put({
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        })
    
class ParallelCollector(BaseCollector):

    def __init__(self, 
            env, pf, replay_buffer,
            worker_nums = 4,
            **kwargs):
        
        super().__init__(
            env, pf, replay_buffer,
            **kwargs)

        # self.pool = mp.Pool(worker_nums)
        # self.manager = mp.Manager()

        self.pf.share_memory()
        self.pf.to(self.device)

        assert isinstance(replay_buffer, SharedBaseReplayBuffer), \
            "Should Use Shared Replay buffer"
        self.replay_buffer = replay_buffer
        
        # self.envs = [ copy.copy(self.env) for _ in range(worker_nums)] 
        # self.eval_envs = [ copy.copy(self.env) for _ in range(worker_nums)] 

        # # self.eval_env = copy.copy(env)
        # self.eval_env._reward_scale = 1

        self.manager = mp.Manager()
        self.shared_que = self.manager.Queue()

        self.worker_nums = worker_nums
        self.workers = []
        env_info = self.env_info
        for i in range(worker_nums):
            env_info.env_rank = i
            self.workers.append(
                mp.Process(target=train_worker_process, args=(self.__class__, self.pf,
                    env_info, self.replay_buffer, self.shared_que 
                )).start()
            )
        
    def start_episode(self):
        self.current_step = 0

    def finish_episode(self):
        pass
    
    # def get_actions(self, input_dic):
    #     ob = input_dic["ob"]

    def train_one_epoch(self):
        train_rews = []
        train_epoch_reward = 0

        for _ in range(self.worker_nums):
            worker_rst = self.shared_que.get()
            train_rews += worker_rst["train_rewards"]
            train_epoch_reward += worker_rst["train_epoch_reward"]
        

        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        }
        
    def eval_one_epoch(self):

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
                if self.eval_render:
                    eval_env.render()

            eval_rews.append(rew)
            # self.episode_rewards.append(rew)

            done = False
        
        eval_env.close()
        del eval_env

        # eval_infos["Eval_Rewards_Average"] = np.mean( eval_rews )
        eval_infos["eval_rewards"] = eval_rews
        return eval_infos
