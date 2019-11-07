
import torch
import torch.multiprocessing as mp
import copy
import numpy as np
import gym
from collections import deque

from .base import BaseCollector
from .base import EnvInfo

from torchrl.replay_buffers.shared import SharedBaseReplayBuffer

def train_worker_process(cls, shared_pf, env_info,
    replay_buffer, shared_que,
    start_barrier, terminate_mark ):

    replay_buffer.rebuild_from_tag()
    pf = copy.deepcopy(shared_pf)
    c_ob = env_info.env.reset()
    train_rew = 0
    while True:
        start_barrier.wait()
        if terminate_mark.value == 1:
            break
        pf.load_state_dict(shared_pf.state_dict())

        train_rews = []
        train_epoch_reward = 0    

        for _ in range(env_info.epoch_frames):
            next_ob, done, reward, _ = cls.take_actions(pf, env_info, c_ob, cls.get_actions, replay_buffer )
            c_ob = next_ob
            train_rew += reward
            train_epoch_reward += reward
            if done:
                train_rews.append(train_rew)
                train_rew = 0

        shared_que.put({
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        })

def eval_worker_process(shared_pf, 
    env_info, shared_que, start_barrier, terminate_mark):

    pf = copy.deepcopy(shared_pf)
    env_info.env.eval()
    env_info.env._reward_scale = 1

    while True:
        start_barrier.wait()
        if terminate_mark.value == 1:
            break
        pf.load_state_dict(shared_pf.state_dict())

        eval_rews = []  

        done = False
        for _ in range(env_info.eval_episodes):

            eval_ob = env_info.env.reset()
            rew = 0
            while not done:
                act = pf.eval( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))
                eval_ob, r, done, _ = env_info.env.step( act )
                rew += r
                if env_info.eval_render:
                    env_info.env.render()

            eval_rews.append(rew)
            done = False

        shared_que.put({
            'eval_rewards':eval_rews
        })

class ParallelCollector(BaseCollector):

    def __init__(self, 
            env, pf, replay_buffer,
            worker_nums = 4,
            eval_worker_nums = 1,
            **kwargs):
        
        super().__init__(
            env, pf, replay_buffer,
            **kwargs)

        self.pf.share_memory()
        self.pf.to(self.device)

        assert isinstance(replay_buffer, SharedBaseReplayBuffer), \
            "Should Use Shared Replay buffer"
        self.replay_buffer = replay_buffer

        self.worker_nums = worker_nums
        self.eval_worker_nums = eval_worker_nums

        self.manager = mp.Manager()

        self.workers = []
        self.shared_que = self.manager.Queue()
        self.start_barrier = mp.Barrier(worker_nums+1)
        self.terminate_mark = mp.Value( 'c', 0 )
                
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue()
        self.eval_start_barrier = mp.Barrier(eval_worker_nums+1)

        env_info = self.env_info
        
        for i in range(worker_nums):
            env_info.env_rank = i
            p = mp.Process(
                target=train_worker_process,
                args=( self.__class__, self.pf,
                    env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.terminate_mark))
            p.start()
            self.workers.append(p)

        for i in range(eval_worker_nums):
            eval_p = mp.Process(
                target=eval_worker_process,
                args=(self.pf,
                    env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.terminate_mark))
            eval_p.start()
            self.eval_workers.append(eval_p)

    def start_episode(self):
        self.current_step = 0

    def finish_episode(self):
        pass
    
    def terminate(self):
        self.terminate_mark.value = 1
        for p in self.workers:
            p.join()
        
        for p in self.eval_workers:
            p.join()

    def train_one_epoch(self):
        self.start_barrier.wait()
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
        self.eval_start_barrier.wait()
        eval_rews = []

        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            eval_rews += worker_rst["eval_rewards"]
        
        return {
            'eval_rewards':eval_rews,
        }
