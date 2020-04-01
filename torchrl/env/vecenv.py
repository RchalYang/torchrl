import numpy as np
from .base_wrapper import BaseWrapper

class VecEnv(BaseWrapper):
    def __init__(self, env_nums, env_funcs, env_args):
        self.env_nums = env_nums
        self.env_funcs = env_funcs
        self.env_args = env_args
        if isinstance(env_funcs, list):
            assert len(env_funcs) == env_nums
            assert len(env_args) == env_args
        else:
            self.env_funcs = [env_funcs for _ in range(env_nums)]
            self.env_args = [env_args for _ in range(env_nums)]

        self.envs = [env_func(*env_arg) for env_func, env_arg
                     in zip(self.env_funcs, self.env_args)]
        # for _ in env_funcs:

    def train(self):
        for env in self.envs:
            env.train()

    def eval(self):
        for env in self.envs:
            env.eval()

    def reset(self, **kwargs):
        obs = [env.reset() for env in self.envs]
        # for env in self.envs
        self._obs = np.stack(obs)
        return self._obs

    def partial_reset(self, index_mask, **kwargs):
        # print(index_mask)
        indexs = np.argwhere(index_mask == 1).reshape((-1))
        # print(indexs)
        # for index in indexs:
        #     print(index)
        reset_obs = [self.envs[index].reset() for index in indexs]
        self._obs[index_mask] = reset_obs
        # for index in indexs:
        #     ob = self.envs[index].reset()
        #     reset_obs.append(ob)
        return self._obs

    def step(self, actions):
        actions = np.split(actions, self.env_nums)
        result = [env.step(action) for env, action in
                  zip(self.envs, actions)]
        obs, rews, dones, infos = zip(*result)
        self._obs = np.stack(obs)
        # self._
        return self._obs, np.stack(rews), \
               np.stack(dones), np.stack(infos)
        # return self._obs, self._rew, self._done, self._info

    def seed(self, seed):
        for env in self.envs:
            env.seed(seed)

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space