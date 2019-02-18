from gym import Wrapper
from gym.spaces import Box
import gym
import numpy as np
from gym import Env
from collections import deque
import cv2

class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BaseWrapper, self).__init__(env)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class NormObs(gym.ObservationWrapper, BaseWrapper):
    """
    Normalized Observation => Optional, Use Momentum
    """
    def __init__( self, env, obs_alpha = 0.001 ):
        super(NormObs,self).__init__(env)
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(env.observation_space.shape[0])
        self._obs_var = np.ones(env.observation_space.shape[0])

    def _update_obs_estimate(self, obs):
        
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(obs - self._obs_mean)

    def _apply_normalize_obs(self, raw_obs):
        if self.training:
            self._update_obs_estimate(raw_obs)
        return (raw_obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def observation(self, observation):
        return self._apply_normalize_obs(observation)

class NormAct(gym.ActionWrapper, BaseWrapper):
    """
    Normalized Action      => [ -1, 1 ]
    """
    def __init__(self, env):
        super(NormAct, self).__init__(env)
        ub = np.ones(self.env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)
    
    def action(self, action):        
        lb = self.env.action_space.low
        ub = self.env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        return np.clip(scaled_action, lb, ub)

class RewardShift(gym.RewardWrapper, BaseWrapper):
    def __init__(self, env, reward_scale = 1):
        super(RewardShift, self).__init__(env)
        self._reward_scale = reward_scale
    
    def reward(self, reward):
        return self._reward_scale * reward

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

"""
Origin from OpenAI Baselines
"""
class WarpFrame(gym.ObservationWrapper, BaseWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    def __init__(self, env, width=84, height=84, grayscale=True):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = Box(low=0, high=255,
                shape=(1, self.height, self.width), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=255,
                shape=(3, self.height, self.width), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        frame = np.transpose(frame, (2,0,1) )
        return frame

class FrameStack( BaseWrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=( (shp[0] * k,) + shp[1:] ), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames),axis=0)

class ScaledFloatFrame(gym.ObservationWrapper, BaseWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

def wrap_deepmind(env, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    # if episode_life:
    #     env = EpisodicLifeEnv(env)
    # if 'FIRE' in env.unwrapped.get_action_meanings():
    #     env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    # if clip_rewards:
    #     env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

def wrap_continuous_env(env, obs_norm, obs_alpha, reward_scale ):
    env = RewardShift(env, reward_scale)
    if obs_norm:
        return NormObs(env, obs_alpha=obs_alpha) 
    return env

def get_env( env_id, env_param ):

    env = BaseWrapper( gym.make(env_id) )
    ob_space = env.observation_space
    if len(ob_space.shape) == 3:
        env = wrap_deepmind(env, **env_param)
    else:
        env = wrap_continuous_env(env, **env_param)
    
    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Box):
        return NormAct(env)
    return env
