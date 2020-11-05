from .atari_wrapper import *
from .continuous_wrapper import *
from .base_wrapper import *

def wrap_deepmind(env, frame_stack=False, scale=False, clip_rewards=False):
    """
    Wrap the environment into the environment.

    Args:
        env: (todo): write your description
        frame_stack: (todo): write your description
        scale: (float): write your description
        clip_rewards: (bool): write your description
    """
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def wrap_continuous_env(env, obs_norm, reward_scale):
    """
    Wrap environment variables in the environment variable.

    Args:
        env: (todo): write your description
        obs_norm: (todo): write your description
        reward_scale: (str): write your description
    """
    env = RewardShift(env, reward_scale)
    if obs_norm:
        return NormObs(env)
    return env


def get_env(env_id, env_param):
    """
    Return an environment object.

    Args:
        env_id: (str): write your description
        env_param: (todo): write your description
    """
    env = gym.make(env_id)
    if str(env.__class__.__name__).find('TimeLimit') >= 0:
        env = TimeLimitAugment(env)
    env = BaseWrapper(env)
    if "rew_norm" in env_param:
        env = NormRet(env, **env_param["rew_norm"])
        del env_param["rew_norm"]

    ob_space = env.observation_space
    if len(ob_space.shape) == 3:
        env = wrap_deepmind(env, **env_param)
    else:
        env = wrap_continuous_env(env, **env_param)


    # act_space = env.action_space
    # if isinstance(act_space, gym.spaces.Box):
    #     return NormAct(env)
    return env
