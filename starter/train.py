import torchrl.networks as networks
import torchrl.policies as policies
from torchrl.utils import Logger
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from torchrl.utils.omegaconf_utils import omegadict_to_dict, print_dict
from torchrl.utils import set_seed
from torchrl.env import get_vec_env
import torch
import os.path as osp

from torchrl.replay_buffers import OnPolicyReplayBuffer
from torchrl.trainer import PPOTrainer
from torchrl.collector import OnPolicyCollector
from torchrl.agent import ActorCriticVAgent

# OmegaConf & Hydra Config
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver(
    "contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg)


@hydra.main(config_name="config", config_path="../config/")
def launch_rlg_hydra(cfg: DictConfig):

  # ensure checkpoints can be specified as relative paths
  if cfg.checkpoint:
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)

  cfg_dict = omegadict_to_dict(cfg)
  print_dict(cfg_dict)

  sim_device = torch.device(cfg.sim_device)
  rl_device = torch.device(cfg.rl_device)

  env = get_vec_env(
      cfg_dict["env"]["env_name"],
      cfg_dict["env"]["wrap"],
      cfg_dict["vec_env_nums"],
      device=sim_device
  )
  eval_env = get_vec_env(
      cfg_dict["env"]["env_name"],
      cfg_dict["env"]["wrap"],
      cfg_dict["vec_env_nums"],
      device=sim_device
  )

  # sets seed. if seed is -1 will pick a random one
  set_seed(
      env, cfg.seed, sim_device, rl_device,
      torch_deterministic=cfg.torch_deterministic
  )

  buffer_param = cfg_dict["train"]["replay_buffer"]

  experiment_name = cfg.exp_id
  logger = Logger(
      experiment_name, cfg.env.env_name, cfg.seed,
      cfg_dict, cfg.log_dir, cfg.overwrite
  )
  cfg_dict["train"]["general_setting"]["env"] = env

  replay_buffer = OnPolicyReplayBuffer(
      max_replay_buffer_size=int(buffer_param["size"]),
      num_envs=cfg_dict["vec_env_nums"],
      device=sim_device
  )
  cfg_dict["train"]["general_setting"]["replay_buffer"] = replay_buffer

  cfg_dict["train"]["general_setting"]["logger"] = logger
  cfg_dict["train"]["general_setting"]["device"] = cfg.rl_device

  cfg_dict["train"]["net"]["base_type"] = networks.MLPBase
  cfg_dict["train"]["net"]["activation_func"] = torch.nn.ReLU
  pf = policies.GuassianContPolicyBasicBias(
      input_dim=env.observation_space.shape[0],
      action_dim=env.action_space.shape[0],

      **cfg_dict["train"]["net"],
      **cfg_dict["train"]["policy"]
  )
  vf = networks.Net(
      input_dim=env.observation_space.shape[0],
      output_dim=1,
      **cfg_dict["train"]["net"]
  )
  agent = ActorCriticVAgent(
      pf=pf,
      vf=vf,
      with_target_pf=True
  )

  print(pf)
  print(vf)
  cfg_dict["train"]["general_setting"]["collector"] = OnPolicyCollector(
      env=env, eval_env=eval_env, agent=agent,
      replay_buffer=replay_buffer,
      rl_device=cfg.rl_device, sim_device=cfg.sim_device,
      train_render=False,
      **cfg_dict["train"]["collector"]
  )
  cfg_dict["train"]["general_setting"]["save_dir"] = osp.join(
      logger.work_dir, "model"
  )
  agent = PPOTrainer(
      agent=agent,
      **cfg_dict["train"]["ppo"],
      **cfg_dict["train"]["general_setting"]
  )
  agent.train()


if __name__ == "__main__":
  launch_rlg_hydra()
