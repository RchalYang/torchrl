# TorchRL

Pytorch Implementation for RL Methods

Environments with continuous & discrete action space are supported.

Environments with 1d & 3d observation space are supported.

Multi-Process Env is supported

Fully GPU training is supported (Like Isaac Gym)

## Requirements
1. General Requirements
* Pytorch 1.7
* Gym(0.10.9)
* Mujoco(1.50.1)
* tabulate (for log)
* tensorboardX (log file output)
2. Tensorboard Requirements
* Tensorflow: to start tensorboard or read log in tf records

## Installation
1. use 
use **environment.yml** to create virtual envrionment
```
    conda create -f environment.yml
    source activate py_off
```

2. Mannually install all requirements


## Usage
specify parameters for algorithms in config file & specify log directory / seed / device in argument

```
    python examples/ppo_continuous_vec.py --config config/ppo_halfcheetah.json --seed 0 --device 0 --id ppo_halfcheetah
```

Checkout examples folder for detailed informations

## Currently contains:
* On-Policy Methods:
    * Reinforce
    * A2C(Actor Critic)
    * PPO(Proximal Policy Optimization)
    * TRPO
    * V-MPO
    * DAAC
    * IDAAC
    * PPG
* Off-Policy Methods:
    * Soft Actor Critic: SAC(TwinSAC)
    * Deep Deterministic Policy Gradient :DDPG
    * TD3
    * DQN:
        * Basic Double DQN
        * Bootstrapped DQN
        * QRDQN
* Imitation Learning:
    * Behaviour Cloning
    * GAIL

## Update:

Major Update:
* Update code structure:
    * separate agent & trainer
* Use Hydra for configuration