# Pytorch-Off-Policy-Agent

Pytorch Implementation for Off Policy RL Methods

Environments with continuous & discrete action space are supported.

Environments with 1d & 3d observation space are supported.

## Requirements
1. General Requirements
* Pytorch 0.4.1
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
    python main.py --config config/sac_halfcheetah.json --seed 0 --device 0
```

## Currently contains:
* Soft Actor Critic: SAC(TwinSAC)
* Deep Deterministic Policy Gradient :DDPG
* TD3
* DQN:
    * Basic Double DQN

![HalfCheetah-v2 SAC DDPG](./fig/HalfCheetah-v2.png "HalfCheetah-v2")

## TODO:
1. Add More Algorithm
* DQN: C51 / QRDQN / IQN
