import torch
import torch.optim as optim

import gym
from tensorboardX import SummaryWriter
import copy
import glob
import os
import time
import shutil
import os.path as osp
import stat
from collections import deque
import numpy as np

from off_ppo import OffPPO
from env import VecNormalize
from env import NormalizedActions
from argument import get_args
from model import Policy
from model import QNet
from model import VNet

from replay_buffer import SimpleReplayBuffer


args = get_args()

def main():

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    training_env = VecNormalize(NormalizedActions( (gym.make(args.env_name)) ))
    eval_env = copy.deepcopy(training_env)
    training_env.train()
    eval_env.eval()

    pf = Policy( training_env.observation_space.shape[0], training_env.action_space.shape[0], 256 )

    vf = VNet( training_env.observation_space.shape[0], 256 )
    qf = QNet( training_env.observation_space.shape[0], training_env.action_space.shape[0], 256 )

    pf.to(device)
    vf.to(device)
    qf.to(device)

    agent = OffPPO(
                pf,
                vf,
                qf,

                plr = args.plr,
                vlr = args.vlr,
                qlr = args.qlr,
                clip_param = args.clip_param, 
                ent_coeff = args.entropy_coef,

                # max_grad_norm=args.max_grad_norm,
                target_hard_update_period=args.hard_update_interval,
                tau=args.tau,
                use_soft_update=args.soft_update,
                optimizer_class=optim.Adam,

                discount = args.discount,
                device = device,
                max_grad_norm = args.max_grad_norm,
                norm = args.norm,
                entropy_coef = args.entropy_coef
            )

    # rollouts = RolloutStorage(args.num_steps, args.num_processes,
    #                     envs.observation_space.shape, envs.action_space,
    #                     actor_critic.recurrent_hidden_state_size)
    replay_buffer = SimpleReplayBuffer( args.buffer_size, training_env.observation_space.shape[0], training_env.action_space.shape[0] )

    ob = training_env.reset()
    # rollouts.obs[0].copy_(obs)
    # rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    work_dir = osp.join("log", args.env_name, str(args.seed),  args.id )
    if osp.exists( work_dir ):
        os.chmod(work_dir, stat.S_IWUSR)
        shutil.rmtree(work_dir)
    writer = SummaryWriter( work_dir )

    # start = time.time()
    for j in range( args.num_epochs ):
        for step in range(args.epoch_frames):
            # Sample actions
            with torch.no_grad():
                _, _, action, _ = pf.explore( torch.Tensor( ob ).to(device).unsqueeze(0) )

            action = action.detach().cpu().numpy()

            if np.isnan(action).any():
                print("NaN detected. BOOM")
                exit()
            # Obser reward and next obs
            next_ob, reward, done, _ = training_env.step(action)

            replay_buffer.add_sample(ob, action, reward, done, next_ob )

            if replay_buffer.num_steps_can_sample() > 10 * args.batch_size:
                for _ in range( args.opt_times ):
                    batch = replay_buffer.random_batch(args.batch_size)
                    infos = agent.update( batch )
                    for info in infos:
                        writer.add_scalar("Training/{}".format(info), infos[info] , j * args.epoch_frames + step )
            
            ob = next_ob 
            if done:
                ob = training_env.reset()
            

        total_num_steps = (j + 1) * args.epoch_frames

        eval_env.ob_rms = copy.deepcopy(training_env.ob_rms)
        for _ in range(args.eval_episodes):

            eval_ob = eval_env.reset()
            rew = 0
            done = False
            while not done:
                act = pf.eval( torch.Tensor( eval_ob ).to(device).unsqueeze(0) )
                eval_ob, r, done, _ = eval_env.step( act.detach().cpu().numpy() )
                rew += r
            episode_rewards.append(rew)
            print(rew)
         
        writer.add_scalar("Eval/Reward", np.mean(episode_rewards) , total_num_steps)

        print("Epoch {}, Evaluation using {} episodes: mean reward {:.5f}\n".
            format(j, len(episode_rewards),
                    np.mean(episode_rewards)))


if __name__ == "__main__":
    main()
