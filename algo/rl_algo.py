import numpy as np
import torch

class RLAlgo():

    def __init__(self,
        env,
        logger,

        discount=0.99,
        target_hard_update_period = 1000,
        use_soft_update = True,
        tau = 0.001,
        opt_times = 1,

        device = 'cpu',
    ):
        self.env = env

        self.device = device
        
        self.discount = discount

        self.target_hard_update_period = target_hard_update_period
        self.use_soft_update = use_soft_update
        self.tau = tau

        self.opt_times = opt_times

        # self.tf_writer = tf_writer

        self.logger = logger
        
        self.training_update_num = 0

        self.current_step = 0

        # self.max_episode_frames = max_episode_frames

        self.min_pool = min_pool

        self.replay_buffer
        self.batch_size = batch_size

        self.episode_rewards = deque(maxlen=10)

    def pretrain(self):
        pass
    
    def flush_tf_board(self, info):
        pass
    
    def flush_logger(seelf, info):
        pass

    def eval():
        eval_env = copy.deepcopy(self.env)
        eval_env.eval()
        done = False
        for _ in range(self.eval_episodes):

            eval_ob = eval_env.reset()
            rew = 0
            while not done:
                act = self.pf.eval( torch.Tensor( eval_ob ).to(self.device).unsqueeze(0) )
                eval_ob, r, done, _ = eval_env.step( act.detach().cpu().numpy() )
                rew += r
            self.episode_rewards.append(rew)
            done = False

    def train(self):
        self.pretrain()

        self.current_step = 0
        start = time.time()
        for j in range( self.num_epochs ):
            for step in range(self.epoch_frames):
                # Sample actions
                with torch.no_grad():
                    _, _, action, _ = pf.explore( torch.Tensor( ob ).to(device).unsqueeze(0) )

                action = action.detach().cpu().numpy()
                action = action[0]

                if np.isnan(action).any():
                    print("NaN detected. BOOM")
                    exit()
                # Obser reward and next obs
                next_ob, reward, done, _ = self.env.step(action)

                self.replay_buffer.add_sample(ob, action, reward, done, next_ob )

                if replay_buffer.num_steps_can_sample() > self.min_pool:
                    for _ in range( self.opt_times ):
                        batch = self.replay_buffer.random_batch(self.batch_size)
                        infos = self.update( batch )
                        for info in infos:
                            self.writer.add_scalar("Training/{}".format(info), infos[info] , j * args.epoch_frames + step )
                
                ob = next_ob
                self.current_step += 1
                if done or self.current_step >= self.max_episode_frames:
                    ob = self.env.reset()
                    self.current_step = 0
                
            total_num_steps = (j + 1) * args.epoch_frames

            #eval_env.ob_rms = copy.deepcopy(training_env.ob_rms)
            self.eval()
            
            writer.add_scalar("Eval/Reward", np.mean(episode_rewards) , total_num_steps)

            print("Epoch {}, Evaluation using {} episodes: mean reward {:.5f}\n".
                format(j, len(episode_rewards),
                        np.mean(episode_rewards)))

