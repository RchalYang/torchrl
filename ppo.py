import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import log_probs,flow_loss

class PPO():
    def __init__(self,
                 pf,
                 vf,

                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 entropy_coef,
                 plr=3e-4,
                 vlr=3e-4,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False):

        self.pf = pf
        self.vf = vf
        
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.pf_optim = optim.Adam(self.pf.parameters(), lr=plr, eps=eps)
        self.vf_optim = optim.Adam(self.vf.parameters(), lr=vlr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch,  actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, next_obs_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -( torch.min(surr1, surr2) ).mean()

                value_loss = 0.5 * F.mse_loss(return_batch, values)

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
