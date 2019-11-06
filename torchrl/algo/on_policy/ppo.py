import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn as nn

from .a2c import A2C
import torchrl.algo.utils as atu

class PPO(A2C):
    """
    Actor Critic
    """
    def __init__(
        self,
        pf,
        clip_para = 0.2,
        opt_epochs = 10,
        **kwargs
    ):

        self.target_pf = copy.deepcopy(pf)
        super(PPO, self).__init__(pf = pf, **kwargs)

        self.clip_para = clip_para
        self.opt_epochs = opt_epochs
    
    def update_per_epoch(self):

        sample = self.replay_buffer.last_sample( ['next_obs', 'terminals' ] )
        last_value = 0
        if not sample['terminals']:
            last_ob = torch.Tensor( sample['next_obs'] ).to(self.device).unsqueeze(0) 
            last_value = self.vf( last_ob ).item()
        
        if self.gae:
            self.replay_buffer.generalized_advantage_estimation(last_value, self.discount, self.tau)
        else:
            self.replay_buffer.discount_reward(last_value, self.discount)

        for _ in range( self.opt_epochs ):    
            for batch in self.replay_buffer.one_iteration(self.batch_size, self.sample_key, self.shuffle):
                infos = self.update( batch )
                self.logger.add_update_info( infos )
        
        atu.copy_model_params_from_to(self.pf, self.target_pf)



    def update(self, batch):
        self.training_update_num += 1
        
        obs = batch['obs']
        actions = batch['acts']
        advs = batch['advs']
        estimate_returns = batch['estimate_returns']

        obs = torch.Tensor(obs).to( self.device )
        actions = torch.Tensor(actions).to( self.device )
        advs = torch.Tensor(advs).to( self.device )
        estimate_returns = torch.Tensor(estimate_returns).to( self.device )

        # Normalize the advantage
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        out = self.pf.update( obs, actions )
        log_probs = out['log_prob']
        ent = out['ent']
       
        target_out = self.target_pf.update( obs, actions )
        target_log_probs = target_out['log_prob']
        
        ratio = torch.exp( log_probs - target_log_probs )            

        surrogate_loss_pre_clip = ratio * advs
        surrogate_loss_clip = torch.clamp(ratio, 
                        1.0 - self.clip_para,
                        1.0 + self.clip_para) * advs

        policy_loss = -torch.mean(torch.min(surrogate_loss_clip, surrogate_loss_pre_clip))
        policy_loss = policy_loss - self.entropy_coeff * ent.mean()

        values = self.vf(obs)
        vf_loss = self.vf_criterion( values, estimate_returns )

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        info = {}
        info['Traning/policy_loss'] = policy_loss.item()
        info['Traning/vf_loss'] = vf_loss.item()

        info['advs/mean'] = advs.mean().item()
        info['advs/std'] = advs.std().item()
        info['advs/max'] = advs.max().item()
        info['advs/min'] = advs.min().item()

        info['ratio/max'] = ratio.max().item()
        info['ratio/min'] = ratio.min().item()
#        print(info["ratio/max"])
        return info

    @property
    def networks(self):
        return [
            self.pf,
            self.vf,
            self.target_pf
        ]
    
