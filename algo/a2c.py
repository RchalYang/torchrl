import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from .on_rl_algo import OnRLAlgo

class A2C(OnRLAlgo):
    """
    Actor Critic
    """
    def __init__(
        self,
        pf, vf, 
        plr, vlr,
        optimizer_class=optim.Adam,
        entropy_coeff = 0.001,
        **kwargs
    ):
        super(A2C, self).__init__(**kwargs)
        self.pf = pf
        self.vf = vf
        self.to(self.device)

        self.plr = plr
        self.vlr = vlr

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )

        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=self.vlr,
        )

        self.entropy_coeff = entropy_coeff
        
        self.vf_criterion = nn.MSELoss()
    
    def update(self, batch):
        self.training_update_num += 1

        info = {}

        obs = batch['obs']
        actions = batch['actions']
        advs = batch['advs']
        estimate_returns = batch['estimate_returns']

        obs = torch.Tensor(obs).to( self.device )
        actions = torch.Tensor(actions).to( self.device )
        advs = torch.Tensor(advs).to( self.device )
        estimate_returns = torch.Tensor(estimate_returns).to( self.device )

        # Normalize the advantage
        info['advs/mean'] = advs.mean().item()
        info['advs/std'] = advs.std().item()
        info['advs/max'] = advs.max().item()
        info['advs/min'] = advs.min().item()

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        out = self.pf.update( obs, actions )
        log_probs = out['log_prob']
        ent = out['ent']

        policy_loss = -log_probs * advs
        policy_loss = policy_loss.mean() - self.entropy_coeff * ent.mean()

        values = self.vf(obs)
        vf_loss = self.vf_criterion( values, estimate_returns )

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        info['Traning/policy_loss'] = policy_loss.item()
        info['Traning/vf_loss'] = vf_loss.item()

        
        info['v_pred/mean'] = values.mean().item()
        info['v_pred/std'] = values.std().item()
        info['v_pred/max'] = values.max().item()
        info['v_pred/min'] = values.min().item()

        info['ent'] = ent.mean().item()

        return info
