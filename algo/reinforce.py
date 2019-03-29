import numpy as np

import torch
import torch.optim as optim

from .on_rl_algo import OnRLAlgo
from networks.nets import ZeroNet

class Reinforce(OnRLAlgo):
    """
    Reinforce
    """
    def __init__(
        self,
        pf, 
        plr,
        optimizer_class=optim.Adam,
        entropy_coeff = 0.001,
        **kwargs
    ):
        super(Reinforce, self).__init__(**kwargs)
        self.pf = pf
        # Use a vacant value network to simplify the code structure
        self.vf = ZeroNet()
        self.to(self.device)

        self.plr = plr

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )

        self.entropy_coeff = entropy_coeff
        
        self.gae = False
    
    def update(self, batch):
        self.training_update_num += 1
        
        obs = batch['obs']
        actions = batch['actions']
        advs = batch['advs']

        obs = torch.Tensor(obs).to( self.device )
        actions = torch.Tensor(actions).to( self.device )
        advs = torch.Tensor(advs).to( self.device )

        # Normalize the advantage
        # print(advs)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        #print(advs)

        out = self.pf.update( obs, actions )
        
        log_probs = out['log_prob']
        # print(log_probs)
        ent = out['ent']

        policy_loss = -log_probs * advs
        #print(log_probs.shape)
        #print(advs.shape)
        policy_loss = policy_loss.mean() - self.entropy_coeff * ent.mean()

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        info = {}
        info['Traning/policy_loss'] = policy_loss.item()

        info['returns/mean'] = advs.mean().item()
        info['returns/std'] = advs.std().item()
        info['returns/max'] = advs.max().item()
        info['returns/min'] = advs.min().item()

        return info
