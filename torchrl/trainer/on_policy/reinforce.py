import torch
import torch.optim as optim
from .on_policy_trainer import OnPolicyTrainer
from torchrl.networks.nets import ZeroNet


class ReinforceTrainer(OnPolicyTrainer):
    """
    Reinforce
    """
    def __init__(
        self,
        plr: float = 3e-4,
        optimizer_class: object = optim.Adam,
        entropy_coeff: float = 0.001,
        **kwargs
    ):
        super(ReinforceTrainer, self).__init__(**kwargs)
        self.plr = plr
        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr
        )
        self.entropy_coeff = entropy_coeff
        self.gae = False

    def update(self, batch):
        self.training_update_num += 1

        info = {}

        obs = batch['obs']
        acts = batch['acts']
        advs = batch['advs']

        info['advs/mean'] = advs.mean().item()
        info['advs/std'] = advs.std().item()
        info['advs/max'] = advs.max().item()
        info['advs/min'] = advs.min().item()

        out = self.agent.update(obs, acts)
        log_probs = out['log_prob']
        ent = out['ent']

        assert log_probs.shape == advs.shape
        policy_loss = -log_probs * advs
        policy_loss = policy_loss.mean() - self.entropy_coeff * ent.mean()

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pf.parameters(), 0.5)
        self.pf_optimizer.step()

        info['Training/policy_loss'] = policy_loss.item()

        info['ent'] = ent.mean().item()
        info['logprob/mean'] = log_probs.mean().item()
        info['logprob/std'] = log_probs.std().item()
        info['logprob/max'] = log_probs.max().item()
        info['logprob/min'] = log_probs.min().item()

        if 'std' in out:
            # Log for continuous
            std = out["std"]
            info['std/mean'] = std.mean().item()
            info['std/std'] = std.std().item()
            info['std/max'] = std.max().item()
            info['std/min'] = std.min().item()
        return info

    @property
    def optimizers(self):
        return [
            ("pf_optim", self.pf_optimizer)
        ]
