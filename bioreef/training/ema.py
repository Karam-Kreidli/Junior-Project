"""Exponential moving average of trainable parameters.

All ranks keep identical EMA shadows (DDP syncs weights, the update is
deterministic), so no cross-rank communication is needed. Validation and the
best checkpoint use the EMA weights.
"""

import torch


class EMA:
    def __init__(self, module, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.detach().clone()
            for n, p in module.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, module):
        for n, p in module.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, module):
        """Swap params for EMA shadow; return a backup of the originals."""
        backup = {}
        for n, p in module.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore(self, module, backup):
        for n, p in module.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}
