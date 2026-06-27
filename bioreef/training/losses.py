"""Training losses for Stage 1.

CBFocalLoss is the standard-mode species loss; the hierarchical HSLMLoss
(bioreef._2_stage1) generalizes it with genus/family terms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBFocalLoss(nn.Module):
    """Cui et al. (2019) — effective-number class weighting + focal modulation."""

    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, device="cuda"):
        super().__init__()
        samples_per_class = np.array(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.register_buffer(
            "weights", torch.tensor(weights, dtype=torch.float32, device=device)
        )
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.weights)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
