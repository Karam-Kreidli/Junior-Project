"""Class-balanced distributed sampler for decoupled head re-training."""

import math

import numpy as np
from torch.utils.data import Sampler


class BalancedDistributedSampler(Sampler):
    """Draw equal counts per class each epoch (with replacement for minority
    classes), sharded across DDP ranks — equal gradient signal across the full
    species distribution. samples_per_class defaults to the median class count.
    """

    def __init__(self, samples, num_replicas, rank, samples_per_class=None, seed=0):
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        class_to_indices = {}
        for i, s in enumerate(samples):
            cls = s["class_idx"]
            class_to_indices.setdefault(cls, []).append(i)
        self.class_to_indices = class_to_indices
        self.num_classes = len(class_to_indices)

        if samples_per_class is None:
            counts = [len(v) for v in class_to_indices.values()]
            samples_per_class = int(np.median(counts))
        self.samples_per_class = samples_per_class

        total = self.num_classes * self.samples_per_class
        self.total_size = math.ceil(total / num_replicas) * num_replicas
        self.num_samples = self.total_size // num_replicas

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        indices = []
        for cls_indices in self.class_to_indices.values():
            chosen = rng.choice(
                cls_indices,
                size=self.samples_per_class,
                replace=len(cls_indices) < self.samples_per_class,
            )
            indices.extend(chosen.tolist())

        rng.shuffle(indices)
        indices += indices[:(self.total_size - len(indices))]  # pad to divisible
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
