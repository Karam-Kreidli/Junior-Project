"""
BioReef.ai — Stage-1 training building blocks.

The reusable pieces of the DDP trainer (loss, sampler, dataset, EMA, metrics,
DDP infra), extracted from scripts/training/train_stage1.py so the script is a
thin torchrun orchestrator. Training is off the inference data-path, so this is
an unnumbered peer package (like the stages it draws preprocessing from).
"""

from .losses import CBFocalLoss
from .sampler import BalancedDistributedSampler
from .dataset import Stage1Dataset
from .ema import EMA
from .metrics import compute_map
from .ddp import setup_ddp, cleanup_ddp, get_logger, report_memory, safe_imread

__all__ = [
    "CBFocalLoss", "BalancedDistributedSampler", "Stage1Dataset", "EMA",
    "compute_map", "setup_ddp", "cleanup_ddp", "get_logger", "report_memory",
    "safe_imread",
]
