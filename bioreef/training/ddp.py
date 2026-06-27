"""DDP + logging infra for the Stage-1 trainer (torchrun-launched)."""

import logging
import os
import sys

import cv2
import torch
import torch.distributed as dist


def setup_ddp() -> int:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def get_logger(local_rank: int) -> logging.Logger:
    logger = logging.getLogger("train_ddp")
    logger.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger


def report_memory(local_rank: int) -> str:
    allocated = torch.cuda.memory_allocated(local_rank) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(local_rank) / (1024 ** 3)
    return f"VRAM [GPU {local_rank}]: {allocated:.2f} GB / {reserved:.2f} GB"


def safe_imread(path: str):
    """cv2.imread that silences libjpeg/libpng stderr spam on corrupt frames."""
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    finally:
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stderr)
    return img
