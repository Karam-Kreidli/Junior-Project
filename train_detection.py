"""
BioReef.ai — Detection Training (DINO + FDR)
=============================================
Trains the DINO decoder + FDR bbox head on frozen DINOv3 patch tokens.

Usage:
    torchrun --nproc_per_node=2 train_detection.py

    torchrun --nproc_per_node=2 train_detection.py \
        --csv_path data/metadata/frame_metadata.csv \
        --img_dir /media/openuae/UUI/frames_waternet \
        --epochs 24 --batch_size 4 --input_size 512

Architecture:
    DINOv3 ViT-B/16 (frozen) → patch tokens (B, 1024, 768) at 512×512
    → BioReefDetector (DINO decoder + FDR head + CDN)
    → DetectionLoss (Hungarian matching + GIoU + DFL + CDN loss)

Reference:
    Zhang et al. (2022), "DINO: DETR with Improved DeNoising Anchor Boxes"
    Peng et al. (2024), "D-FINE: Redefine Regression Task in DETRs as FDR"
"""

import os
import argparse
import logging

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from bioreef.models.backbone import ViTBackbone
from bioreef.models.detector import BioReefDetector
from bioreef.losses.detection_loss import DetectionLoss
from bioreef.data.detection_dataset import (
    load_detection_data,
    split_detection_frames,
    DetectionDataset,
    detection_collate,
)


# =============================================================================
# DDP Setup (mirrors train_stage1.py)
# =============================================================================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def get_logger(local_rank: int):
    logger = logging.getLogger("train_detection")
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


# =============================================================================
# Training / Validation
# =============================================================================

def train_one_epoch(
    backbone, detector, criterion, optimizer, scaler, dataloader, device, epoch, logger_fn
):
    detector.train()
    total_loss = 0.0
    num_batches = 0

    pbar = logger_fn(dataloader, epoch, "Train")

    for batch in pbar:
        images = batch['images'].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                patch_tokens = backbone.extract_patch_tokens(images)
            outputs = detector(patch_tokens, targets=targets)
            losses = criterion(outputs, targets)
            loss = losses['total_loss']

        scaler.scale(loss).backward()
        # Gradient clipping — standard for DETR-family models
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(backbone, detector, criterion, dataloader, device, epoch, logger_fn):
    detector.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = logger_fn(dataloader, epoch, "Val")

    for batch in pbar:
        images = batch['images'].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]

        with torch.amp.autocast('cuda'):
            patch_tokens = backbone.extract_patch_tokens(images)
            outputs = detector(patch_tokens, targets=None)
            # For validation loss, we still need targets for matching
            losses = criterion(outputs, targets)
            loss = losses['total_loss']

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BioReef.ai Detection Training")
    parser.add_argument("--csv_path", type=str, default="data/metadata/frame_metadata.csv")
    parser.add_argument("--img_dir", type=str, default="/media/openuae/UUI/frames_waternet")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_fdr_bins", type=int, default=17)
    parser.add_argument("--output", type=str, default="bioreef_detection.pt")
    args = parser.parse_args()

    local_rank = setup_ddp()
    logger = get_logger(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    # --- Data ---
    img_dirs = [
        args.img_dir,
        "data/frames_waternet_1",
        "data/frames_waternet_2",
        "/media/openuae/UUI/frames_waternet_3",
    ]
    frames, sp_to_idx, idx_to_sp = load_detection_data(args.csv_path, img_dirs)
    num_classes = len(sp_to_idx)
    train_frames, val_frames, _ = split_detection_frames(frames)

    train_ds = DetectionDataset(train_frames, input_size=args.input_size, is_train=True)
    val_ds = DetectionDataset(val_frames, input_size=args.input_size, is_train=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, collate_fn=detection_collate,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True, collate_fn=detection_collate,
    )

    # --- Models ---
    backbone = ViTBackbone(freeze=True).to(device)

    detector = BioReefDetector(
        backbone_dim=backbone.embed_dim,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_classes=num_classes,
        num_decoder_layers=args.num_decoder_layers,
        num_fdr_bins=args.num_fdr_bins,
    ).to(device)

    detector = DDP(detector, device_ids=[local_rank], find_unused_parameters=False)

    criterion = DetectionLoss(
        num_classes=num_classes,
        num_fdr_bins=args.num_fdr_bins,
    ).to(device)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        detector.parameters(),
        lr=args.lr * world_size,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    # --- Progress bar helper ---
    def make_pbar(dataloader, epoch, phase):
        desc = f"Epoch {epoch}/{args.epochs} [{phase}]"
        if local_rank == 0:
            return tqdm(dataloader, desc=desc)
        return dataloader

    # --- Logging ---
    if local_rank == 0:
        logger.info("=" * 60)
        logger.info("BioReef.ai — Detection Training (DINO + FDR)")
        logger.info(f"Backbone     : DINOv3 ViT-B/16 (FROZEN)")
        logger.info(f"Detector     : DINO decoder ({args.num_decoder_layers}L) + FDR ({args.num_fdr_bins} bins)")
        logger.info(f"Resolution   : {args.input_size}×{args.input_size}")
        logger.info(f"Queries      : {args.num_queries}")
        logger.info(f"Classes      : {num_classes} species (+1 background)")
        logger.info(f"Epochs       : {args.epochs}")
        logger.info(f"Batch        : {args.batch_size} × {world_size} = {args.batch_size * world_size}")
        logger.info(f"Train/Val    : {len(train_frames)} / {len(val_frames)} frames")
        logger.info(f"LR           : {args.lr * world_size:.1e}")
        logger.info(f"Output       : {args.output}")
        logger.info("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            backbone, detector, criterion, optimizer, scaler,
            train_dl, device, epoch, make_pbar,
        )
        val_loss = validate(
            backbone, detector, criterion, val_dl, device, epoch, make_pbar,
        )

        scheduler.step()

        # Aggregate losses across ranks
        loss_tensor = torch.tensor([train_loss, val_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_train = loss_tensor[0].item() / world_size
        avg_val = loss_tensor[1].item() / world_size

        if local_rank == 0:
            logger.info(
                f"Epoch [{epoch:02d}/{args.epochs}] "
                f"Train: {avg_train:.4f} | Val: {avg_val:.4f}"
            )
            logger.info(f"  {report_memory(local_rank)}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    'detector': detector.module.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val,
                    'num_classes': num_classes,
                    'sp_to_idx': sp_to_idx,
                    'args': vars(args),
                }, args.output)
                logger.info(f"  [+] New best model saved! (Val Loss: {avg_val:.4f})")

    cleanup_ddp()


if __name__ == "__main__":
    main()
