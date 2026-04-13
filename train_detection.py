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

LOSS_KEYS = ["loss_cls", "loss_bbox", "loss_giou", "loss_dfl", "loss_dn_cls"]


def train_one_epoch(
    backbone, detector, criterion, optimizer, scaler, dataloader, device, epoch, logger_fn
):
    detector.train()
    trainable_backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    backbone_trainable = len(trainable_backbone_params) > 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    totals = {"total_loss": 0.0, **{k: 0.0 for k in LOSS_KEYS}}
    num_batches = 0

    pbar = logger_fn(dataloader, epoch, "Train")

    for batch in pbar:
        images = batch['images'].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            if backbone_trainable:
                patch_tokens = backbone.extract_patch_tokens(images)
            else:
                with torch.no_grad():
                    patch_tokens = backbone.extract_patch_tokens(images)
            outputs = detector(patch_tokens, targets=targets)
            losses = criterion(outputs, targets)
            loss = losses['total_loss']

        scaler.scale(loss).backward()
        # Gradient clipping — standard for DETR-family models
        scaler.unscale_(optimizer)
        # Manually all-reduce backbone gradients across ranks (not in DDP)
        if backbone_trainable and world_size > 1:
            for p in trainable_backbone_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
        trainable_params = list(detector.parameters()) + trainable_backbone_params
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()

        totals["total_loss"] += loss.item()
        for k in LOSS_KEYS:
            if k in losses:
                totals[k] += losses[k].item()
        num_batches += 1

    n = max(num_batches, 1)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(backbone, detector, criterion, dataloader, device, epoch, logger_fn):
    detector.eval()
    totals = {"total_loss": 0.0, **{k: 0.0 for k in LOSS_KEYS}}
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

        totals["total_loss"] += loss.item()
        for k in LOSS_KEYS:
            if k in losses:
                totals[k] += losses[k].item()
        num_batches += 1

    n = max(num_batches, 1)
    return {k: v / n for k, v in totals.items()}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BioReef.ai Detection Training")
    parser.add_argument("--csv_path", type=str, default="data_oz/metadata/frame_metadata.csv")
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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--unfreeze_blocks", type=int, default=0,
                        help="Number of final DINOv3 blocks to unfreeze for domain adaptation (0 = fully frozen)")
    args = parser.parse_args()

    local_rank = setup_ddp()
    logger = get_logger(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    # --- Data ---
    img_dirs = [
        args.img_dir,
        "data_oz/frames_waternet_1",
        "data_oz/frames_waternet_2",
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
    if args.unfreeze_blocks > 0:
        backbone.unfreeze_blocks(args.unfreeze_blocks)
    # Backbone is NOT wrapped in DDP; we manually all-reduce unfrozen grads below
    backbone_module = backbone

    detector = BioReefDetector(
        backbone_dim=backbone_module.embed_dim,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_classes=num_classes,
        num_decoder_layers=args.num_decoder_layers,
        num_fdr_bins=args.num_fdr_bins,
    ).to(device)

    detector = DDP(detector, device_ids=[local_rank], find_unused_parameters=True)

    criterion = DetectionLoss(
        num_classes=num_classes,
        num_fdr_bins=args.num_fdr_bins,
    ).to(device)

    # --- Optimizer ---
    # Backbone unfrozen blocks use 10x lower LR to avoid destroying pretrained features
    param_groups = [{"params": detector.parameters(), "lr": args.lr * world_size}]
    backbone_params = [p for p in backbone_module.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr * world_size * 0.1})
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    # --- Resume ---
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        detector.module.load_state_dict(ckpt['detector'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        else:
            for _ in range(start_epoch - 1):
                scheduler.step()
        if local_rank == 0:
            logger.info(f"Resumed from {args.resume} (epoch {start_epoch - 1}, val_loss={best_val_loss:.4f})")

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
        backbone_status = f"DINOv3 ViT-B/16 ({'FROZEN' if args.unfreeze_blocks == 0 else f'top {args.unfreeze_blocks} blocks UNFROZEN @ LR={args.lr * world_size * 0.1:.1e}'})"
        logger.info(f"Backbone     : {backbone_status}")
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

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            backbone, detector, criterion, optimizer, scaler,
            train_dl, device, epoch, make_pbar,
        )
        # Validation runs only on rank 0 (avoids cross-rank dataloader stalls)
        # Pass backbone_module (unwrapped) so we don't hit DDP sync hooks on rank 0 alone
        all_keys = ["total_loss"] + LOSS_KEYS
        if local_rank == 0:
            val_metrics = validate(
                backbone_module, detector.module, criterion, val_dl, device, epoch, make_pbar,
            )
            val_vals_t = torch.tensor([val_metrics.get(k, 0.0) for k in all_keys], device=device)
        else:
            val_vals_t = torch.zeros(len(all_keys), device=device)
        dist.broadcast(val_vals_t, src=0)

        scheduler.step()

        # Aggregate train losses across ranks
        train_vals_t = torch.tensor([train_metrics.get(k, 0.0) for k in all_keys], device=device)
        dist.all_reduce(train_vals_t, op=dist.ReduceOp.SUM)
        train_vals = (train_vals_t / world_size).tolist()
        val_vals   = val_vals_t.tolist()
        avg_train  = train_vals[0]
        avg_val    = val_vals[0]

        if local_rank == 0:
            logger.info(
                f"Epoch [{epoch:02d}/{args.epochs}] "
                f"Train: {avg_train:.4f} | Val: {avg_val:.4f}"
            )
            # Per-component breakdown
            comp_parts = []
            for i, k in enumerate(LOSS_KEYS):
                short = k.replace("loss_", "")
                comp_parts.append(f"{short}={val_vals[i + 1]:.3f}")
            logger.info(f"  Val  components: {' | '.join(comp_parts)}")
            comp_parts = []
            for i, k in enumerate(LOSS_KEYS):
                short = k.replace("loss_", "")
                comp_parts.append(f"{short}={train_vals[i + 1]:.3f}")
            logger.info(f"  Train components: {' | '.join(comp_parts)}")
            logger.info(f"  {report_memory(local_rank)}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    'detector': detector.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
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
