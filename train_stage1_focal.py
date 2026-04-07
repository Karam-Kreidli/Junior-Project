"""
BioReef.ai — Stage 1 Dual-GPU DDP Focal Loss Fine-Tuning
=========================================================
Distributed Data Parallel implementation specifically designed for Phase 9
Hard Negative Mining. It resumes from the Phase 8 standard weights and 
swaps the criterion dynamically to Focal Loss to aggressively zero-out 
well-learned common species and strictly penalize the rare 507-class long tail.

Optimizations:
    - Focal Loss (alpha=0.25, gamma=2.0) for extreme dataset imbalance
    - LR reduced 10x to 1e-5 to prevent shatter on frozen backbone embeddings
    - DDP Sharded Datasets with NCCL backend
"""

import os
import json
import time
import math
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import logging
import sys
from tqdm import tqdm

from bioreef.models.backbone import DINOv2Backbone
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester, WaterNetRestorer, MarineAugmentor
from bioreef.evaluation.hd_evaluator import HDEvaluator

# =============================================================================
# Setup & Logging
# =============================================================================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def get_logger(local_rank):
    logger = logging.getLogger("train_focal_ddp")
    logger.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger

def safe_imread(path):
    """Read image while suppressing libpng C-level warnings on stderr."""
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

# =============================================================================
# Custom Focal Loss (Long-Tail Target)
# =============================================================================

class FocalLoss(nn.Module):
    """
    Mathematical replacement for CrossEntropyLoss.
    Exponentially decays loss contribution from common "easy" examples 
    and forces massive gradient backpropagation exclusively to rare/hard classes.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# =============================================================================
# Dataset
# =============================================================================

class Stage1Dataset(Dataset):
    def __init__(self, samples, img_dir, is_train=True, use_waternet=False):
        self.samples = samples
        self.img_dir = img_dir
        
        self.harvester = ContextHarvester(target_resolution=224, small_object_threshold=0.05)
        self.restorer = WaterNetRestorer() if use_waternet else None
        self.augmentor = MarineAugmentor(enabled=is_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            s = self.samples[idx]
            frame = safe_imread(s['img_path'])
            
            if frame is None:
                raise ValueError(f"OpenCV could not read {s['img_path']}")
                
            if self.restorer is not None:
                frame = self.restorer(frame)
                
            augmented = self.augmentor(frame)
            streams = self.harvester.harvest(augmented, s['bbox'])
            
            return {
                'streams': streams,
                'label': s['class_idx'],
                'species': s['species']
            }
        except Exception as e:
            # Randomly skip corrupted files on external drives
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)


def get_taxonomy_tree(csv_path):
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
        
    tree = {}
    for _, row in df.dropna(subset=['species', 'genus', 'family']).iterrows():
        tree[row['species']] = {
            'genus': row['genus'],
            'family': row['family'],
            'species': row['species']
        }
    return tree

def split_dataset(csv_path, img_dir):
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    unique_species = sorted(df['species'].dropna().unique().tolist())
    species_to_class = {sp: idx for idx, sp in enumerate(unique_species)}
    class_to_species = {idx: sp for sp, idx in species_to_class.items()}
    
    all_samples = []
    for _, row in df.iterrows():
        if pd.isna(row['species']):
            continue
            
        sp_name = row['species']
        cls_idx = species_to_class[sp_name]
        img_path = os.path.join(img_dir, row['file_name'])
        
        if not os.path.exists(img_path):
            for alt in [
                "data/frames_waternet_1",
                "data/frames_waternet_2",
                "/media/openuae/UUI/frames_waternet_3",
            ]:
                candidate = os.path.join(alt, row['file_name'])
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if os.path.exists(img_path):
            all_samples.append({
                'img_path': img_path,
                'bbox': [int(row['x0']), int(row['y0']), int(row['x1']), int(row['y1'])],
                'class_idx': cls_idx,
                'species': sp_name
            })

    import random
    random.seed(42)
    random.shuffle(all_samples)
    
    n = len(all_samples)
    train_samples = all_samples[:int(n * 0.8)]
    val_samples = all_samples[int(n * 0.8):int(n * 0.9)]
    return train_samples, val_samples, len(unique_species), class_to_species

def compute_map(y_true, y_scores, num_classes):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    if y_true_bin.shape[1] <= 1:
        return 0.0
    try:
        return average_precision_score(y_true_bin, y_scores, average="micro")
    except Exception:
        return 0.0

def report_memory(local_rank):
    allocated = torch.cuda.memory_allocated(local_rank) / (1024**3)
    reserved = torch.cuda.memory_reserved(local_rank) / (1024**3)
    return f"VRAM [GPU {local_rank}]: {allocated:.2f} GB / {reserved:.2f} GB"

# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/metadata/frame_metadata.csv")
    parser.add_argument("--img_dir", type=str, default="/media/openuae/UUI/frames_waternet")
    parser.add_argument("--use_waternet", action="store_true")
    # Reduced automatically to 10 epochs for Fine-Tuning
    parser.add_argument("--epochs", type=int, default=10)
    # The checkpoint generated by Phase 8 cross-entropy training
    parser.add_argument("--resume", type=str, default="bioreef_stage1_ddp.pt")
    args = parser.parse_args()

    local_rank = setup_ddp()
    logger = get_logger(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    train_samples, val_samples, num_classes, idx_to_sp = split_dataset(args.csv_path, args.img_dir)
    
    train_ds = Stage1Dataset(train_samples, args.img_dir, is_train=True, use_waternet=args.use_waternet)
    val_ds = Stage1Dataset(val_samples, args.img_dir, is_train=False, use_waternet=args.use_waternet)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_dl = DataLoader(train_ds, batch_size=8, sampler=train_sampler, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_dl = DataLoader(val_ds, batch_size=8, sampler=val_sampler, num_workers=4, pin_memory=True, prefetch_factor=2)

    backbone = DINOv2Backbone(freeze=True).to(device)
    
    mceam = MCEAM(embed_dim=768, num_context_levels=3, output_dim=256, num_heads=8, use_checkpointing=True).to(device)
    head = nn.Linear(256, num_classes).to(device)

    # SECURE WEIGHT INGESTION (Transfer Learning)
    if os.path.exists(args.resume):
        if local_rank == 0:
            logger.info(f"Loading Base Checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        mceam.load_state_dict(ckpt['mceam'])
        head.load_state_dict(ckpt['head'])
    else:
        if local_rank == 0:
            logger.warning(f"CRITICAL: Base checkpoint '{args.resume}' missing! Starting from absolute scratch!")

    mceam_ddp = DDP(mceam, device_ids=[local_rank], find_unused_parameters=False)
    head_ddp = DDP(head, device_ids=[local_rank])

    # 10x Smaller LR mathematically protects the precise 1.28 HD weights we just imported
    optimizer = optim.AdamW(
        list(mceam_ddp.parameters()) + list(head_ddp.parameters()), 
        lr=1e-5 * world_size,  
        weight_decay=0.01
    )
    epochs = args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mathematical Replacement of standard CE with Focal Loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scaler = torch.amp.GradScaler('cuda') 

    hd_evaluator = HDEvaluator(taxonomy_tree=get_taxonomy_tree(args.csv_path))
    best_hd = float('inf')
    
    if local_rank == 0:
        logger.info("="*60)
        logger.info("BioReef.ai DDP [FOCAL LOSS] Config")
        logger.info("="*60)
        logger.info(f"GPUs         : {world_size}")
        logger.info(f"Epochs       : {epochs} (Fine-Tuning)")
        logger.info(f"Num Classes  : {num_classes}")
        logger.info("="*60)

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        mceam_ddp.train()
        head_ddp.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        train_iter = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [Train]") if local_rank == 0 else train_dl
        
        for batch in train_iter:
            streams = {k: v.to(device) for k, v in batch['streams'].items()}
            labels_tensor = batch['label'].to(device)
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    features = backbone(streams)
                out = mceam_ddp(features)
                preds = head_ddp(out['embedding'])
                # Aggressive Focal Penalty natively applied here
                loss = criterion(preds, labels_tensor)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += loss.item()

        tensor_train_loss = torch.tensor([train_loss], device=device)
        dist.all_reduce(tensor_train_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = (tensor_train_loss.item() / world_size) / len(train_dl)

        mceam_ddp.eval()
        head_ddp.eval()
        val_loss = 0.0
        all_preds, all_trues, all_scores = [], [], []
        hd_evaluator.reset()
        
        val_iter = tqdm(val_dl, desc=f"Epoch {epoch}/{epochs} [Val]") if local_rank == 0 else val_dl
        
        with torch.no_grad():
            for batch in val_iter:
                streams = {k: v.to(device) for k, v in batch['streams'].items()}
                labels_tensor = batch['label'].to(device)
                
                with torch.amp.autocast('cuda'):
                    features = backbone(streams)
                    out = mceam_ddp(features)
                    preds = head_ddp(out['embedding'])
                    loss = criterion(preds, labels_tensor)
                    
                val_loss += loss.item()
                probs = torch.softmax(preds, dim=1)
                all_scores.append(probs.cpu().numpy())
                for p_idx, t_str in zip(preds.argmax(dim=1).cpu().numpy(), batch['species']):
                    hd_evaluator.log_prediction(idx_to_sp[p_idx], t_str)

        tensor_val_loss = torch.tensor([val_loss], device=device)
        dist.all_reduce(tensor_val_loss, op=dist.ReduceOp.SUM)
        avg_val_loss = (tensor_val_loss.item() / world_size) / len(val_dl)
        
        local_map = compute_map(
            [s['class_idx'] for s in val_samples[val_sampler.rank : len(val_samples) : val_sampler.num_replicas]], 
            np.vstack(all_scores) if all_scores else np.zeros((1, num_classes)),
            num_classes
        )
        local_hd = hd_evaluator.compute_aggregate()['mean_hd']

        metric_tensor = torch.tensor([local_map, local_hd], device=device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        global_map = metric_tensor[0].item() / world_size
        global_hd = metric_tensor[1].item() / world_size

        scheduler.step()
        
        if local_rank == 0:
            logger.info(f"Epoch [{epoch:02d}/{epochs}] "
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                        f"| Val HD: {global_hd:.4f} | Val mAP: {global_map:.4f}")
            logger.info(f"  {report_memory(local_rank)}")
            
            if global_hd < best_hd:
                best_hd = global_hd
                torch.save({
                    'mceam': mceam_ddp.module.state_dict(),
                    'head': head_ddp.module.state_dict()
                }, "bioreef_stage1_focal.pt")
                logger.info(f"  [+] New best FOCAL model saved! (HD: {global_hd:.4f})")

    cleanup_ddp()

if __name__ == "__main__":
    main()
