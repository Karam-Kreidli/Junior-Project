"""
BioReef.ai — Stage 1 Dual-GPU DDP Training Script
==================================================
Distributed Data Parallel (DDP) implementation for training the Stage 1
MCEAM module on two 8GB NVIDIA Quadro RTX 4000 GPUs.

Optimizations:
    - Process Group: NCCL backend for fast GPU communication.
    - Dataset Partitioning: torch.utils.data.distributed.DistributedSampler
      prevents image overlap between GPUs.
    - VRAM Management: 
        - batch_size=8 *per GPU* (Effective BN = 16)
        - kwargs `use_checkpointing=True` on MCEAM
        - torch.cuda.amp.GradScaler for Mixed Precision (FP16)
    - Checkpointing: Rank 0 exclusively handles logging and saving weights.

Usage (launch via torchrun):
    torchrun --nproc_per_node=2 train_stage1_ddp.py
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import logging

from bioreef.models.backbone import DINOv2Backbone
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester, WaterNetRestorer, MarineAugmentor
from bioreef.evaluation.hd_evaluator import HDEvaluator

# =============================================================================
# Setup & Logging
# =============================================================================

def setup_ddp():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Destroy the process group."""
    dist.destroy_process_group()

def get_logger(local_rank):
    """Only Rank 0 logs to standard output."""
    logger = logging.getLogger("train_ddp")
    logger.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger


import sys

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
        s = self.samples[idx]
        frame = safe_imread(s['img_path'])
        
        if frame is None:
            # Fallback to white noise if image is corrupted
            frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
            
        if self.restorer is not None:
            frame = self.restorer(frame)
            
        augmented = self.augmentor(frame)
        streams = self.harvester.harvest(augmented, s['bbox'])
        
        return {
            'streams': streams,
            'label': s['class_idx'],
            'species': s['species']
        }

# =============================================================================
# Utilities
# =============================================================================

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
    
    # Create an integer mapping for the true species strings
    unique_species = sorted(df['species'].dropna().unique().tolist())
    species_to_class = {sp: idx for idx, sp in enumerate(unique_species)}
    class_to_species = {idx: sp for sp, idx in species_to_class.items()}
    
    all_samples = []
    
    for _, row in df.iterrows():
        # Clean up Pandas nan/float issues
        if pd.isna(row['species']):
            continue
            
        sp_name = row['species']
        cls_idx = species_to_class[sp_name]
        
        img_path = os.path.join(img_dir, row['file_name'])
        
        # Search across all 3 external hard drive partitions
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
        
        # Verify the file was found
        if os.path.exists(img_path):
            all_samples.append({
                'img_path': img_path,
                'bbox': [int(row['x0']), int(row['y0']), int(row['x1']), int(row['y1'])],
                'class_idx': cls_idx,
                'species': sp_name
            })

    # Shuffle with fixed seed for consistent split
    import random
    random.seed(42)
    random.shuffle(all_samples)
    
    n = len(all_samples)
    train_samples = all_samples[:int(n * 0.8)]
    val_samples = all_samples[int(n * 0.8):int(n * 0.9)]
    
    # Return samples and the number of discovered classes
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
    parser.add_argument("--csv_path", type=str, default="data/metadata/subset_5k_metadata.csv", help="Which subset to read")
    parser.add_argument("--img_dir", type=str, default="/media/openuae/UUI/frames_waternet", help="Prepared images")
    parser.add_argument("--use_waternet", action="store_true", help="Run on-the-fly CNN restoration (slow)")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    # 1. Initialize Process Group
    local_rank = setup_ddp()
    logger = get_logger(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    world_size = dist.get_world_size()
    logger.info(f"Initialized Distributed Data Parallel (World Size: {world_size})")

    # 2. Data Preparation
    if local_rank == 0:
        logger.info(f"Parsing metadata from {args.csv_path} and creating splits...")
    
    train_samples, val_samples, num_classes, idx_to_sp = split_dataset(args.csv_path, args.img_dir)
    
    if local_rank == 0:
        logger.info(f"Loaded {len(train_samples) + len(val_samples)} viable local images mapped to exactly {num_classes} native species.")

    train_ds = Stage1Dataset(train_samples, args.img_dir, is_train=True, use_waternet=args.use_waternet)
    val_ds = Stage1Dataset(val_samples, args.img_dir, is_train=False, use_waternet=args.use_waternet)

    # 3. Distributed Samplers
    # Shards the dataset across GPUs. 
    # Note: We omit WeightedRandomSampler here for simplicity in DDP, 
    # but the larger effective batch size mitigates imbalance issues.
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    # 128GB RAM ALLOWS MASSIVE PREFETCHING: num_workers=16 per GPU, pin_memory=True
    # batch_size=8 PER GPU -> Effective batch_size=16
    train_dl = DataLoader(train_ds, batch_size=8, sampler=train_sampler, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_dl = DataLoader(val_ds, batch_size=8, sampler=val_sampler, num_workers=4, pin_memory=True, prefetch_factor=2)

    # 4. Model Configuration
    # Backbone remains frozen; no need to wrap in DDP or apply gradients
    backbone = DINOv2Backbone(freeze=True).to(device)
    
    # Trainable Modules (MCEAM with memory checkpointing turned ON)
    mceam = MCEAM(
        embed_dim=768, 
        num_context_levels=3, 
        output_dim=256, 
        num_heads=8,
        use_checkpointing=True  # VRAM Optimization!
    ).to(device)
    
    # Dynamically match output nodes to the 496 species discovered in the CSV
    head = nn.Linear(256, num_classes).to(device)

    # Wrap trainable modules in DDP
    # We set find_unused_parameters=False because MCEAM routes all tensors to the output gate
    mceam_ddp = DDP(mceam, device_ids=[local_rank], find_unused_parameters=False)
    head_ddp = DDP(head, device_ids=[local_rank])

    # 5. Optimizer & Criterion
    optimizer = optim.AdamW(
        list(mceam_ddp.parameters()) + list(head_ddp.parameters()), 
        lr=1e-4 * world_size,  # Linear scaling rule for LR based on effective batch
        weight_decay=0.01
    )
    epochs = args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision Support

    hd_evaluator = HDEvaluator(taxonomy_tree=get_taxonomy_tree(args.csv_path))
    
    if local_rank == 0:
        logger.info("="*60)
        logger.info("BioReef.ai DDP Training Config")
        logger.info("="*60)
        logger.info(f"GPUs         : {world_size}")
        logger.info(f"Epochs       : {epochs}")
        logger.info(f"Eff. Batch   : 8 x {world_size} = 16")
        logger.info(f"Num Classes  : {num_classes}")
        logger.info(f"Checkpointing: Enabled (MCEAM)")
        logger.info("="*60)
    
    best_hd = float('inf')
    
    # 6. Training Loop
    for epoch in range(1, epochs + 1):
        # Set epoch for sampler to ensure different shuffles per epoch
        train_sampler.set_epoch(epoch)
        
        mceam_ddp.train()
        head_ddp.train()
        train_loss = 0.0
        
        optimizer.zero_grad()
        
        from tqdm import tqdm
        train_iter = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}") if local_rank == 0 else train_dl
        
        for batch in train_iter:
            streams = {k: v.to(device) for k, v in batch['streams'].items()}
            labels_tensor = batch['label'].to(device)
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    features = backbone(streams)
                
                # Checkpointing is handled internally by MCEAM if use_checkpointing=True
                out = mceam_ddp(features)
                preds = head_ddp(out['embedding'])
                loss = criterion(preds, labels_tensor)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                
            train_loss += loss.item()

        # Average loss across all processes
        tensor_train_loss = torch.tensor([train_loss], device=device)
        dist.all_reduce(tensor_train_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = (tensor_train_loss.item() / world_size) / len(train_dl)

        # --- VAL LOOP ---
        mceam_ddp.eval()
        head_ddp.eval()
        val_loss = 0.0
        
        all_preds, all_trues, all_scores = [], [], []
        hd_evaluator.reset()
        # Note: idx_to_sp mapped dynamically from CSV in setup phase
        
        val_iter = tqdm(val_dl, desc=f"Epoch {epoch} [Val]") if local_rank == 0 else val_dl
        
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
                
                # Gather predictions across all GPUs for accurate validation metric calculation
                probs = torch.softmax(preds, dim=1)
                
                # Store local results
                all_scores.append(probs.cpu().numpy())
                for p_idx, t_str in zip(preds.argmax(dim=1).cpu().numpy(), batch['species']):
                    hd_evaluator.log_prediction(idx_to_sp[p_idx], t_str)

        # Average validation loss
        tensor_val_loss = torch.tensor([val_loss], device=device)
        dist.all_reduce(tensor_val_loss, op=dist.ReduceOp.SUM)
        avg_val_loss = (tensor_val_loss.item() / world_size) / len(val_dl)
        
        # Calculate local metrics
        local_map = compute_map(
            [s['class_idx'] for s in val_samples[val_sampler.rank : len(val_samples) : val_sampler.num_replicas]], 
            np.vstack(all_scores) if all_scores else np.zeros((1, num_classes)),
            num_classes
        )
        local_hd = hd_evaluator.compute_aggregate()['mean_hd']

        # DDP metric sharing (simplified average across nodes)
        metric_tensor = torch.tensor([local_map, local_hd], device=device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        global_map = metric_tensor[0].item() / world_size
        global_hd = metric_tensor[1].item() / world_size

        scheduler.step()
        
        # 7. Logging (Rank 0 ONLY)
        if local_rank == 0:
            logger.info(f"Epoch [{epoch:02d}/{epochs}] "
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                        f"| Val HD: {global_hd:.4f} | Val mAP: {global_map:.4f}")
            logger.info(f"  {report_memory(local_rank)}")
            
            # Substantial HD improvement check
            if global_hd < best_hd:
                best_hd = global_hd
                # Extract original modules from DDP wrapper to save clean state dicts
                torch.save({
                    'mceam': mceam_ddp.module.state_dict(),
                    'head': head_ddp.module.state_dict()
                }, "bioreef_stage1_ddp.pt")
                logger.info(f"  [+] New best DDP model saved! (HD: {global_hd:.4f})")

    cleanup_ddp()

if __name__ == "__main__":
    main()
