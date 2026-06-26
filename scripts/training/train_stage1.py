"""
BioReef.ai — Stage 1 Training
==============================
Two modes:

  1. Standard (default): Train MCEAM + head from scratch with CB-Focal Loss.
        torchrun --nproc_per_node=2 train_stage1.py

  2. Decoupled: Load existing checkpoint, freeze MCEAM, re-train head only
     with class-balanced sampling (Kang et al., 2020).
        torchrun --nproc_per_node=2 train_stage1.py --decouple --checkpoint bioreef_stage1_ddp.pt

     Use this to test whether classifier bias (not embedding quality) is the
     bottleneck before committing to a full retrain with CB-Focal Loss.
"""

import os
import sys
import math
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import logging
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- repo-root bootstrap: this script lives in scripts/<area>/; add the
# repo root (two levels up) to sys.path so `import bioreef` resolves no
# matter the cwd or how the script is invoked. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from bioreef._2_stage1._22_backbone import ViTBackbone
from bioreef._2_stage1._23_mceam import MCEAM
from bioreef._1_preprocess._11_restoration import WaterNetRestorer
from bioreef._1_preprocess._12_context import ContextHarvester
from bioreef._1_preprocess._13_augmentation import MarineAugmentor
from bioreef._4_eval._42_hd_evaluator import HDEvaluator
from bioreef._2_stage1 import HSLMLoss

# =============================================================================
# DDP Setup
# =============================================================================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def get_logger(local_rank):
    logger = logging.getLogger("train_ddp")
    logger.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger

def safe_imread(path):
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
# CB-Focal Loss (standard mode)
# =============================================================================

class CBFocalLoss(nn.Module):
    """Cui et al. (2019) — effective number weighting + focal modulation."""
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, device='cuda'):
        super().__init__()
        samples_per_class = np.array(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32, device=device))
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weights)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# =============================================================================
# Balanced Distributed Sampler (decoupled mode)
# =============================================================================

class BalancedDistributedSampler(Sampler):
    """
    Samples equal numbers from each class, distributed across DDP ranks.

    For each epoch, draws `samples_per_class` examples from every class
    (with replacement for minority classes). This gives the head equal
    gradient signal across the full species distribution.

    samples_per_class defaults to the median class count — a middle ground
    that oversamples rare classes without excessively repeating common ones.
    """

    def __init__(self, samples, num_replicas, rank, samples_per_class=None, seed=0):
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        class_to_indices = {}
        for i, s in enumerate(samples):
            cls = s['class_idx']
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

        # Pad to be evenly divisible across ranks
        indices += indices[:(self.total_size - len(indices))]

        # Each rank takes every num_replicas-th element
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

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

# Dataset-prep functions now live in the library so other scripts import them
# from there instead of from this training script (bioreef._1_preprocess._15_dataset_split).
# Re-exported here for backward compatibility with anything that still does
# `from train_stage1 import split_dataset` etc.
from bioreef._1_preprocess._15_dataset_split import (   # noqa: E402,F401
    is_placeholder_species,
    get_taxonomy_tree,
    build_taxonomy_maps,
    split_dataset,
)

def compute_map(y_true, y_scores, num_classes):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    if y_true_bin.shape[1] <= 1:
        return 0.0
    try:
        return average_precision_score(y_true_bin, y_scores, average="macro")
    except Exception:
        return 0.0

def report_memory(local_rank):
    allocated = torch.cuda.memory_allocated(local_rank) / (1024**3)
    reserved = torch.cuda.memory_reserved(local_rank) / (1024**3)
    return f"VRAM [GPU {local_rank}]: {allocated:.2f} GB / {reserved:.2f} GB"


class EMA:
    """Exponential Moving Average of trainable parameters.

    All ranks maintain identical EMA shadow copies (DDP keeps weights in sync,
    and the EMA update is deterministic), so no cross-rank communication needed.
    """
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
        """Swap current params with EMA shadow; return backup of original params."""
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

# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    from bioreef._9_pipeline.config import TrainingConfig, DEFAULT_CONFIG_PATH
    parser = argparse.ArgumentParser(
        description="BioReef.ai Stage 1 DDP trainer. Settings come from the "
                    "config file's `training:` section; launch with torchrun, "
                    "e.g. torchrun --nproc_per_node=2 "
                    "scripts/training/train_stage1.py [--config config.yaml]")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help=f"Pipeline config YAML. Default: {DEFAULT_CONFIG_PATH}")
    cli = parser.parse_args()

    # All training knobs live in config.yaml's training section. `args` is the
    # TrainingConfig dataclass; its attribute names match what the loop below
    # already uses (csv_path, img_dir, epochs, hslm, family_weight, ...), so
    # the training logic is unchanged.
    args = TrainingConfig.from_yaml(cli.config)

    if args.epochs is None:
        args.epochs = 10 if args.decouple else 30

    local_rank = setup_ddp()
    logger = get_logger(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    logger.info(f"Initialized DDP (World Size: {world_size})")

    train_samples, val_samples, _test_samples, num_classes, idx_to_sp, sp_counts = split_dataset(
        args.csv_path, args.img_dir, min_samples=args.min_samples
    )

    logger.info(f"Loaded {len(train_samples) + len(val_samples)} images across {num_classes} species.")

    train_ds = Stage1Dataset(train_samples, args.img_dir, is_train=True, use_waternet=args.use_waternet)
    val_ds = Stage1Dataset(val_samples, args.img_dir, is_train=False, use_waternet=args.use_waternet)

    if args.decouple:
        train_sampler = BalancedDistributedSampler(
            train_samples, num_replicas=world_size, rank=local_rank
        )
    else:
        train_sampler = DistributedSampler(train_ds, shuffle=True)

    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True, prefetch_factor=2)

    backbone = ViTBackbone(freeze=True).to(device)
    mceam = MCEAM(embed_dim=768, num_context_levels=3, output_dim=256, num_heads=8, use_checkpointing=True).to(device)
    head = nn.Linear(256, num_classes).to(device)

    if args.decouple:
        # Load checkpoint and freeze MCEAM — only head is trainable.
        # DDP cannot wrap a fully-frozen module, so mceam stays unwrapped.
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        mceam.load_state_dict(ckpt['mceam'])
        head.load_state_dict(ckpt['head'])
        for p in mceam.parameters():
            p.requires_grad_(False)
        mceam.eval()
        mceam_ddp = mceam  # no DDP — frozen, called under no_grad
        head_ddp = DDP(head, device_ids=[local_rank])
        if local_rank == 0:
            logger.info(f"Loaded checkpoint: {args.checkpoint}")
            logger.info("MCEAM frozen — training head only.")
    else:
        mceam_ddp = DDP(mceam, device_ids=[local_rank], find_unused_parameters=False)
        head_ddp = DDP(head, device_ids=[local_rank])

    # HSLM applies only to standard mode (decouple uses a balanced sampler,
    # which would partially fight CB-Focal's class weighting).
    use_hslm = args.hslm and not args.decouple
    if args.hslm and args.decouple and local_rank == 0:
        logger.warning("--hslm ignored in --decouple mode (incompatible with "
                       "the balanced sampler); using CrossEntropyLoss.")

    if args.decouple:
        # Head-only optimizer at higher LR — fewer parameters, balanced batches
        optimizer = optim.AdamW(head_ddp.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
    else:
        optimizer = optim.AdamW(
            list(mceam_ddp.parameters()) + list(head_ddp.parameters()),
            lr=1e-4 * world_size,
            weight_decay=0.01
        )
        if use_hslm:
            s2g, s2f, n_gen, n_fam, n_missing = build_taxonomy_maps(
                idx_to_sp, get_taxonomy_tree(args.csv_path)
            )
            if local_rank == 0 and n_missing:
                logger.warning(
                    f"{n_missing}/{num_classes} species missing taxonomy — "
                    "mapped to __unknown__ genus/family buckets."
                )
            criterion = HSLMLoss(
                sp_counts, s2g, s2f, n_gen, n_fam,
                family_weight=args.family_weight,
                genus_weight=args.genus_weight,
                species_weight=args.species_weight,
                beta=args.beta, gamma=args.gamma, device=device,
            )
        else:
            criterion = CBFocalLoss(sp_counts, beta=args.beta, gamma=args.gamma, device=device)

    epochs = args.epochs
    warmup_epochs = args.warmup_epochs if not args.decouple else 0

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler('cuda')
    hd_evaluator = HDEvaluator(taxonomy_tree=get_taxonomy_tree(args.csv_path))

    # EMA — tracks MCEAM (if trainable) and head. Validation + best checkpoint use EMA weights.
    ema_mceam = None if args.decouple else EMA(mceam_ddp.module, decay=args.ema_decay)
    ema_head = EMA(head_ddp.module, decay=args.ema_decay)
    best_hd = float('inf')

    if local_rank == 0:
        logger.info("=" * 60)
        if args.decouple:
            logger.info("BioReef.ai — Decoupled Head Re-Training")
            logger.info(f"Checkpoint   : {args.checkpoint}")
            logger.info(f"Trainable    : Head only (MCEAM frozen)")
            logger.info(f"Sampler      : BalancedDistributedSampler")
            logger.info(f"Loss         : CrossEntropyLoss")
            logger.info(f"LR           : 1e-3")
            logger.info(f"Output       : bioreef_stage1_decoupled.pt")
        else:
            logger.info("BioReef.ai — Standard Training")
            logger.info(f"Trainable    : MCEAM + Head")
            logger.info(f"Sampler      : DistributedSampler")
            if use_hslm:
                logger.info(
                    f"Loss         : HSLM (CB-Focal species + marginalized "
                    f"genus/family | weights f={args.family_weight} "
                    f"g={args.genus_weight} s={args.species_weight}, "
                    f"beta={args.beta}, gamma={args.gamma})"
                )
            else:
                logger.info(f"Loss         : CB-Focal Loss (beta={args.beta}, gamma={args.gamma})")
            logger.info(f"Warmup       : {warmup_epochs} epochs (linear)")
            logger.info(f"Output       : bioreef_stage1.pt")
        logger.info(f"Backbone     : DINOv3 ViT-B/16 (FULLY FROZEN)")
        logger.info(f"Resolution   : 224x224")
        logger.info(f"Head         : Linear(256, {num_classes})")
        logger.info(f"Epochs       : {epochs}")
        logger.info(f"Batch        : {args.batch_size} x {world_size} = {args.batch_size * world_size}")
        logger.info(f"Train/Val    : {len(train_samples)} / {len(val_samples)}")
        logger.info("=" * 60)

    output_path = "bioreef_stage1_decoupled.pt" if args.decouple else "bioreef_stage1.pt"

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)

        if not args.decouple:
            mceam_ddp.train()
        head_ddp.train()
        train_loss = 0.0

        optimizer.zero_grad()

        train_iter = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [Train]") if local_rank == 0 else train_dl

        for batch in train_iter:
            streams = {k: v.to(device) for k, v in batch['streams'].items()}
            labels = batch['label'].to(device)

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    features = backbone(streams)
                if args.decouple:
                    with torch.no_grad():
                        out = mceam_ddp(features)
                else:
                    out = mceam_ddp(features)
                preds = head_ddp(out['embedding'])
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # EMA update (after the real weights have been stepped)
            if ema_mceam is not None:
                ema_mceam.update(mceam_ddp.module)
            ema_head.update(head_ddp.module)

            train_loss += loss.item()

        tensor_train_loss = torch.tensor([train_loss], device=device)
        dist.all_reduce(tensor_train_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = (tensor_train_loss.item() / world_size) / len(train_dl)

        # --- Validation (uses EMA weights) ---
        mceam_ddp.eval()
        head_ddp.eval()
        val_loss = 0.0
        all_scores = []
        all_targets = []
        hd_evaluator.reset()

        # Swap EMA weights in for evaluation
        mceam_backup = ema_mceam.apply_to(mceam_ddp.module) if ema_mceam is not None else None
        head_backup = ema_head.apply_to(head_ddp.module)

        val_iter = tqdm(val_dl, desc=f"Epoch {epoch}/{epochs} [Val]") if local_rank == 0 else val_dl

        with torch.no_grad():
            for batch in val_iter:
                streams = {k: v.to(device) for k, v in batch['streams'].items()}
                labels = batch['label'].to(device)

                with torch.amp.autocast('cuda'):
                    features = backbone(streams)
                    out = mceam_ddp(features)
                    preds = head_ddp(out['embedding'])
                    loss = criterion(preds, labels)

                val_loss += loss.item()
                probs = torch.softmax(preds, dim=1)
                all_scores.append(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy().tolist())

                for p_idx, t_str in zip(preds.argmax(dim=1).cpu().numpy(), batch['species']):
                    hd_evaluator.log_prediction(idx_to_sp[p_idx], t_str)

        # Restore training weights for the next epoch
        if ema_mceam is not None and mceam_backup is not None:
            ema_mceam.restore(mceam_ddp.module, mceam_backup)
        ema_head.restore(head_ddp.module, head_backup)

        tensor_val_loss = torch.tensor([val_loss], device=device)
        dist.all_reduce(tensor_val_loss, op=dist.ReduceOp.SUM)
        avg_val_loss = (tensor_val_loss.item() / world_size) / len(val_dl)

        local_map = compute_map(
            all_targets,
            np.vstack(all_scores) if all_scores else np.zeros((1, num_classes)),
            num_classes
        )
        local_hd_data = hd_evaluator.compute_aggregate()
        local_hd = local_hd_data['mean_hd']
        local_acc = local_hd_data['species_accuracy']

        # Top-5 accuracy from softmax scores
        if all_scores and len(all_targets) > 0:
            scores_arr = np.vstack(all_scores)
            targets_arr = np.array(all_targets)
            top5_preds = np.argsort(scores_arr, axis=1)[:, -5:]
            local_top5 = float(np.mean([t in top5_preds[i] for i, t in enumerate(targets_arr)]))
        else:
            local_top5 = 0.0

        metric_tensor = torch.tensor([local_map, local_hd, local_acc, local_top5], device=device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        global_map = metric_tensor[0].item() / world_size
        global_hd = metric_tensor[1].item() / world_size
        global_acc = metric_tensor[2].item() / world_size
        global_top5 = metric_tensor[3].item() / world_size

        scheduler.step()

        if local_rank == 0:
            logger.info(f"Epoch [{epoch:02d}/{epochs}] "
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                        f"| Val HD: {global_hd:.4f} | Val mAP: {global_map:.4f} "
                        f"| Top-1: {global_acc*100:.2f}% | Top-5: {global_top5*100:.2f}%")
            # HSLM per-level breakdown (last validation batch) for visibility
            if use_hslm and getattr(criterion, "last_components", None):
                c = criterion.last_components
                logger.info(f"  HSLM components — species: {c['species']:.4f} | "
                            f"genus: {c['genus']:.4f} | family: {c['family']:.4f}")
            logger.info(f"  {report_memory(local_rank)}")

            if global_hd < best_hd:
                best_hd = global_hd
                # Save EMA weights (what validation scored on). For decouple mode,
                # MCEAM is frozen so we save its loaded state unchanged.
                if args.decouple:
                    mceam_state = mceam_ddp.state_dict()
                else:
                    mceam_state = mceam_ddp.module.state_dict()
                    mceam_state.update(ema_mceam.state_dict())
                head_state = head_ddp.module.state_dict()
                head_state.update(ema_head.state_dict())
                # Persist the species index→name mapping in the checkpoint.
                # Without it, decoding the head's class indices back to species
                # depends on re-deriving the split from the exact training
                # image set — which is not reproducible once those images
                # change. Saving idx_to_sp makes the checkpoint self-describing.
                torch.save({
                    'mceam': mceam_state,
                    'head': head_state,
                    'idx_to_sp': idx_to_sp,
                    'num_classes': num_classes,
                    'min_samples': args.min_samples,
                }, output_path)
                logger.info(f"  [+] New best model saved! (HD: {global_hd:.4f})")

    cleanup_ddp()

if __name__ == "__main__":
    main()
