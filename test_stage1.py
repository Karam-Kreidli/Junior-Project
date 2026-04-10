"""
BioReef.ai — Stage 1 Evaluation (Flat Head)
============================================
Evaluates on the isolated 10% blind test set.
Matches the flat-head architecture from train_stage1.py.
"""

import os
import shutil
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import logging
import argparse
import cv2

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from bioreef.evaluation.hd_evaluator import HDEvaluator
from bioreef.data.data_factory import ContextHarvester

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("test_stage1")

# =============================================================================
# Dataset (224x224 — matches training)
# =============================================================================

class TestDataset(Dataset):
    def __init__(self, samples, img_dir):
        self.samples = samples
        self.img_dir = img_dir
        self.harvester = ContextHarvester(target_resolution=224, small_object_threshold=0.05)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            s = self.samples[idx]
            frame = cv2.imread(s['img_path'], cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Unreadable")
            streams = self.harvester.harvest(frame, s['bbox'])
            return {
                'streams': streams,
                'label': s['class_idx'],
                'species': s['species']
            }
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))

# =============================================================================
# Helpers
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
            'genus': row['genus'], 'family': row['family'], 'species': row['species']
        }
    return tree

def compute_map(y_true, y_scores, num_classes):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    if y_true_bin.shape[1] <= 1:
        return 0.0
    return average_precision_score(y_true_bin, y_scores, average="macro")

def get_blind_test_set(csv_path):
    """Extract the final 10% as a deterministic blind test set."""
    import pandas as pd
    import random

    df = pd.read_csv(csv_path).dropna(subset=['species'])
    unique_sp = sorted(df['species'].unique().tolist())
    sp_to_idx = {sp: i for i, sp in enumerate(unique_sp)}
    idx_to_sp = {i: sp for sp, i in sp_to_idx.items()}

    all_samples = []
    for _, row in df.iterrows():
        img_path = ""
        for alt in ["data_oz/frames_waternet_1", "data_oz/frames_waternet_2", "/media/openuae/UUI/frames_waternet_3"]:
            candid = os.path.join(alt, row['file_name'])
            if os.path.exists(candid):
                img_path = candid
                break
        if img_path:
            all_samples.append({
                'img_path': img_path,
                'bbox': [int(row['x0']), int(row['y0']), int(row['x1']), int(row['y1'])],
                'class_idx': sp_to_idx[row['species']],
                'species': row['species']
            })

    random.seed(42)
    random.shuffle(all_samples)
    test_samples = all_samples[int(len(all_samples) * 0.9):]
    return test_samples, len(unique_sp), idx_to_sp

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data_oz/metadata/frame_metadata.csv")
    parser.add_argument("--img_dir", type=str, default="data_oz/frames_waternet")
    parser.add_argument("--weights", type=str, default="bioreef_stage1.pt")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if not os.path.exists(args.weights):
        logger.error(f"Weights not found: {args.weights}")
        return

    shutil.copy(args.weights, "bioreef_stage1_final.pt")

    output_dir = "outputs/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    test_samples, num_classes, idx_to_sp = get_blind_test_set(args.csv_path)
    logger.info(f"Blind Test: {len(test_samples)} images, {num_classes} species")

    test_ds = TestDataset(test_samples, args.img_dir)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # Model: frozen backbone + MCEAM + flat head (matches training)
    backbone = ViTBackbone(freeze=True).to(device)
    mceam = MCEAM(embed_dim=768, num_context_levels=3, output_dim=256, num_heads=8).to(device)
    head = nn.Linear(256, num_classes).to(device)

    ckpt = torch.load("bioreef_stage1_final.pt", map_location=device, weights_only=True)
    mceam.load_state_dict(ckpt['mceam'])
    head.load_state_dict(ckpt['head'])

    mceam.eval()
    head.eval()

    hd_evaluator = HDEvaluator(taxonomy_tree=get_taxonomy_tree(args.csv_path), output_dir=output_dir)

    all_preds = []
    all_trues = []
    all_scores = []

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_dl):
            streams = {k: v.to(device) for k, v in batch['streams'].items()}
            features = backbone(streams)
            out = mceam(features)
            preds = head(out['embedding'])
            probs = torch.softmax(preds, dim=1).cpu().numpy()

            all_scores.append(probs)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_trues.extend(batch['label'].numpy())

            for p_idx, t_str in zip(preds.argmax(dim=1).cpu().numpy(), batch['species']):
                hd_evaluator.log_prediction(idx_to_sp[p_idx], t_str)

    all_scores_np = np.vstack(all_scores)
    all_trues_np = np.array(all_trues)
    all_preds_np = np.array(all_preds)

    test_map = compute_map([s['class_idx'] for s in test_samples], all_scores_np, num_classes)
    metrics = hd_evaluator.compute_aggregate()

    # Top-1 accuracy
    top1_acc = np.mean(all_preds_np == all_trues_np)

    # Top-5 accuracy
    top5_preds = np.argsort(all_scores_np, axis=1)[:, -5:]
    top5_correct = sum(t in top5_preds[i] for i, t in enumerate(all_trues_np))
    top5_acc = top5_correct / len(all_trues_np)

    logger.info("=" * 60)
    logger.info("STAGE 1 FINAL METRICS (BLIND TEST)")
    logger.info("=" * 60)
    logger.info(f"Macro mAP  : {test_map:.4f}")
    logger.info(f"Top-1 Acc  : {top1_acc:.2%}")
    logger.info(f"Top-5 Acc  : {top5_acc:.2%}")
    logger.info(f"Mean HD    : {metrics['mean_hd']:.4f}")
    logger.info("=" * 60)

    # Per-species breakdown
    from collections import Counter
    species_correct = Counter()
    species_total = Counter()
    for t, p in zip(all_trues_np, all_preds_np):
        sp_name = idx_to_sp[t]
        species_total[sp_name] += 1
        if t == p:
            species_correct[sp_name] += 1

    per_species = []
    for sp_name, total in species_total.items():
        acc = species_correct[sp_name] / total
        per_species.append((sp_name, acc, total))

    per_species.sort(key=lambda x: -x[1])

    logger.info(f"\n  Top 20 species (by accuracy):")
    for name, acc, count in per_species[:20]:
        logger.info(f"    {name:40s} {acc:.2%}  (n={count})")

    logger.info(f"\n  Bottom 20 species (by accuracy):")
    for name, acc, count in per_species[-20:]:
        logger.info(f"    {name:40s} {acc:.2%}  (n={count})")

    logger.info("=" * 60)

    # Confusion Matrix
    cm = confusion_matrix(all_trues, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title(f'Stage 1 Confusion Matrix ({num_classes} Species)', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix → {cm_path}")

if __name__ == "__main__":
    main()
