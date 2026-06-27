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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import logging
import argparse
import cv2

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- repo-root bootstrap: this script lives in scripts/<area>/; add the
# repo root (two levels up) to sys.path so `import bioreef` resolves no
# matter the cwd or how the script is invoked. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from bioreef._2_stage1._22_backbone import ViTBackbone
from bioreef._2_stage1._23_mceam import MCEAM
from bioreef._4_eval import HDEvaluator
from bioreef._1_preprocess._12_context import ContextHarvester
from bioreef._1_preprocess._15_dataset_split import get_taxonomy_tree
from bioreef.training import compute_map

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

def get_blind_test_set(csv_path, min_samples=20):
    """Extract the final 10% as a deterministic blind test set.

    Uses the same species filtering as train_stage1.py to ensure
    class indices match the trained checkpoint.
    """
    import pandas as pd
    import random
    from collections import Counter

    IMG_DIRS = [
        "data_oz/frames_waternet_1",
        "data_oz/frames_waternet_2",
    ]

    df = pd.read_csv(csv_path).dropna(subset=['species'])

    # First pass: discover frames and count per species
    raw_samples = []
    for _, row in df.iterrows():
        img_path = ""
        for alt in IMG_DIRS:
            candid = os.path.join(alt, row['file_name'])
            if os.path.exists(candid):
                img_path = candid
                break
        if img_path:
            x0, y0, x1, y1 = int(row['x0']), int(row['y0']), int(row['x1']), int(row['y1'])
            raw_samples.append({
                'img_path': img_path,
                'bbox': [x0, y0, x1 - x0, y1 - y0],  # xyxy → xywh
                'species': row['species'],
            })

    # Filter species below threshold (must match train_stage1.py)
    sp_counter = Counter(s['species'] for s in raw_samples)
    kept_species = sorted(sp for sp, cnt in sp_counter.items() if cnt >= min_samples)
    sp_to_idx = {sp: i for i, sp in enumerate(kept_species)}
    idx_to_sp = {i: sp for sp, i in sp_to_idx.items()}

    all_samples = []
    for s in raw_samples:
        if s['species'] not in sp_to_idx:
            continue
        all_samples.append({
            'img_path': s['img_path'],
            'bbox': s['bbox'],
            'class_idx': sp_to_idx[s['species']],
            'species': s['species'],
        })

    random.seed(42)
    random.shuffle(all_samples)
    test_samples = all_samples[int(len(all_samples) * 0.9):]
    return test_samples, len(kept_species), idx_to_sp

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data_oz/metadata/frame_metadata.csv")
    parser.add_argument("--img_dir", type=str, default="data_oz/frames_waternet_1")
    parser.add_argument("--min_samples", type=int, default=20,
                        help="Species sample threshold (must match training).")
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

    test_samples, num_classes, idx_to_sp = get_blind_test_set(args.csv_path, min_samples=args.min_samples)
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

    # Prefer the checkpoint's embedded species mapping (train_stage1.py saves
    # idx_to_sp). The CSV-derived map from get_blind_test_set is only correct
    # if the CSV matches the training image set — so when the checkpoint
    # carries its own, that is authoritative. A size mismatch is flagged.
    ckpt_map = ckpt.get('idx_to_sp')
    if ckpt_map:
        ckpt_map = {int(k): v for k, v in ckpt_map.items()}
        if len(ckpt_map) != num_classes:
            logger.warning(
                f"Checkpoint species map has {len(ckpt_map)} entries but the "
                f"CSV-derived test set has {num_classes} classes — index "
                f"alignment is not guaranteed."
            )
        idx_to_sp = ckpt_map
    else:
        logger.warning(
            "Checkpoint has no embedded species mapping; using the "
            "CSV-derived map (correct only if the CSV matches training)."
        )

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

    # =========================================================================
    # Confusion Matrix — heatmap PNG + raw matrix + ranked confused pairs (#24)
    # =========================================================================
    cm = confusion_matrix(all_trues, all_preds, labels=list(range(num_classes)))

    # --- Heatmap PNG (overview only — 260x260 is unreadable with labels) -----
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title(f'Stage 1 Confusion Matrix ({num_classes} Species)', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix heatmap → {cm_path}")

    # --- Raw matrix as labelled CSV (queryable, unlike the PNG) --------------
    # Row 0 / column 0 carry species names so the matrix can be inspected
    # or loaded directly for the per-species hard-mining analysis (#24).
    species_names = [idx_to_sp[i] for i in range(num_classes)]
    cm_csv = os.path.join(output_dir, "test_confusion_matrix.csv")
    with open(cm_csv, "w", encoding="utf-8", newline="") as f:
        f.write("true\\pred," + ",".join(species_names) + "\n")
        for i, row_name in enumerate(species_names):
            f.write(row_name + "," + ",".join(str(int(v)) for v in cm[i]) + "\n")
    logger.info(f"Confusion matrix CSV    → {cm_csv}")

    # --- Ranked confused pairs (the #24 hard-mining deliverable) -------------
    # Off-diagonal entries only — diagonal is correct predictions, not errors.
    # Each pair: how often a true species was misclassified as another, with
    # the error rate relative to that true species' test support.
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    row_support = cm.sum(axis=1)  # total test samples per true species

    pairs = []
    for ti in range(num_classes):
        for pi in range(num_classes):
            count = int(cm_offdiag[ti, pi])
            if count > 0:
                support = int(row_support[ti])
                pairs.append((
                    idx_to_sp[ti], idx_to_sp[pi], count,
                    count / support if support else 0.0,
                ))
    pairs.sort(key=lambda x: (-x[2], -x[3]))

    TOP_N = 25
    logger.info("=" * 60)
    logger.info(f"TOP {TOP_N} CONFUSED PAIRS (hard-mining targets — #24)")
    logger.info("=" * 60)
    logger.info(f"  {'true species':<32s} {'→ predicted as':<32s} {'n':>4s}  err%")
    for true_sp, pred_sp, count, rate in pairs[:TOP_N]:
        logger.info(f"  {true_sp:<32s} {pred_sp:<32s} {count:>4d}  {rate:.1%}")

    pairs_csv = os.path.join(output_dir, "test_confused_pairs.csv")
    with open(pairs_csv, "w", encoding="utf-8", newline="") as f:
        f.write("true_species,predicted_species,count,error_rate\n")
        for true_sp, pred_sp, count, rate in pairs:
            f.write(f"{true_sp},{pred_sp},{count},{rate:.4f}\n")
    logger.info("=" * 60)
    logger.info(f"All {len(pairs)} confused pairs → {pairs_csv}")

if __name__ == "__main__":
    main()
