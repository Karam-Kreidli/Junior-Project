"""
BioReef.ai — Classifier Inference & Visualization
===================================================
Runs the test split through the trained Stage 1 classifier and saves:
  - results/correct_predictions.png   : grid of confident correct predictions
  - results/near_misses.png           : grid where top-1 wrong but top-5 correct
  - results/attention_maps.png        : attention heatmaps per context stream

Usage:
    python visualize_classifier.py
    python visualize_classifier.py --checkpoint bioreef_stage1.pt --n_correct 16 --n_miss 16
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from train_stage1 import split_dataset, Stage1Dataset, safe_imread
from bioreef.data.data_factory import ContextHarvester

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="bioreef_stage1.pt")
    p.add_argument("--csv_path",    default="data_oz/metadata/frame_metadata.csv")
    p.add_argument("--img_dir",     default="data_oz/frames_waternet_1")
    p.add_argument("--min_samples", type=int, default=20)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--n_correct",   type=int, default=16, help="Correct predictions to show")
    p.add_argument("--n_miss",      type=int, default=16, help="Near-misses to show")
    p.add_argument("--n_attn",      type=int, default=4,  help="Attention map examples")
    p.add_argument("--out_dir",     default="results")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, num_classes, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Read class count from checkpoint to handle train/eval split mismatches
    ckpt_num_classes = ckpt["head"]["weight"].shape[0]
    if ckpt_num_classes != num_classes:
        print(f"[!] Class count mismatch: checkpoint={ckpt_num_classes}, split={num_classes}. Using checkpoint value.")
        num_classes = ckpt_num_classes

    backbone = ViTBackbone(freeze=True).to(device)
    mceam    = MCEAM(embed_dim=768, num_context_levels=3, output_dim=256, num_heads=8).to(device)
    head     = nn.Linear(256, num_classes).to(device)

    mceam.load_state_dict(ckpt["mceam"])
    head.load_state_dict(ckpt["head"])

    backbone.eval()
    mceam.eval()
    head.eval()
    return backbone, mceam, head

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(backbone, mceam, head, dataloader, device):
    records = []  # list of dicts per sample

    for batch in tqdm(dataloader, desc="Inference"):
        streams = {k: v.to(device) for k, v in batch["streams"].items()}
        labels  = batch["label"].to(device)

        with torch.amp.autocast("cuda"):
            features = backbone(streams)
            out      = mceam(features)
            logits   = head(out["embedding"])

        probs   = torch.softmax(logits, dim=1).cpu().numpy()
        top5    = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
        top1    = top5[:, 0]
        labels_np = labels.cpu().numpy()

        for i in range(len(labels_np)):
            records.append({
                "true_idx":    int(labels_np[i]),
                "true_name":   batch["species"][i],
                "top1_idx":    int(top1[i]),
                "top5_idx":    top5[i].tolist(),
                "top1_conf":   float(probs[i, top1[i]]),
                "img_path":    batch["img_path"][i],
                "bbox":        batch["bbox"][i],
            })

    return records

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def get_roi_crop(img_path, bbox, size=224):
    """Returns a letterboxed ROI crop as an RGB numpy array."""
    img = safe_imread(img_path)
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    x, y, w, h = [int(v) for v in bbox]
    crop = img[max(0,y):y+h, max(0,x):x+w]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    crop = cv2.resize(crop, (size, size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def save_grid(records, idx_to_sp, title, out_path, n=16, mode="correct"):
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=14, y=1.01)

    for ax in axes:
        ax.axis("off")

    count = 0
    for r in records:
        if count >= n:
            break

        correct = r["top1_idx"] == r["true_idx"]
        in_top5 = r["true_idx"] in r["top5_idx"]

        if mode == "correct" and not correct:
            continue
        if mode == "near_miss" and (correct or not in_top5):
            continue

        crop = get_roi_crop(r["img_path"], r["bbox"])
        ax = axes[count]
        ax.imshow(crop)

        pred_name = idx_to_sp[r["top1_idx"]].replace("_", " ")
        true_name = r["true_name"].replace("_", " ")

        if mode == "correct":
            label = f"{pred_name}\n{r['top1_conf']*100:.1f}%"
            color = "green"
        else:
            label = f"Pred: {pred_name}\nTrue: {true_name}"
            color = "darkorange"

        ax.set_title(label, fontsize=7, color=color, pad=3)
        count += 1

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_attention_maps(records, backbone, mceam, device, idx_to_sp, out_path, n=4):
    harvester = ContextHarvester(target_resolution=224, small_object_threshold=0.05)
    stream_names = ["social", "habitat", "full_frame"]
    stream_labels = ["3× Social", "5× Habitat", "Full Frame"]

    fig, axes = plt.subplots(n, 4, figsize=(14, n * 3.5))
    fig.suptitle("Cross-Attention Maps per Context Stream", fontsize=13)

    col_labels = ["ROI Crop"] + stream_labels
    for col, lbl in enumerate(col_labels):
        axes[0, col].set_title(lbl, fontsize=10, fontweight="bold")

    shown = 0
    for r in records:
        if shown >= n:
            break
        if r["top1_idx"] != r["true_idx"]:
            continue

        img = safe_imread(r["img_path"])
        if img is None:
            continue

        streams = harvester.harvest(img, [int(v) for v in r["bbox"]])
        streams_b = {k: v.unsqueeze(0).to(device) for k, v in streams.items()}

        with torch.no_grad(), torch.amp.autocast("cuda"):
            features = backbone(streams_b)
            out = mceam(features, return_attention=True)

        # ROI crop
        roi_np = streams["roi"].permute(1, 2, 0).cpu().numpy()
        roi_np = (roi_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        roi_np = np.clip(roi_np, 0, 1)
        axes[shown, 0].imshow(roi_np)
        axes[shown, 0].set_ylabel(r["true_name"].replace("_", " "), fontsize=7)
        axes[shown, 0].axis("off")

        attentions = out.get("attentions", {})
        for col, (sname, slabel) in enumerate(zip(stream_names, stream_labels), start=1):
            context_np = streams[sname].permute(1, 2, 0).cpu().numpy()
            context_np = (context_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            context_np = np.clip(context_np, 0, 1)

            axes[shown, col].imshow(context_np, alpha=0.6)

            if sname in attentions:
                # attn shape: (B, heads, 1, N) — average across heads, reshape to 14×14
                attn = attentions[sname][0].mean(0).squeeze(0).cpu().float().numpy()  # (196,)
                attn = attn.reshape(14, 14)
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
                attn_up = cv2.resize(attn, (224, 224))
                axes[shown, col].imshow(attn_up, cmap="jet", alpha=0.45)

            axes[shown, col].axis("off")

        shown += 1

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset split...")
    _, _, test_samples, num_classes, idx_to_sp, _ = split_dataset(
        args.csv_path, args.img_dir, min_samples=args.min_samples
    )
    print(f"Test samples: {len(test_samples)} | Classes: {num_classes}")

    print("Loading model...")
    backbone, mceam, head = load_model(args.checkpoint, num_classes, device)

    # After load_model may have adjusted num_classes to match checkpoint
    ckpt_classes = head.out_features
    if ckpt_classes != len(idx_to_sp):
        # Pad idx_to_sp with placeholders for any extra checkpoint classes
        for i in range(len(idx_to_sp), ckpt_classes):
            idx_to_sp[i] = f"unknown_{i}"

    # Dataset — no augmentation, no WaterNet (already pre-applied to frames)
    test_ds = Stage1Dataset(test_samples, args.img_dir, is_train=False, use_waternet=False)

    # Expose img_path and bbox in batch via custom collate
    def collate(batch):
        return {
            "streams":  {k: torch.stack([b["streams"][k] for b in batch]) for k in batch[0]["streams"]},
            "label":    torch.tensor([b["label"] for b in batch]),
            "species":  [b["species"] for b in batch],
            "img_path": [b["img_path"] for b in batch],
            "bbox":     [b["bbox"] for b in batch],
        }

    # Patch Stage1Dataset to expose img_path and bbox
    orig_getitem = test_ds.__getitem__
    def patched_getitem(idx):
        s = test_ds.samples[idx]
        item = orig_getitem(idx)
        item["img_path"] = s["img_path"]
        item["bbox"] = s["bbox"]
        return item
    test_ds.__getitem__ = patched_getitem

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate)

    print("Running inference...")
    records = run_inference(backbone, mceam, head, loader, device)

    # Metrics
    top1 = sum(r["top1_idx"] == r["true_idx"] for r in records) / len(records)
    top5 = sum(r["true_idx"] in r["top5_idx"] for r in records) / len(records)
    print(f"\nTest Top-1: {top1*100:.2f}%  |  Top-5: {top5*100:.2f}%")

    # Sort by confidence for best visuals
    correct  = sorted([r for r in records if r["top1_idx"] == r["true_idx"]],
                      key=lambda r: -r["top1_conf"])
    near_miss = [r for r in records
                 if r["top1_idx"] != r["true_idx"] and r["true_idx"] in r["top5_idx"]]

    print(f"Correct: {len(correct)} | Near-misses: {len(near_miss)}")

    save_grid(correct,   idx_to_sp, "Correct Predictions (highest confidence)",
              os.path.join(args.out_dir, "correct_predictions.png"),
              n=args.n_correct, mode="correct")

    save_grid(near_miss, idx_to_sp, "Near-Misses (Top-1 wrong, Top-5 correct)",
              os.path.join(args.out_dir, "near_misses.png"),
              n=args.n_miss, mode="near_miss")

    save_attention_maps(correct, backbone, mceam, device, idx_to_sp,
                        os.path.join(args.out_dir, "attention_maps.png"),
                        n=args.n_attn)

    print("\nDone. Outputs saved to:", args.out_dir)


if __name__ == "__main__":
    main()
