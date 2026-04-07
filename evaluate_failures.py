"""
BioReef.ai — Stage 1 Failure Gallery Generator
==============================================
Evaluates the Validation Set using the latest best checkpoint, extracting the
top 5 samples with the highest Loss and top 5 with the highest Hierarchical Distance (HD).

Generates a visual 'Failure Gallery' grid allowing ecological analysis of what
confuses the model (e.g., murky water vs. true biological ambiguity).
"""

import os
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from collections import defaultdict

from bioreef.models.backbone import DINOv2Backbone
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester, WaterNetRestorer
from bioreef.evaluation.hd_evaluator import HDEvaluator
from train_stage1 import Stage1Dataset, split_dataset, get_taxonomy_tree

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("failure_gallery")

def draw_info_on_image(img_path, bbox, true_label, pred_label, metric_name, metric_val):
    # Load raw image
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((1080, 1920, 3), dtype=np.uint8)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    x, y, w, h = bbox
    # Draw Bounding Box (Red for failure)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 4)
    
    # Text Setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    lines = [
        f"True: {true_label}",
        f"Pred: {pred_label}",
        f"{metric_name}: {metric_val:.4f}"
    ]
    
    y0, dy = 50, 50
    for i, line in enumerate(lines):
        y_pos = y0 + i*dy
        # Draw shadow
        cv2.putText(img, line, (22, y_pos+2), font, font_scale, (0, 0, 0), thickness+1)
        # Draw white text
        cv2.putText(img, line, (20, y_pos), font, font_scale, (255, 255, 255), thickness)
        
    # Resize to something manageable for a grid, e.g., 640x360
    img_resized = cv2.resize(img, (640, 360))
    return img_resized

def save_gallery(images, filename, title):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {filename}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    data_dir = r"c:\Users\karoo\University\Junior\data\ozfish_data"
    json_path = os.path.join(data_dir, "annotation", "subset_final.json")
    img_dir = os.path.join(data_dir, "images")
    output_dir = r"c:\Users\karoo\University\Junior\outputs\visual_audit"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load exact same Val Split
    _, val_samples, _ = split_dataset(json_path, img_dir)
    logger.info(f"Validation set size: {len(val_samples)}")
    
    val_ds = Stage1Dataset(val_samples, img_dir, is_train=False)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    # 2. Load Models
    backbone = DINOv2Backbone(freeze=True).to(device)
    mceam = MCEAM(embed_dim=768, num_context_levels=3, output_dim=256, num_heads=8).to(device)
    head = nn.Linear(256, 3).to(device)
    
    checkpoint_path = r"c:\Users\karoo\University\Junior\bioreef_stage1_best.pt"
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
        
    logger.info("Loading best checkpoint weights...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    mceam.load_state_dict(ckpt['mceam'])
    head.load_state_dict(ckpt['head'])
    
    mceam.eval()
    head.eval()
    
    criterion = nn.CrossEntropyLoss(reduction='none') # Get per-sample loss
    hd_evaluator = HDEvaluator(taxonomy_tree=get_taxonomy_tree())
    idx_to_sp = {0: "SpiderFish", 1: "TulesOneFish", 2: "TulesTwoFish"}
    
    results = []
    
    logger.info("Evaluating Validation Set for Failures...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dl)):
            streams = {k: v.to(device) for k, v in batch['streams'].items()}
            labels_tensor = batch['label'].to(device)
            
            features = backbone(streams)
            out = mceam(features)
            preds = head(out['embedding'])
            
            # Loss
            loss = criterion(preds, labels_tensor).item()
            
            # HD
            pred_idx = preds.argmax(dim=1).item()
            true_idx = labels_tensor.item()
            
            pred_sp = idx_to_sp[pred_idx]
            true_sp = batch['species'][0]
            
            hd_score, _ = hd_evaluator.compute_distance(pred_sp, true_sp)
            
            results.append({
                'idx': i,
                'img_path': val_samples[i]['img_path'],
                'bbox': val_samples[i]['bbox'],
                'true_sp': true_sp,
                'pred_sp': pred_sp,
                'loss': loss,
                'hd': hd_score
            })
            
    # --- Highest Loss ---
    results_by_loss = sorted(results, key=lambda x: x['loss'], reverse=True)
    top_5_loss = results_by_loss[:5]
    
    # --- Highest HD ---
    # Secondary sort by loss so ties in HD are broken by lowest confidence
    results_by_hd = sorted(results, key=lambda x: (x['hd'], x['loss']), reverse=True)
    top_5_hd = results_by_hd[:5]
    
    logger.info("\n=== Top 5 Validation Losses ===")
    loss_imgs = []
    for r in top_5_loss:
        logger.info(f"Loss: {r['loss']:.4f} | HD: {r['hd']} | True: {r['true_sp']} | Pred: {r['pred_sp']}")
        img = draw_info_on_image(r['img_path'], r['bbox'], r['true_sp'], r['pred_sp'], "Loss", r['loss'])
        loss_imgs.append(img)
        
    save_gallery(
        loss_imgs, 
        os.path.join(output_dir, "failure_gallery_loss.png"), 
        f"Failure Gallery: Highest Validation Loss (Model Uncertainty)"
    )
    
    logger.info("\n=== Top 5 Hierarchical Distances ===")
    hd_imgs = []
    for r in top_5_hd:
        logger.info(f"HD: {r['hd']} | Loss: {r['loss']:.4f} | True: {r['true_sp']} | Pred: {r['pred_sp']}")
        img = draw_info_on_image(r['img_path'], r['bbox'], r['true_sp'], r['pred_sp'], "HD", r['hd'])
        hd_imgs.append(img)
        
    save_gallery(
        hd_imgs, 
        os.path.join(output_dir, "failure_gallery_hd.png"), 
        f"Failure Gallery: Highest Hierarchical Distance (Taxonomic Error)"
    )

if __name__ == "__main__":
    main()
