# BioReef.ai — Metrics Log

Rolling log of training and evaluation metrics across the pipeline.
New results are appended at the top. Older entries kept for comparison.

---

## Stage 0 — YOLO Detector (Multi-Class, pre-swap)

**Configuration**
- Model: `yolo11m.pt` (Ultralytics YOLOv11-medium)
- Classes: 260 species (multi-class)
- Image size: 640
- Batch: 8, device: 0,1 (2× Quadro 4000, 8 GB)
- Training epochs: 98 (interrupted/resumed between epoch 95→96)

**Best checkpoint (epoch ~85–86)**
| Metric | Value |
|---|---|
| mAP@0.5 | **0.445** |
| mAP@0.5:0.95 | 0.384 |
| Precision (best) | ~0.52 |
| Recall (best) | ~0.45 |

**Validation run (`runs/detect/val/`)**
| Metric | Value |
|---|---|
| mAP@0.5 (all classes) | 0.425 |
| F1 peak | 0.36 @ conf=0.074 |
| Precision @ conf=1.0 | 0.96 |

**Notes**
- `val/cls_loss` diverged from epoch 50+ (1.81 → 2.35) — classification head overfitting on long-tail 260-class distribution.
- `val/box_loss` + `val/dfl_loss` remained stable — localization is sound.
- F1 peak at very low confidence (0.074) indicates the detector is under-confident.
- Decision: **retrain as class-agnostic single-class ("fish") detector.** Species ID is delegated to MCEAM downstream.

---

## Stage 1 — MCEAM Classifier (Species ID)

**Configuration**
- Backbone: DINOv3 ViT-B/16 (frozen, `facebook/dinov3-vitb16-pretrain-lvd1689m`)
- Head: MCEAM (4-stream cross-attention → 256-dim → Linear 260)
- Classes: 260 species (min_samples=20)
- Loss: CB-Focal (effective number weighting + focal modulation)
- LR schedule: Linear warmup (5 epochs) → Cosine decay
- Batch: 8, device: 0,1 (2× Quadro 4000, 8 GB)

**Best checkpoint (epoch 28/30)**
| Metric | Value |
|---|---|
| Top-1 accuracy | **57.86%** |
| Val mAP (macro) | 0.4756 |
| Val HD (Hierarchical Distance) | **1.1252** (target: 2.0) |
| Train loss | 1.0152 |
| Val loss | 1.0901 |

**Resource utilization**
- VRAM: 0.54 GB / 0.71 GB per card — heavily underused. Can likely bump batch to 32–64.

**Notes**
- HD=1.125 means most misclassifications stay within the correct genus (HD=1 = wrong species, right genus).
- Healthy train/val loss gap (no overfitting yet).
- Only 30 epochs run — cosine schedule barely decayed; more training would likely help.

---

## Update Protocol

New metrics go at the top of the relevant section with a date stamp.
Keep prior runs for comparison — do not delete history.
