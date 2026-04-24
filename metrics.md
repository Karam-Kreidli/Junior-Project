# BioReef.ai — Metrics Log

Rolling log of training and evaluation metrics across the pipeline.
New results are appended at the top. Older entries kept for comparison.

---

## Stage 0 — Detector Retrain on GDINO-Cleaned Labels

### 2026-04-24 — YOLOv11m on cleaned train (44k → 455k labels) — IN PROGRESS

**Configuration**
- Model: YOLOv11m, single-class ("fish")
- Dataset: `datasets/ozfish_cleaned/` (train + val/test pseudo-labeled via Grounding DINO, threshold=0.30 for val/test)
- imgsz=640, batch=24, device=0,1
- Epochs: 100, patience=15
- Augmentation: close_mosaic=0, copy_paste=0.3, label_smoothing=0.05

**Best so far (epoch 25–33 plateau)**
| Metric | Value | Δ vs 2026-04-20 single-class (pre-cleaning) |
|---|---|---|
| mAP@0.5 | **0.845** | +0.42 (≈2×) |
| mAP@0.5:0.95 | **0.700** | +0.33 (≈2×) |
| Precision | 0.78 | +0.37 |
| Recall | 0.79 | +0.32 |

**Notes**
- Label cleaning was the true bottleneck. With ~10× more labels, the detector generalizes properly and confidence scores are calibrated.
- Plateau emerging ~epoch 25 — probably near convergence.
- Training loss: box=0.75, cls=0.67, dfl=0.93.
- Val has 4232 images with 42,927 cleaned labels (vs ~5,684 in the noisy val).

---

## End-to-End Pipeline Evaluation

### 2026-04-23 — Detector (single-class @ imgsz=960) + MCEAM (epoch 65 of in-progress run)

**Setup**
- Val split: 5684 labeled fish across unique frames (same deterministic split as training)
- Detector: YOLOv11m single-class, best.pt at ~epoch 67 (mAP50=0.423)
- Classifier: `bioreef_stage1.pt` (epoch 65, Top-1=62.80% on GT crops)
- Match: IoU ≥ 0.5, greedy by detection confidence

**Confidence sweep**
| conf | recall | top1_matched | top5_matched | HD | **e2e_top1** |
|---|---|---|---|---|---|
| 0.05 | **87.21%** | 63.95% | 84.85% | **0.9714** | **55.77%** |
| 0.10 | 78.57% | 63.64% | 84.55% | 0.9830 | 50.00% |
| 0.25 | 55.15% | 61.69% | 82.81% | 1.0376 | 34.03% |
| 0.50 | 29.96% | 61.13% | 81.86% | 1.0552 | 18.31% |

**Key findings**
- End-to-end Top-1 peaks at **conf=0.05** (55.77%) — 3× better than the mAP-optimal conf=0.5 (18.31%).
- Classifier accuracy on matched crops is stable ~61–64% regardless of detector confidence.
- Low conf does **not** degrade the classifier — HD at conf=0.05 (0.9714) is actually *better* than standalone classifier on GT boxes (1.0072).
- **Recommendation:** run production inference at `conf=0.05`, filter false positives downstream with a classifier-confidence threshold and track-level aggregation.
- Detector was never the bottleneck it appeared to be — tuning for recall instead of mAP nearly doubles end-to-end accuracy.

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

### 2026-04-22 — Run with batch=32, EMA, 100ep schedule (in progress)

**Configuration**
- Backbone: DINOv3 ViT-B/16 (frozen)
- Head: MCEAM (4-stream cross-attention → 256-dim → Linear 260)
- Classes: 260 species (min_samples=20)
- Loss: CB-Focal
- LR schedule: Linear warmup (5 epochs) → Cosine decay
- Batch: 32 per rank × 2 = 64 effective
- EMA: decay=0.999 on MCEAM + head; validation + best checkpoint use EMA weights
- Target: 100 epochs

**Best so far (epoch 85/100)**
| Metric | Value | Δ vs 2026-04-18 best |
|---|---|---|
| Top-1 accuracy | **64.00%** | +6.14 pp |
| Top-5 accuracy | **85.07%** | (new — not logged before) |
| Val mAP (macro) | 0.5702 (ep 86) | +0.095 |
| Val HD | **0.9685** | −0.157 (lower is better) |
| Train loss | 0.4328 | |
| Val loss | 0.9346 (min 0.9319 @ ep 87) | |

**Progression**
| Epoch | HD | Top-1 | Top-5 | mAP | Val |
|---|---|---|---|---|---|
| 41 | 1.0995 | 59.46 | 82.11 | 0.512 | 1.015 |
| 50 | 1.0415 | 61.56 | 83.59 | 0.542 | 0.965 |
| 60 | 1.0158 | 62.51 | 84.10 | 0.553 | 0.956 |
| 65 | 1.0072 | 62.80 | 84.21 | 0.554 | 0.947 |
| 82 | 0.9762 | 63.73 | 85.07 | 0.568 | 0.937 |
| 85 | 0.9685 | 64.00 | 85.07 | 0.570 | 0.935 |
| 88 | 0.9748 | 63.82 | 85.19 | 0.569 | 0.934 |

**Notes**
- **HD broke below 1.0 at epoch 82** — average wrong prediction now stays within the correct genus.
- Plateaued from epoch 85 — cosine tail is converging. Val loss still near minimum (0.932).
- Final Top-1 expected to settle near 64.0–64.3%.
- 12 epochs remaining.

---

### 2026-04-18 — Original run (batch=8, 30ep)

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
