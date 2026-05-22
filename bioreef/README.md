# 🐠 BioReef.ai

**Marine Biodiversity Monitoring System — Gulf of Oman**

A three-stage deep learning pipeline for high-precision species detection, tracking, and hierarchical classification in underwater reef environments.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BioReef.ai Pipeline                          │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐ │
│  │   Stage 1    │   │   Stage 2    │   │      Stage 3        │ │
│  │  Detection   │──▶│  Tracking    │──▶│  Classification     │ │
│  │              │   │              │   │                     │ │
│  │ DINOv2 +     │   │ ByteTrack +  │   │ ViT-LSTM +         │ │
│  │ MCEAM Fusion │   │ EMA Re-ID    │   │ HSLM (Taxonomy)    │ │
│  └──────────────┘   └──────────────┘   └─────────────────────┘ │
│                                                                 │
│  Evaluation: HOTA (Tracking) + HD (Hierarchical Distance)      │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: Context-Aware Spatial Detection
- **4-stream Context Harvester** — ROI (1x), Social (3x), Habitat (5x), Full Frame
- **DINOv2 ViT-B/14** — Frozen Vision Transformer backbone
- **MCEAM** — Multi-Head Cross-Attention fusing fish features with habitat context

### Stage 2: Hybrid Spatiotemporal Tracking
- **ByteTrack** — Dual-threshold association with low-confidence rescue
- **Kalman Filter + CMC** — Motion prediction with camera motion compensation
- **EMA Feature Bank** — DINOv2-based appearance signatures for Re-ID

### Stage 3: Hierarchical Behavioral Classification
- **ViT-LSTM** — Spatial (DINOv2) + Temporal (LSTM behavioral fingerprinting)
- **HSLM** — Family → Genus → Species multi-head classification
- **Taxonomic Guardrails** — Consistency masking to prevent biological leaps

---

## Project Structure

```
bioreef/
├── configs/
│   └── stage1.yaml              # Hydra/YAML configuration
├── data/
│   ├── __init__.py
│   └── data_factory.py          # Preprocessing: Water-Net + Context Harvester
├── models/
│   ├── __init__.py
│   ├── backbone.py              # DINOv2 ViT-B/14 wrapper
│   └── mceam.py                 # MCEAM cross-attention fusion
├── evaluation/
│   ├── __init__.py
│   ├── hota_evaluator.py        # HOTA tracking metric
│   └── hd_evaluator.py          # Hierarchical Distance metric
├── utils/
│   ├── __init__.py
│   └── taxonomy.py              # WoRMS API client + taxonomic tree
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Install dependencies
pip install -r bioreef/requirements.txt

# DINOv2 weights auto-download via PyTorch Hub
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
```

---

## Target Metrics

| Metric | Target | Benchmark |
|:---|:---|:---|
| Classification Precision | >95% | 98.28% |
| mAP (Detection) | >85% | 89.45% |
| HOTA (Tracking) | >70% | 74.20% |
| Hierarchical Distance (HD) | <2.0 | 1.54 |

---

## References

- Lee et al. (2026), *MATANet: Multi-Context Attention and Taxonomy-Aware Network*
- Oquab et al. (2023), *DINOv2: Learning Robust Visual Features without Supervision*
- Li et al. (2019), *An Underwater Image Enhancement Benchmark (Water-Net)*
- Luiten et al. (2021), *HOTA: A Higher Order Metric for Multi-Object Tracking*

---

**BioReef.ai** — *Accurate, Taxonomic, Ecologically Grounded.*
