# BioReef.ai — Implementation Tasks

> **Project:** Marine Biodiversity Monitoring System — Gulf of Oman
> **Architecture:** 3-Stage Pipeline (Detection → Tracking → Classification)
> **Rules:** All implementations follow `.agent/rules.md` guardrails

---

## Phase 1: Stage 1 — Context-Aware Spatial Detection

### Task 1.1: Data Factory (`bioreef/data/data_factory.py`)
- [ ] **Water-Net Restoration** — Integrate pretrained Water-Net for spectral recovery (white balance, gamma, local enhancement)
- [ ] **Context Harvester** — Implement 4-stream concentric cropping (1x ROI, 3x Social, 5x Habitat, Full Frame)
- [ ] **Size-Adaptive ROI** — High-res initial crop (512×512) for small fish (<5% frame area) before downsampling
- [ ] **Taxonomic Parser** — Hierarchical label engineering (Species → Genus → Family) with ambiguity filtering
- [ ] **Normalization** — Bicubic resize to 224×224, aspect-ratio-preserving letterboxing, ImageNet Z-score normalization
- [ ] **Marine Augmentor** — Domain-specific augmentations (turbidity noise, marine snow, motion blur, photometric jitter)

### Task 1.2: MCEAM Module (`bioreef/models/mceam.py`)
- [ ] **DINOv2 Backbone Wrapper** — Frozen ViT-B/14, extract [CLS] tokens + patch embeddings per stream
- [ ] **Multi-Head Cross-Attention** — ROI features as Query, context patches as Key/Value
- [ ] **Context Fusion** — Concatenate attended features from all 3 context levels with ROI embedding
- [ ] **CARAFE Upsampling** — Content-aware feature reassembly for small-object detection refinement

### Task 1.3: Evaluation Logging (`bioreef/evaluation/`)
- [ ] **HOTA Evaluator** — Detection Accuracy (DetA) + Association Accuracy (AssA) balanced metric; JSON logging
- [ ] **HD Evaluator** — Hierarchical Distance computation using taxonomic tree; weighted penalty by taxonomic level
- [ ] **WoRMS Validator** — REST API client for cross-referencing species metadata against World Register of Marine Species

---

## Phase 2: Stage 2 — Hybrid Spatiotemporal Tracking *(future)*
- [ ] ByteTrack + Dual-Threshold Association
- [ ] Kalman Filter with Camera Motion Compensation (CMC)
- [ ] EMA-based appearance feature bank for Re-ID
- [ ] Spatiotemporal tracklet generation (16–30 frames)

## Phase 3: Stage 3 — Hierarchical Classification *(future)*
- [ ] Time-Distributed DINOv2 for per-frame spatial features
- [ ] LSTM for temporal behavioral fingerprinting
- [ ] HSLM multi-head classifier (Family → Genus → Species)
- [ ] Taxonomic Guardrails (Consistency Mask) for inference
