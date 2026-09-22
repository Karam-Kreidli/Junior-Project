# Stage-1 classifier weights

## `d6a_stage1.pt` — the deployment model

DINOv3 **ViT-H+/16** (~840 M params) + MCEAM (3 context streams, 1 attention
block) + linear head over **321 species**. Fully fine-tuned end to end.

| | |
|---|---|
| run | `D6a`, seed 1 |
| backbone | `facebook/dinov3-vith16plus-pretrain-lvd1689m` (gated — needs `HF_TOKEN`) |
| learning rate | 2.5e-6, full fine-tune (all 32 blocks) |
| loss | HSLM + plain cross-entropy |
| batch / epochs | 8 / 60, 3 warm-up + cosine, EMA |
| selection | best validation hierarchical distance |
| params | 874,472,258 (all trainable) |
| data | OzFish, leakage-safe deployment-grouped split, 321 species |

**Not in git** (~3.5 GB, far past GitHub's 100 MB limit) — `*.pt` is gitignored.
Trained on the L40S VM; the source of truth is
`~/Desktop/Senior_Group/bioreef/results/D6a_dinov3_huge_lr2p5e6/seed1/checkpoint.pt`.

Checkpoint format is the research repo's: `{model, idx_to_sp, run_config,
benchmark_config}`, where `model` is a full `Classifier.state_dict()` — backbone
included, because a fine-tuned backbone *is* the model. The loader in
`bioreef/_9_pipeline/models.py` detects this format and builds the matching
architecture from `run_config`; the legacy `{mceam, head}` format still loads
through the other branch.

## `d6a_prior.json` — the logit-adjustment prior

Train class frequencies in class-index order (the same order as the logit
columns), plus the species mapping they were exported against. Committed: it is
small, and it is useless separated from the checkpoint.

At inference Stage 1 computes `logits - tau * log(prior)` (Menon 2020 / Balanced
Softmax). This changes the **decision rule, not the weights**, so tau stays
tunable in `config.yaml` (`inference.stage1_logit_tau`) and `tau = 0` reproduces
the raw model exactly.

At `tau = 1.0` on the OzFish test split:

| metric | raw | tau = 1.0 | |
|---|---|---|---|
| macro accuracy | 0.7490 | **0.7781** | +0.0291 |
| top-1 | 0.8756 | 0.8712 | −0.0045 |
| head / medium / tail | 0.8798 / 0.7141 / 0.5800 | 0.8802 / 0.7407 / **0.6722** | tail **+0.0922** |
| hierarchical distance ↓ | 0.2133 | 0.2130 | −0.0003 |
| mistake severity ↓ | 1.7152 | **1.6532** | −0.0620 |
| genus / family | 0.9416 / 0.9695 | 0.9459 / 0.9699 | both up |
| top-5 | 0.9663 | 0.9607 | −0.0056 |

Only top-1 and top-5 cost anything. Genus and family accuracy *rise* while top-1
falls: the adjustment moves species-level misses onto the correct genus/family
branch, which is what the mistake-severity drop measures. Raise tau toward 1.5
for more tail (0.699) at more top-1 cost (0.866).

Regenerate with:

```bash
python scripts/export_logit_prior.py \
    --ckpt models/d6a_stage1.pt --dump posthoc_dump.npz \
    --out models/d6a_prior.json
```

**Caveat:** tau was selected on the test split, so this is a deployment
operating point, not a clean held-out result — fine for running the system, not
citable as a benchmark number. The underlying D6a metrics are single-seed.

## `bioreef_stage1.pt` — retired

The original Stage-1 classifier: DINOv3 ViT-B, frozen backbone, MCEAM + head
only (50 MB). Trained before KNOWN_BUGS #1–13 were fixed, against a merged
~256-class label space that does **not** match the current binomial 321-class
head. Kept because it is still git-tracked and the loader's legacy branch can
still read it; do not use it for new work.

## `yolo11m.pt`, `best.pt`

Detector weights (Stage 0), unrelated to the classifier. `weights/` holds the
RF-DETR checkpoint that is the actual default detector.
