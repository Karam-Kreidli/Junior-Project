# Run Manifest

The full training campaign. Each config lives in `configs/runs/` and is run with
`python scripts/run.py <id> --seed <N>` (single-GPU). Every paper config runs at
seeds {0,1,2}; the final table is `RESULTS.md` (built by `scripts/aggregate.py`).

**Proposed model = A15** (unfrozen DINOv3 + MCEAM + HSLM + plain CE). C09 is the
**frozen baseline** and the start of the adaptation-depth frontier, not the
proposed model.

Status legend: `[ ]` not started · `[~]` partial · `[x]` all 3 seeds done

## Benchmark panel — baselines + references

| Status | Run | Model | Family | Backbone | Context | Hierarchy | Loss + Sampler |
|---|---|---|---|---|---|---|---|
| [x] | C01 | linear_probe | dino | dinov3 frozen | none | flat | CE + random |
| [x] | C03 | resnet50 | timm | resnet50 (FT) | n/a | flat | CE + random |
| [x] | C04 | efficientnetv2_s | timm | effnetv2-s (FT) | n/a | flat | CE + random |
| [x] | C05 | convnext_tiny | timm | convnext-t (FT) | n/a | flat | CE + random |
| [x] | C06 | vit_base | timm | vit-b/16 (FT) | n/a | flat | CE + random |
| [x] | C07 | swin_base | timm | swin-b (FT) | n/a | flat | CE + random |
| [~] | C08 | matanet | matanet | DINOv2-large (FT, their repo) | native | per-level | native |
| [x] | C09 | proposed (FROZEN REF) | dino | dinov3 frozen | MCEAM 3sc/1blk | HSLM | CB-Focal + random |

C08 = MATANet, run from the official repo on our split (see `matanet/README.md`).
**All 3 C08 seeds complete (2026-08-23)** — the panel is COMPLETE at 24 configs ×
3 seeds = 74 runs. C08 3-seed mean: top1 0.844±.002, macro 0.692±.012, HD
0.288±.008 — a **dead tie with A15 on every headline metric** at **13.7× the
parameters** (1,347,468,072 vs 98,157,378 total, both fully trainable; counted
exactly with `matanet/count_matanet_params.py`). That tie is the paper's
parameter-efficiency headline.

## Proposed model + ablations

Config-only ablations branch off the **frozen reference C09** (one field changed).
The unfrozen loss chain (A13/A14/A15/A12) and the adaptation-depth frontier
(C09→A9→A10→A11) build toward the proposed unfrozen model **A15**.

| Status | Run | One-factor change / role | Priority |
|---|---|---|---|
| [x] | A1 | C09 backbone DINOv3 → frozen DINOv2-base (frozen backbone-generation comparison) | core |
| [x] | A2 | context off: MCEAM removed (head on pooled ROI) | core |
| [x] | A3 | single context stream (social only) vs all three | core |
| [x] | A4 | attention depth 1 → 2 blocks | core |
| [x] | A5 | attention depth 1 → 4 blocks | optional |
| [x] | A6 | hierarchy off: HSLM → flat softmax (species-only), frozen | core |
| [x] | A7 | sampler random → balanced (frozen) | core |
| [x] | A8 | all long-tail handling off: (CB-Focal) → plain CE, frozen | core |
| [x] | A9 | frozen → last 2 blocks unfrozen (depth frontier) | core |
| [x] | A10 | frozen → last 4 blocks unfrozen | optional |
| [x] | A11 | frozen → FULL fine-tune (all blocks + embeddings), lr 1e-4 | optional |
| [x] | A12 | A11 with lr 1e-4 → 1e-5 (HSLM + CB-Focal, unfrozen) | optional |
| [x] | A13 | unfrozen HSLM off: flat CB-Focal | core |
| [x] | A14 | unfrozen flat plain CE (loss-chain endpoint) | core |
| [x] | **A15** | **PROPOSED: unfrozen HSLM + plain CE (lr 1e-5)** | core |
| [x] | A16 | A15 backbone DINOv3 → DINOv2-base (unfrozen backbone comparison) | optional |

## Deployment (NOT paper-benchmark runs)

All D runs are **1 seed** unless noted. None are paper-benchmark runs.

| Status | Run | Role | Priority |
|---|---|---|---|
| [~] | D1 | deployment config, CB-Focal (Junior 35-species transfer) | optional |
| [~] | D2 | deployment config, plain CE | optional |
| [~] | D3 | A15 recipe on DINOv3 **ViT-L/16**, lr 1e-5 (inherited) — **failed**, lr too hot | optional |
| [~] | D3a | D3 + lr **5e-6** — the ViT-L winner | optional |
| [~] | D3b | D3 + lr 2e-6 — too cold | optional |
| [~] | D4 | A15 + layer-wise LR decay (LLRD) | optional |
| [ ] | D5 | ViT-L + LLRD | optional |
| [~] | D6 | A15 recipe on DINOv3 **ViT-H+**, lr 5e-6 — **best overall** | optional |
| [~] | D7 | ViT-L + CB-Focal (CB-Focal twin of D3a) — **best tail** | optional |
| [ ] | D8 | ViT-H+ + CB-Focal @5e-6 — never run, **moot** (CB-Focal lost at 2.5e-6) | optional |
| [~] | D6a | D6 + lr 2.5e-6 — **FINAL DEPLOYMENT MODEL** (seeds 0 and 1) | optional |
| [~] | D8a | D8 + lr 2.5e-6 (CB-Focal branch) — lost to D6a | optional |

**Deployment ladder results (1 seed each):**

| Run | Backbone | macro | top1 | HD ↓ | MistSev ↓ | tail |
|---|---|---|---|---|---|---|
| A15 | ViT-B (86 M) | 0.690 | 0.842 | 0.291 | 1.844 | 0.444 |
| D3a | ViT-L @5e-6 (300 M) | 0.727 | 0.859 | 0.242 | 1.712 | 0.524 |
| D7 | ViT-L + CB-Focal | 0.729 | 0.857 | 0.247 | 1.724 | **0.578** |
| D6 | ViT-H+ @5e-6 (840 M) | 0.746 | 0.865 | 0.221 | **1.643** | 0.565 |
| D8a | ViT-H+ @2.5e-6 + CB-Focal | 0.733 | 0.865 | 0.222 | 1.652 | 0.556 |
| **D6a** | **ViT-H+ @2.5e-6 (seed 1)** | **0.749** | **0.876** | **0.213** | 1.715 | **0.580** |
| **D6a + logit-adjust tau=1.0** | **same weights, post-hoc** | **0.778** | 0.871 | 0.213 | 1.653 | **0.672** |

**Scaling verdict (2026-09-12): saturated.** 86 M → 300 M → 840 M moved top-1
0.842 → 0.859 → 0.865 — ~10× the parameters for **+0.023**. The ceiling is the
data (~49 k train crops, 321 classes, long tail) and the architecture, not
backbone capacity. **No ViT-7B rung.**

**FINAL DEPLOYMENT MODEL (2026-09-20): D6a seed 1 + logit-adjust tau=1.0.**
DINOv3 ViT-H+, plain CE, lr 2.5e-6, full fine-tune. The post-hoc logit adjustment
(subtract `1.0 * log(prior_c)` from each logit before argmax — no retraining, head
weights unchanged) buys **tail +0.092** (.580 → .672), macro +0.029 and mistSev
−0.062 for −0.005 top1. CB-Focal's tail advantage **inverts** with backbone size:
it won at ViT-L (D7 .578 vs D3a .524) and lost at ViT-H+ (D8a .556 vs D6a .573,
matched seed 0).

> ⚠️ **The D ranking is NOT paper-citable.** Every D row is 1–2 seeds, versus the
> 3-seed A/C panel, and gaps sit inside the panel's tail std (±.02 to ±.11).
> The post-hoc tau was additionally selected on the TEST split (`dump_posthoc.py`
> dumps test only), so it is a deployment operating point, **not** a clean
> held-out result. The A/C panel rows above are 3-seed and unaffected.

D runs are the deployment through-line, not part of the benchmark table.
`aggregate.py --campaign configs/campaign.yaml` excludes them from the paper
table (AUDIT.md #14).

## Notes

- **C08 MATANet** runs from the official repo, not `run.py` — see
  `configs/runs/C08_matanet.yaml` and `matanet/README.md`.
- The attention-mass / context-stream analysis is computed from A15's test
  inferences — no separate run.
- Every result JSON records its resolved config + seed; see AUDIT.md for the
  provenance/checksum work still pending before release.
- **Public release:** the benchmark shipped as a standalone repo on 2026-08-30
  (`ozfish-finegrained-staging`) — README + RESULTS + splits + configs, MIT, one
  clean commit per the fresh-repo decision (AUDIT.md anchoring decision #3).
