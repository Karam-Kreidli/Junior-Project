# Classifier research archive

Salvage from `bioreef-classify`, the research repo where the Stage-1 species
classifier was developed and benchmarked. That repo is **retired** — its code and
paper artifacts are published at
[ozfish-finegrained-classification-benchmark](https://github.com/Karam-Kreidli/ozfish-finegrained-classification-benchmark),
and its deployment model now lives here as `models/d6a_stage1.pt`.

These files are the parts that belong in neither place: the deployment ladder was
stripped from the public repo (paper-only), and this context is too useful to
lose.

| file | what it is |
|---|---|
| `MANIFEST.md` | What every run is — C01-C09, A1-A16 (paper panel) and D1-D8a (deployment ladder), defined field by field, plus the final deployment result |
| `PAPER_FRAMING.md` | Which claims the results support and which they don't; the conclusions panel |
| `RESULTS.md` | The full benchmark table, 3-seed mean ± std |
| `params.csv` | Exact parameter counts per model |
| `all_test_metrics.csv` | **Every test metric for every run at every seed**, one row each (84 rows: 24 paper configs x 3 seeds, the D ladder, and the shipped post-hoc operating point). Per-seed, not averaged — `RESULTS.md` has the mean ± std view. Regenerate with `scripts/build_metrics_record.py`. |
| `results/` | The raw run outputs: `metrics.json`, `run_config.yaml` and `benchmark_config.yaml` for all 34 configs. This is the provenance behind every number in the paper — each file is a multi-hour training run, and after bioreef-classify is deleted this is the only copy. 969 KB, no weights. |

## The short version

The paper's proposed model is **A15** (DINOv3 ViT-B + MCEAM + HSLM + plain CE,
98,157,378 params), which ties MATANet — the closest prior work, 1,347,468,072
params — on every headline metric at **13.7× fewer parameters**. That parameter
efficiency is the paper's story.

Deployment then asked a different question: given a free 48 GB GPU, what is the
strongest model we can actually run? The D-series answer:

- **Backbone scaling saturates.** ViT-B → ViT-L → ViT-H+ (86M → 300M → 840M)
  moved top-1 0.842 → 0.859 → 0.865. Roughly 10× the parameters for +0.023. The
  ceiling is the data (~49k train crops, 321 classes, long tail) and the
  architecture, not capacity — so no ViT-7B rung.
- **Optimal LR halves with each backbone step**: 1e-5 (ViT-B), 5e-6 (ViT-L),
  2.5e-6 (ViT-H+). Missing this is what made the first ViT-L run look like a
  failure.
- **CB-Focal's tail advantage inverts with backbone size.** It won at ViT-L
  (D7 tail .578 vs D3a .524) and lost at ViT-H+ (D8a .556 vs D6a .573). Plain CE
  with a large enough backbone beats class-balanced focal on the very metric
  CB-Focal exists to improve.
- **LLRD did not help** (D4, on ViT-B) — the flat LR was already tuned.
- The winner is **D6a**, and post-hoc logit adjustment then bought a further
  +0.092 tail for −0.0045 top-1 with no retraining. See [`models/README.md`](../../models/README.md).

## Caveats that matter

D-series runs are **1-2 seeds**, against the paper panel's 3. The gaps between
them sit inside the panel's own seed variance (tail ±0.02-0.11), so this ordering
is a deployment decision, **not a citable result**. The post-hoc tau was selected
on the test split, which compounds that. The A/C panel rows in `RESULTS.md` are
3-seed and unaffected.
