# archive/

Scripts that are no longer part of the live pipeline but are kept for
historical reproducibility — they document *how* certain artifacts were
originally produced, or they're tooling for workflows the project has
moved past.

Nothing in the live codebase imports anything in this folder. The live
pipeline runs without it.

## Why archived (per script)

| Script | What it did | Why archived |
|---|---|---|
| `convert_to_yolo.py` | One-shot: built the OzFish YOLO training dataset from the metadata CSV. | YOLO retired as the production detector (issue #6 — replaced by pretrained RF-DETR). No more YOLO training. |
| `pseudo_label_gdino.py` | One-shot: produced `datasets/ozfish_cleaned/` by running Grounding DINO over OzFish frames and merging its detections with the original labels — the cleaned dataset the YOLO detector was trained on. | The dataset exists; the recipe doesn't need to be re-run. YOLO is retired anyway. |
| `test_grounding_dino.py` | Zero-shot Grounding DINO sanity check that motivated the pseudo-labeling above. Wrote to `gdino_out/` (also deleted). | Exploration done; outcome was incorporated into the pseudo-labeling pipeline. |
| `audit_labels.py` | YOLO-era tool: ran `best.pt` at high confidence to flag GT boxes the model missed and detector predictions the GT missed. | Tied to YOLO-era label auditing. Won't be re-run on RF-DETR. |
| `visualize_classifier.py` | Generated the (now-deleted) `results/correct_predictions.png`, `results/near_misses.png`, `results/attention_maps.png` visualizations. | The outputs are stale; the script could regenerate fresh versions but hasn't been run since April. |
| `visualize_pipeline.py` | Generated the (now-deleted) `demo_vis/conf_0.0X/` per-conf debug visualizations of YOLO + classifier output. | Pre-#6, pre-RF-DETR. The replacement is `compare_overlays.py` (RF-DETR-vs-YOLO at the prediction level). |
| `export_onnx.py` | One-shot ONNX export of Stage 1 (DINOv3 backbone + MCEAM + head). Note: the head dim was hardcoded to 3 — never updated for the real 260-class model, so it has *never* successfully exported the current checkpoint. | Half-finished. If ONNX export becomes relevant for deployment, rewrite it (~30 min on current checkpoints, which now save `num_classes` per issue #24). |

## Running an archived script

These scripts import from the project — `bioreef.*`, `train_stage1`,
`eval_pipeline`, etc. To run one without breaking those imports:

```bash
# from the project root, not from inside archive/:
python archive/visualize_classifier.py [args...]
```

If you ever need to run one from inside `archive/`, set `PYTHONPATH=..` so
the imports resolve.
