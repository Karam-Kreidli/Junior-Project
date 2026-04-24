# BioReef.ai — Open Problems Before Stage 2 (Tracker)

Tracked issues that should be addressed before or during tracker integration.
Ordered by priority (impact × likelihood).

---

## 1. Re-ID embedding mismatch (CRITICAL)

**Problem:** BoTSORT uses appearance embeddings to maintain identity across frames. The current plan is to feed MCEAM's 256-dim embedding — but MCEAM is trained for *species discrimination*, not *individual discrimination*. Two individuals of the same species end up close in embedding space, which is exactly what you don't want for tracking.

**Consequences if ignored:**
- ID swaps between two same-species fish in the same frame/neighborhood
- Over-merging — one track spanning multiple individuals
- Broken count metric (the primary downstream deliverable)

**Options:**
- **(a)** Swap in a dedicated re-ID model (OSNet, FastReID) — standard BoTSORT stack. Requires an extra forward pass per detection.
- **(b)** Concatenate MCEAM embedding with a low-level CNN feature vector (earlier ViT layer patch features, pooled). Gives species-awareness + instance-level texture cues.
- **(c)** Use motion-only tracking first (BoTSORT IoU + Kalman, no re-ID) and evaluate how often identity is lost. If rare, skip re-ID altogether.

**Validation plan:** before committing to a strategy, run BoTSORT on a short test clip with 2+ same-species fish visible and manually check for ID swaps.

---

## 2. Softmax scores not saved in `.npz` output

**Problem:** `infer_stage1.py` only loads MCEAM, not the classifier head. The saved archive has `frame_ids, bboxes, confidences, embeddings, class_ids` — no per-detection species probabilities. Stage 2 needs these for track-level species aggregation (averaged softmax, majority vote, etc.).

**Consequences if ignored:** Stage 2 has to either re-run MCEAM + head on every tracked detection (wasteful) or load the head and reapply to saved embeddings (possible but awkward).

**Fix:** update `infer_stage1.py` to:
- Load the head from `bioreef_stage1.pt`
- Apply it to MCEAM embeddings → softmax
- Save top-5 (indices + probabilities) per detection instead of all 260 probs (saves disk space)

New `.npz` fields: `top5_indices (N, 5)` and `top5_scores (N, 5)`.

---

## 3. `infer_stage1.py` default `conf_threshold` is 0.3

**Problem:** `eval_pipeline.py` sweep confirmed `conf=0.05` is the optimal production setting (e2e_top1=55.77% vs 18.31% at conf=0.5). But `infer_stage1.py`'s default is still 0.3.

**Fix:** change default to 0.05. Trivial one-liner edit.

---

## 4. Tracking-level ground truth availability

**Problem:** the val split has per-frame annotations (species, bbox) but we haven't confirmed whether OzFish provides *track IDs* — i.e., "this fish in frame 100 is the same individual as in frame 101." Without track-level GT we can't compute MOTA / IDF1 / ID switches, which are the standard tracker metrics.

**Next step:** inspect the OzFish CSV / metadata for any per-individual identifier column (check `frame_metadata.csv` schema). If absent:
- Either accept degraded eval (e.g., track-level species accuracy only — count tracks whose aggregated species matches a labeled fish in the track's temporal window)
- Or manually label a small test clip (~100 frames, 1–2 videos) with track IDs for qualitative evaluation

---

## 5. Track aggregation logic + thresholds (DESIGN)

**Problem:** the "filter at the track level, not the detection level" decision needs concrete parameters. Unspecified:
- Minimum track length (frames) before a track is emitted as "a fish"
- Species aggregation method: averaged softmax, majority vote, weighted by detection confidence?
- Minimum mean softmax confidence to emit a species label (vs "unknown")
- What to do with tracks whose aggregated top-1 is a low-sample / low-confidence class

**Proposal to discuss:**
- Min track length: 5 frames (BoTSORT default is ~3)
- Species = `argmax(mean(softmax))` across all track frames
- Min mean confidence: 0.4 → else mark track as "unidentified fish" and report at the genus level from the taxonomy tree
- Use HD on track-level predictions as the primary quality metric

---

## 6. Detector label noise (DEFERRED)

**Problem:** current detector ceiling (mAP50 ~0.42) is driven largely by annotation noise in OzFish — missed fish, loose boxes, inconsistent tightness. Discussed in earlier session but deferred.

**Options (in order of effort):**
- **(a)** Auto-audit + pseudo-labeling via `cleanlab` + current `best.pt` (weekend's work, +3–8 mAP50 expected)
- **(b)** Box refinement with SAM (tighter boxes, +1–3 mAP50-95)
- **(c)** Full manual relabel (days, highest quality)

Not blocking Stage 2 — detector is "good enough" at `conf=0.05` where recall is 87%. Can revisit post-Stage-2 if count accuracy is poor.

---

## Summary priority

| # | Issue | Effort | Blocking Stage 2? |
|---|---|---|---|
| 1 | Re-ID embedding mismatch | Medium (a day to set up) | **Yes** |
| 2 | Save softmax in `.npz` | Low (30 min) | Yes |
| 3 | conf_threshold default | Low (1 min) | No |
| 4 | Track-level GT | Low (audit CSV) to High (label data) | For eval only |
| 5 | Track aggregation design | Low (discuss) | Yes |
| 6 | Detector label noise | Weekend+ | No |
