"""
Test classifier prediction stability across raw vs WaterNet-restored inputs.

The question this answers: if we drop WaterNet from the production inference
pipeline (because it hurts the detector by 5.6 pp recall), how much does
the classifier's behavior change as a consequence?

The classifier was trained exclusively on WaterNet-restored OzFish frames.
If we feed it raw Khorfakkan crops in production, that's a train/inference
distribution shift. This script measures the magnitude of that shift
WITHOUT requiring species labels (which we don't have for Khorfakkan).

Strategy:
    For every GT box in the given clips, run the FULL Stage 1 classifier
    (ContextHarvester 4-stream → backbone → MCEAM → head) twice:
      (a) using the raw source frame, and
      (b) using the WaterNet-restored source frame.
    Compare the predictions on each crop:
      - Top-1 agreement rate (do the two inputs predict the same species?)
      - Top-5 agreement rate (does raw's top-1 land in restored's top-5?)
      - Softmax cosine similarity (how similar are the full distributions?)

If predictions are highly stable (e.g. >90% Top-1 agreement) → MCEAM is
robust to the distribution shift; dropping WaterNet is safe.
If predictions diverge wildly (e.g. <60% agreement) → MCEAM is sensitive
to the input distribution; retraining on raw OzFish is justified.

This is the CHEAP test that gates the EXPENSIVE retrain decision.

Usage:
    # Default: test on all three labeled Khorfakkan clips
    python classifier_stability_test.py

    # Override clip list
    python classifier_stability_test.py --clip <dir> [--clip <dir> ...]

    # Each clip dir is expected to contain:
    #   <clip>/annotations/instances_default.json
    #   <clip>/images/        (or images/default/)
    #   <clip>/images_restored/ (created if missing via restore_frames.py logic)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- repo-root bootstrap: this script lives in scripts/<area>/; add the
# repo root (two levels up) to sys.path so `import bioreef` resolves no
# matter the cwd or how the script is invoked. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from bioreef.data.data_factory import (
    ContextHarvester,
    WaterNetRestorer,
)
from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from bioreef.data.dataset_split import resolve_species_mapping


# Default test clips — the three labeled Khorfakkan artifacts on the VM.
DEFAULT_CLIPS = [
    "annotations",
    "Khorfakkan/10-59-21_DEEP_TREKKER_SD_585_784_frames",
    "Khorfakkan/10-59-21_DEEP_TREKKER_SD_13260_13459_frames",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", action="append", default=None,
                   help=f"Clip directory (repeatable). Defaults to all three "
                        f"labeled Khorfakkan clips: {DEFAULT_CLIPS}")
    p.add_argument("--stage1_ckpt", default="models/bioreef_stage1.pt",
                   help="Stage 1 classifier checkpoint. Default: models/bioreef_stage1.pt")
    p.add_argument("--csv_path", default="frame_metadata.csv",
                   help="OzFish CSV (for species mapping fallback).")
    p.add_argument("--min_samples", type=int, default=20,
                   help="Species min-samples threshold for mapping fallback.")
    p.add_argument("--max_per_clip", type=int, default=None,
                   help="Cap the number of GT boxes tested per clip "
                        "(for a faster sanity run). Default: all.")
    p.add_argument("--device", default=None,
                   help="cuda or cpu. Default: auto.")
    return p.parse_args()


def find_images_dir(clip_dir: str) -> str:
    """Locate the raw images dir — handles `images/` and `images/default/`."""
    for candidate in (os.path.join(clip_dir, "images"),
                      os.path.join(clip_dir, "images", "default")):
        if os.path.isdir(candidate) and any(
            f.endswith(".png") for f in os.listdir(candidate)
        ):
            return candidate
    raise FileNotFoundError(f"no images dir found under {clip_dir}")


def find_gt_json(clip_dir: str) -> str:
    """
    Locate the GT JSON. Handles two layouts:
      1. Nested:   <clip>/annotations/instances_default.json   (Khorfakkan clips)
      2. Flat:     <clip>/instances_default.json               (the 'annotations/' clip itself)
    """
    for candidate in (
        os.path.join(clip_dir, "annotations", "instances_default.json"),
        os.path.join(clip_dir, "instances_default.json"),
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"no GT JSON found under {clip_dir}")


def restored_path_for(images_dir: str) -> str:
    """Pick the parallel '_restored' directory name for a given images dir."""
    # If images_dir is 'foo/images', restored is 'foo/images_restored'.
    # If images_dir is 'foo/images/default', restored is 'foo/images_restored/default'.
    if os.path.basename(images_dir) == "default":
        parent = os.path.dirname(images_dir)  # foo/images
        grand = os.path.dirname(parent)        # foo
        return os.path.join(grand, "images_restored", "default")
    return images_dir + "_restored"


def ensure_restored(images_dir: str, restored_dir: str,
                    restorer: WaterNetRestorer) -> int:
    """Restore any frames missing from restored_dir. Returns count restored."""
    os.makedirs(restored_dir, exist_ok=True)
    files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    todo = [f for f in files if not os.path.exists(os.path.join(restored_dir, f))]
    if not todo:
        return 0

    print(f"  restoring {len(todo)} frames -> {restored_dir}")
    t0 = time.time()
    for i, fname in enumerate(todo):
        raw = cv2.imread(os.path.join(images_dir, fname))
        if raw is None:
            print(f"    warn: could not read {fname}", file=sys.stderr)
            continue
        cv2.imwrite(os.path.join(restored_dir, fname), restorer(raw))
        if (i + 1) % 10 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    [{i+1}/{len(todo)}] {rate:.2f} img/s", end="\r")
    print()
    return len(todo)


def load_clip(clip_dir: str) -> Tuple[str, str, List[Dict], Dict[int, str]]:
    """
    Returns (images_dir, restored_dir, annotations, image_id->filename).
    """
    gt_path = find_gt_json(clip_dir)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    images_dir = find_images_dir(clip_dir)
    restored_dir = restored_path_for(images_dir)
    id_to_file = {im["id"]: im["file_name"] for im in gt["images"]}
    annotations = gt["annotations"]
    return images_dir, restored_dir, annotations, id_to_file


@torch.no_grad()
def classify_crop(
    frame_bgr: np.ndarray, bbox_xywh: List[float],
    backbone: ViTBackbone, mceam: MCEAM, head: nn.Module,
    harvester: ContextHarvester, device: torch.device,
) -> np.ndarray:
    """
    Run the full Stage 1 4-stream classifier on one fish crop.
    Returns the (C,) softmax probabilities.
    """
    x, y, w, h = bbox_xywh
    crops = harvester.harvest(frame_bgr, (int(x), int(y),
                                          max(int(w), 1), max(int(h), 1)))
    # crops is a dict {stream_name: tensor(3, 224, 224)}; batch dim of 1
    batched = {name: t.unsqueeze(0).to(device) for name, t in crops.items()}
    feats = backbone(batched)
    mceam_out = mceam(feats)
    fused = mceam_out["embedding"]  # (1, 256)
    logits = head(fused)            # (1, C)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def main() -> int:
    args = parse_args()
    clips = args.clip if args.clip else DEFAULT_CLIPS

    # Validate clips exist before loading any model
    for c in clips:
        if not os.path.isdir(c):
            print(f"error: clip dir not found: {c}", file=sys.stderr)
            return 2

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    # --- Load classifier components ---
    print("loading backbone (DINOv3 ViT-B/16)...")
    backbone = ViTBackbone(freeze=True).to(device).eval()

    print(f"loading Stage 1 checkpoint: {args.stage1_ckpt}")
    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    num_classes = ckpt["head"]["weight"].shape[0]
    idx_to_sp = resolve_species_mapping(ckpt, args.csv_path, args.min_samples)
    print(f"  {num_classes} classes")

    mceam = MCEAM(embed_dim=backbone.embed_dim, num_context_levels=3,
                  output_dim=256, num_heads=8).to(device).eval()
    mceam.load_state_dict(ckpt["mceam"])
    head = nn.Linear(256, num_classes).to(device).eval()
    head.load_state_dict(ckpt["head"])

    harvester = ContextHarvester()

    print("loading WaterNet (for on-demand frame restoration)...")
    restorer = WaterNetRestorer()
    restorer._load_model()

    # --- Per-clip stability metrics ---
    per_clip_stats: Dict[str, Dict] = {}

    for clip_dir in clips:
        print()
        print(f"=== {clip_dir} ===")
        images_dir, restored_dir, annotations, id_to_file = load_clip(clip_dir)
        print(f"  images   : {images_dir}")
        print(f"  restored : {restored_dir}")
        print(f"  GT boxes : {len(annotations)}")

        # Make sure restored frames exist
        ensure_restored(images_dir, restored_dir, restorer)

        # Group annotations by image to avoid re-reading the same frame
        by_image: Dict[int, List[Dict]] = defaultdict(list)
        for a in annotations:
            by_image[a["image_id"]].append(a)

        if args.max_per_clip:
            # Sample boxes across images, not all from the first few
            all_boxes = sum(([a for a in v] for v in by_image.values()), [])
            all_boxes = all_boxes[:args.max_per_clip]
            wanted_ids = {a["image_id"] for a in all_boxes}
            by_image = {k: [a for a in v if a in all_boxes]
                        for k, v in by_image.items() if k in wanted_ids}

        # Stats accumulators for this clip
        top1_agree = 0
        top5_agree = 0
        cos_sims: List[float] = []
        total = 0
        # Diversity tracking — to distinguish "robust" from "collapsed onto
        # a single species." A high-agreement classifier that predicts the
        # same 3 species for every crop isn't robust; it's broken.
        raw_pred_counts: "Counter[int]" = Counter()
        rest_pred_counts: "Counter[int]" = Counter()
        # Per-crop predictions for the second-pass conditional-agreement
        # computation (need to know the dominant raw prediction first).
        per_crop_preds: List[Tuple[int, int]] = []  # (top1_raw, top1_rest)

        n_images = len(by_image)
        t0 = time.time()
        for img_idx, (image_id, boxes) in enumerate(by_image.items()):
            fname = id_to_file.get(image_id)
            if not fname:
                continue
            raw_path = os.path.join(images_dir, os.path.basename(fname))
            rest_path = os.path.join(restored_dir, os.path.basename(fname))
            if not (os.path.exists(raw_path) and os.path.exists(rest_path)):
                continue

            raw_frame = cv2.imread(raw_path)
            rest_frame = cv2.imread(rest_path)
            if raw_frame is None or rest_frame is None:
                continue

            for a in boxes:
                probs_raw = classify_crop(raw_frame, a["bbox"],
                                          backbone, mceam, head,
                                          harvester, device)
                probs_rest = classify_crop(rest_frame, a["bbox"],
                                           backbone, mceam, head,
                                           harvester, device)

                top1_raw = int(probs_raw.argmax())
                top1_rest = int(probs_rest.argmax())
                if top1_raw == top1_rest:
                    top1_agree += 1

                top5_raw = set(np.argsort(probs_raw)[-5:].tolist())
                top5_rest = set(np.argsort(probs_rest)[-5:].tolist())
                if top1_raw in top5_rest and top1_rest in top5_raw:
                    top5_agree += 1

                # Cosine sim of full distributions
                a_norm = probs_raw / (np.linalg.norm(probs_raw) + 1e-12)
                b_norm = probs_rest / (np.linalg.norm(probs_rest) + 1e-12)
                cos_sims.append(float(np.dot(a_norm, b_norm)))

                # Diversity tracking
                raw_pred_counts[top1_raw] += 1
                rest_pred_counts[top1_rest] += 1
                per_crop_preds.append((top1_raw, top1_rest))

                total += 1

            if (img_idx + 1) % 10 == 0 or (img_idx + 1) == n_images:
                rate = total / max(time.time() - t0, 1e-9)
                print(f"  [{img_idx+1}/{n_images} images, "
                      f"{total} crops, {rate:.1f} crops/s]", end="\r")

        print()
        if total == 0:
            print(f"  no crops processed for {clip_dir}")
            continue

        # Diversity: distinct species + dominant-class share + Shannon entropy.
        # A high-agreement clip with diverse predictions = genuinely robust.
        # A high-agreement clip with collapsed predictions = degenerate.
        def shannon_bits(counter: "Counter[int]") -> float:
            total_c = sum(counter.values())
            if total_c <= 1:
                return 0.0
            probs = np.array([c / total_c for c in counter.values()])
            return float(-np.sum(probs * np.log2(probs + 1e-12)))

        raw_distinct = len(raw_pred_counts)
        rest_distinct = len(rest_pred_counts)
        raw_dominant_class, raw_dominant_count = raw_pred_counts.most_common(1)[0]
        raw_dominant_share = raw_dominant_count / total
        rest_dominant_class, rest_dominant_count = rest_pred_counts.most_common(1)[0]
        rest_dominant_share = rest_dominant_count / total
        raw_entropy = shannon_bits(raw_pred_counts)
        rest_entropy = shannon_bits(rest_pred_counts)

        # Conditional agreement on non-dominant raw predictions — does the
        # classifier agree when raw didn't pick its default class?
        non_dominant = [(tr, te) for (tr, te) in per_crop_preds
                        if tr != raw_dominant_class]
        non_dom_total = len(non_dominant)
        non_dom_agree = sum(1 for tr, te in non_dominant if tr == te)
        non_dom_rate = (non_dom_agree / non_dom_total) if non_dom_total else float("nan")

        # Resolve species names for the top-3 (uses the same idx_to_sp the
        # classifier was loaded with; falls back to numeric idx if unmapped).
        def name(i: int) -> str:
            return idx_to_sp.get(i, f"<{i}>")
        raw_top3 = [(name(c), n) for c, n in raw_pred_counts.most_common(3)]
        rest_top3 = [(name(c), n) for c, n in rest_pred_counts.most_common(3)]

        clip_stats = {
            "n_crops": total,
            "top1_agreement": top1_agree / total,
            "top5_agreement": top5_agree / total,
            "mean_cosine_sim": float(np.mean(cos_sims)),
            "median_cosine_sim": float(np.median(cos_sims)),
            "raw_distinct_species": raw_distinct,
            "rest_distinct_species": rest_distinct,
            "raw_dominant_share": raw_dominant_share,
            "rest_dominant_share": rest_dominant_share,
            "raw_entropy_bits": raw_entropy,
            "rest_entropy_bits": rest_entropy,
            "non_dominant_n": non_dom_total,
            "non_dominant_agreement": non_dom_rate,
            "raw_top3": raw_top3,
            "rest_top3": rest_top3,
        }
        per_clip_stats[clip_dir] = clip_stats
        print(f"  Top-1 agreement: {clip_stats['top1_agreement']:.1%}")
        print(f"  Top-5 agreement: {clip_stats['top5_agreement']:.1%}")
        print(f"  mean cosine sim: {clip_stats['mean_cosine_sim']:.3f}")
        print(f"  distinct species predicted: raw={raw_distinct}, restored={rest_distinct}")
        print(f"  dominant-class share:       raw={raw_dominant_share:.1%}, "
              f"restored={rest_dominant_share:.1%}")
        print(f"  entropy (bits):             raw={raw_entropy:.2f}, "
              f"restored={rest_entropy:.2f}")
        if non_dom_total:
            print(f"  agreement on non-dominant   "
                  f"({non_dom_total} crops): {non_dom_rate:.1%}")
        else:
            print(f"  agreement on non-dominant   (n/a — 100% dominant)")
        print(f"  raw top-3:      {raw_top3}")
        print(f"  restored top-3: {rest_top3}")

    # --- Aggregate report ---
    print()
    print("=" * 72)
    print("CLASSIFIER STABILITY: RAW vs WATERNET-RESTORED INPUTS")
    print("=" * 72)
    if not per_clip_stats:
        print("  no clips produced results")
        return 1

    header = (f"{'clip':<50s} {'n':>5s} {'Top-1':>7s} {'Top-5':>7s} "
              f"{'cos':>6s} {'distinct':>10s} {'dom-share':>10s} {'non-dom':>9s}")
    print(header)
    print("-" * len(header))
    total_crops = 0
    weighted_top1 = 0.0
    weighted_top5 = 0.0
    weighted_cos = 0.0
    weighted_nondom_num = 0.0  # weighted by non-dominant count
    weighted_nondom_den = 0
    for clip, s in per_clip_stats.items():
        short = clip[-47:] if len(clip) > 47 else clip
        distinct_str = f"{s['raw_distinct_species']}/{s['rest_distinct_species']}"
        dom_str = f"{s['raw_dominant_share']*100:.0f}/{s['rest_dominant_share']*100:.0f}%"
        nondom_str = (f"{s['non_dominant_agreement']*100:.0f}%"
                      if s['non_dominant_n'] else "n/a")
        print(f"{short:<50s} {s['n_crops']:>5d} "
              f"{s['top1_agreement']*100:>6.1f}% "
              f"{s['top5_agreement']*100:>6.1f}% "
              f"{s['mean_cosine_sim']:>6.3f} "
              f"{distinct_str:>10s} {dom_str:>10s} {nondom_str:>9s}")
        total_crops += s["n_crops"]
        weighted_top1 += s["top1_agreement"] * s["n_crops"]
        weighted_top5 += s["top5_agreement"] * s["n_crops"]
        weighted_cos += s["mean_cosine_sim"] * s["n_crops"]
        if s["non_dominant_n"]:
            weighted_nondom_num += s["non_dominant_agreement"] * s["non_dominant_n"]
            weighted_nondom_den += s["non_dominant_n"]

    print("-" * len(header))
    overall_top1 = weighted_top1 / total_crops
    overall_top5 = weighted_top5 / total_crops
    overall_cos = weighted_cos / total_crops
    overall_nondom = (weighted_nondom_num / weighted_nondom_den
                      if weighted_nondom_den else float("nan"))
    nondom_overall_str = (f"{overall_nondom*100:.0f}%"
                          if not np.isnan(overall_nondom) else "n/a")
    print(f"{'WEIGHTED OVERALL':<50s} {total_crops:>5d} "
          f"{overall_top1*100:>6.1f}% {overall_top5*100:>6.1f}% "
          f"{overall_cos:>6.3f} {'':>10s} {'':>10s} {nondom_overall_str:>9s}")

    # Per-clip top-3 dominant species — the most informative diagnostic.
    print()
    print("TOP-3 PREDICTED SPECIES PER CLIP (raw / restored):")
    for clip, s in per_clip_stats.items():
        short = clip[-50:] if len(clip) > 50 else clip
        print(f"  {short}")
        print(f"    raw      : {s['raw_top3']}")
        print(f"    restored : {s['rest_top3']}")

    # Refined interpretation: use BOTH agreement AND diversity.
    # The "robust" vs "collapsed" distinction matters more than raw agreement.
    print()
    print("INTERPRETATION:")
    # Check for collapsed clips: any clip with dominant-class share > 0.7
    collapsed = [c for c, s in per_clip_stats.items()
                 if s['raw_dominant_share'] > 0.7 or s['rest_dominant_share'] > 0.7]
    high_agreement_clips = [c for c, s in per_clip_stats.items()
                            if s['top1_agreement'] > 0.9]
    if collapsed and any(c in high_agreement_clips for c in collapsed):
        print(f"  → Some clip(s) show HIGH Top-1 agreement coexisting with a")
        print(f"    DEGENERATE prediction pattern (>70% of crops predicted as a")
        print(f"    single species). That is NOT genuine robustness — it is the")
        print(f"    classifier defaulting to one answer regardless of input.")
        print(f"    The expensive retrain would NOT fix this (it's a cross-domain")
        print(f"    OzFish→Khorfakkan problem, not a WaterNet/raw shift problem).")
        print(f"    Look at the TOP-3 SPECIES table above to confirm.")
    elif overall_nondom > 0.7:
        print(f"  → Non-dominant agreement {overall_nondom:.0%}: the classifier")
        print(f"    is robust on the predictions that matter (the non-default")
        print(f"    ones). Dropping WaterNet should not destabilize useful")
        print(f"    classifications. Retrain unlikely to help.")
    elif overall_top1 > 0.90:
        print(f"  → Top-1 agreement {overall_top1:.0%}, no clip appears degenerate.")
        print(f"    The classifier is HIGHLY STABLE. Dropping WaterNet is safe.")
        print(f"    Retrain is unlikely to change much.")
    elif overall_top1 > 0.70:
        print(f"  → Top-1 agreement {overall_top1:.0%}: MODERATE stability with no")
        print(f"    obvious degenerate behavior. Retrain might modestly improve")
        print(f"    consistency; cost-benefit depends on classifier priority.")
    else:
        print(f"  → Top-1 agreement {overall_top1:.0%}: classifier is SENSITIVE to")
        print(f"    the WaterNet/raw boundary, and the cross-clip pattern is")
        print(f"    inconsistent. Before committing to a 4-day retrain, confirm")
        print(f"    via the top-3 species table that this isn't just cross-")
        print(f"    domain failure being labeled 'instability.'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
