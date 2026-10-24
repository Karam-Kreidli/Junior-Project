"""
BioReef.ai — End-to-End Video Demo (Stages 1 + 2)
====================================================
Runs the full pipeline on a raw video file and produces:
  - results/demo_annotated.mp4    : video with bbox + track ID + species overlays
  - results/track_stats.json      : descriptive tracking statistics
  - results/keyframes/*.png       : a few representative frames for the report

Usage:
    python demo_video.py --video path/to/raw.mp4
    python demo_video.py --video raw.mp4 --max_frames 600 --conf 0.05
"""

import os
import json
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
# --- repo-root bootstrap: this script lives in scripts/<area>/; add the
# repo root (two levels up) to sys.path so `import bioreef` resolves no
# matter the cwd or how the script is invoked. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from bioreef._2_stage1 import build_detector

from bioreef._2_stage1._22_backbone import ViTBackbone
from bioreef._2_stage1._23_mceam import MCEAM
from bioreef._1_preprocess._11_restoration import WaterNetRestorer
from bioreef._1_preprocess._12_context import ContextHarvester
from bioreef._3_stage2 import SUTTracker
from bioreef._9_pipeline._92_detect import detect_frame, extract_embeddings
from bioreef._9_pipeline.config import InferenceConfig
from bioreef._9_pipeline.models import load_models
from bioreef.viz import color_for_id
from bioreef._1_preprocess._15_dataset_split import (
    build_species_mapping, resolve_species_mapping,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",          required=True, help="Path to raw video file (.mp4/.avi)")
    p.add_argument("--detector_backend", default="rfdetr", choices=["rfdetr", "yolo"],
                   help="Detector backend (production: rfdetr per #6).")
    p.add_argument("--detection_ckpt", default=None,
                   help="Detector checkpoint. Defaults to "
                        "models/rfdetr_medium_cfd.pth for rfdetr; required for yolo.")
    p.add_argument("--rfdetr_size", default="medium", choices=["medium", "small", "nano"],
                   help="RF-DETR variant (ignored for yolo). Default: medium.")
    p.add_argument("--imgsz", type=int, default=960,
                   help="YOLO inference imgsz (ignored for rfdetr).")
    p.add_argument("--stage1_ckpt",    default="models/d6a_stage1.pt")
    p.add_argument("--stage1_prior",   default="models/d6a_prior.json",
                   help="Logit-adjustment prior beside the checkpoint (see models/README.md)")
    p.add_argument("--classify_every", type=int, default=10,
                   help="Re-classify each track every N frames (1 = every frame); "
                        "the shown species is the track's majority vote")
    p.add_argument("--tau",            type=float, default=1.0,
                   help="Logit-adjustment strength; 1.0 = deployed setting, 0 = off")
    p.add_argument("--csv_path",       default="data_oz/metadata/frame_metadata.csv")
    p.add_argument("--min_samples",    type=int, default=20)
    p.add_argument("--conf",           type=float, default=0.15,
                   help="Detector confidence floor. Raised from 0.05: at 0.05 the "
                        "detector emits ~30 boxes/frame for a ~5-fish scene (~25 "
                        "phantoms), which the tracker then churns into hundreds of "
                        "short tracks. 0.15 keeps recall while cutting the swarm; "
                        "SU-T ignores boxes below 0.1 and starts tracks only at "
                        "--track_thresh, so this is just a pre-filter.")
    p.add_argument("--max_frames",     type=int, default=None, help="Limit frames (None = full video)")
    p.add_argument("--out_dir",        default="results")
    p.add_argument("--keyframe_every", type=int, default=120, help="Save a keyframe every N frames")
    p.add_argument("--waternet",       action="store_true",
                   help="Apply WaterNet restoration before detection. Off by "
                        "default — empirically WaterNet hurts RF-DETR recall "
                        "by ~5.6 pp on Khorfakkan and the classifier's "
                        "predictions are unstable across raw/restored either "
                        "way (#14). Keep off for production; enable only for "
                        "qualitative demos where visual clarity matters.")
    p.add_argument("--save_restored",  action="store_true",
                   help="Also save a side-by-side raw|restored debug video")
    # Tracker (SU-T, motion-only). Defaults = its published MFT25 config.
    p.add_argument("--track_thresh",  type=float, default=0.5,  help="Detection conf needed to START a track; boxes between 0.1 and this can only extend one")
    p.add_argument("--iou_threshold", type=float, default=0.25, help="FishIoU threshold for a match")
    p.add_argument("--max_age",       type=int,   default=30,   help="Frames a lost track survives")
    p.add_argument("--min_hits",      type=int,   default=3,    help="Matches before a track is shown")
    p.add_argument("--containment_thresh",   type=float, default=0.7,
                   help="Drop a larger bbox if a smaller one is more than this fraction inside it")
    p.add_argument("--show",                 action="store_true",
                   help="Also play the annotated video in a window while it runs (q to stop)")
    p.add_argument("--motion_only",          action="store_true",
                   help="Disable classifier and use motion-only tracking (faster, ignores appearance)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stable colour per track ID
# ---------------------------------------------------------------------------

def filter_nested_detections(bboxes: np.ndarray, confs: np.ndarray,
                             containment_thresh: float = 0.7):
    """
    Drop loose bounding boxes that mostly contain a tighter box.

    YOLO's NMS uses standard IoU, so when a large box fully encloses a
    smaller one (IoU ≈ 0.3–0.5), both survive. Downstream this causes
    the tracker to oscillate between two tracks claiming the same fish.
    We keep the tighter inner box (higher fidelity localization) and
    drop the looser outer box.
    """
    n = len(bboxes)
    if n < 2:
        return bboxes, confs, np.ones(n, dtype=bool)

    keep = np.ones(n, dtype=bool)
    areas = bboxes[:, 2] * bboxes[:, 3]

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if areas[i] <= areas[j]:
                continue  # only consider dropping the larger of the pair
            x1 = max(bboxes[i, 0], bboxes[j, 0])
            y1 = max(bboxes[i, 1], bboxes[j, 1])
            x2 = min(bboxes[i, 0] + bboxes[i, 2], bboxes[j, 0] + bboxes[j, 2])
            y2 = min(bboxes[i, 1] + bboxes[i, 3], bboxes[j, 1] + bboxes[j, 3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if inter / max(areas[j], 1e-6) > containment_thresh:
                keep[i] = False
                break

    return bboxes[keep], confs[keep], keep


def draw_overlay(frame, tracks, track_to_species, frame_idx, total_unique):
    out = frame.copy()
    h, w = out.shape[:2]

    for t in tracks:
        x, y, bw, bh = [int(v) for v in t.bbox]
        x2, y2 = x + bw, y + bh
        color = color_for_id(t.track_id)

        cv2.rectangle(out, (x, y), (x2, y2), color, 2)

        species = track_to_species.get(t.track_id)
        label = f"#{t.track_id} {species}" if species else f"#{t.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y - 6, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 2), color, -1)
        cv2.putText(out, label, (x + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # HUD
    hud = f"Frame {frame_idx}  |  Active tracks: {len(tracks)}  |  Unique tracks: {total_unique}"
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(out, hud, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    keyframe_dir = os.path.join(args.out_dir, "keyframes")
    os.makedirs(keyframe_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    if args.motion_only:
        detector = build_detector(
            args.detector_backend,
            weights=args.detection_ckpt,
            model_size=args.rfdetr_size,
            imgsz=args.imgsz,
        )

    # In motion-only mode we track purely off the detector's boxes — no
    # backbone / MCEAM / classifier. Skip loading them entirely so this path
    # runs fully offline (the DINOv3 backbone is gated on HF and needn't be
    # touched just to eyeball tracking). Species overlays are omitted here.
    backbone = mceam = head = harvester = None
    idx_to_sp = {}
    logit_prior, logit_tau = None, 0.0
    if not args.motion_only:
        # Same loader as the pipeline: handles D6a (full fine-tuned Classifier)
        # and the legacy {mceam, head} format, plus the logit-adjustment prior.
        m = load_models(InferenceConfig(
            stage1_ckpt=args.stage1_ckpt, stage1_prior=args.stage1_prior,
            stage1_logit_tau=args.tau, csv_path=args.csv_path, min_samples=args.min_samples,
            detector_backend=args.detector_backend, detection_ckpt=args.detection_ckpt,
            rfdetr_size=args.rfdetr_size, imgsz=args.imgsz))
        detector, backbone, mceam, head, harvester = m.detector, m.backbone, m.mceam, m.head, m.harvester
        idx_to_sp, logit_prior, logit_tau = m.idx_to_sp, m.logit_prior, m.logit_tau
    tracker = SUTTracker(track_thresh=args.track_thresh, iou_threshold=args.iou_threshold,
                         max_age=args.max_age, min_hits=args.min_hits)

    # WaterNet — OFF by default in production (#14, 2026-05-26):
    #   - RF-DETR recall is +5.6 pp on raw vs WaterNet-restored Khorfakkan
    #     (the CFD training distribution already includes murky underwater
    #     footage; "improved" inputs become out-of-distribution).
    #   - Classifier is cross-domain-broken on Khorfakkan either way, so
    #     matching its OzFish-restored training distribution doesn't help.
    # The flag is opt-in for the rare case where a human is watching the
    # output and visual clarity matters more than detector quality.
    waternet = None
    if args.waternet:
        print("Loading WaterNet (spectral restoration — opt-in only)...")
        waternet = WaterNetRestorer()
        waternet._load_model()
        # Sanity check: ensure the real WaterNet loaded, not the Identity fallback
        if hasattr(waternet, "_model") and "Identity" in str(type(waternet._model)):
            print("[!] WARNING: WaterNet failed to load — falling back to passthrough.")
            print("    Inference will run on raw frames; expect degraded accuracy.")
            waternet = None
        else:
            print("WaterNet loaded successfully on GPU.")

    # --- Video I/O ---
    cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames is not None:
        total = min(total, args.max_frames)

    out_path = os.path.join(args.out_dir, "demo_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    debug_writer = None
    if args.save_restored and waternet is not None:
        debug_path = os.path.join(args.out_dir, "raw_vs_restored.mp4")
        debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (W * 2, H))
        print(f"Debug:  {debug_path}")

    print(f"Video: {W}x{H} @ {fps:.1f} fps, {total} frames")
    print(f"Output: {out_path}")
    print(f"WaterNet: {'enabled' if waternet is not None else 'DISABLED (raw frames)'}")

    # --- Tracking state ---
    track_to_species_votes = defaultdict(lambda: defaultdict(int))  # track_id -> species -> votes
    last_classified = {}                                            # track_id -> frame last classified
    track_first_seen = {}
    track_last_seen = {}
    active_count_per_frame = []

    pbar = tqdm(total=total, desc="Processing", unit="f")
    frame_idx = 0

    while frame_idx < total:
        ret, raw_frame = cap.read()
        if not ret:
            break

        # 1. WaterNet restoration — match the training distribution.
        if waternet is not None:
            frame = waternet(raw_frame)
        else:
            frame = raw_frame

        # Side-by-side debug video (optional)
        if debug_writer is not None:
            debug_writer.write(np.hstack([raw_frame, frame]))

        # 2. Detect (on restored frame)
        bboxes, confs, _ = detect_frame(detector, frame, args.conf)

        # 2b. Drop containment-duplicates (loose box wrapping a tight one).
        bboxes, confs, _ = filter_nested_detections(
            bboxes, confs, containment_thresh=args.containment_thresh
        )

        # 3. Track first (SU-T, motion-only: it needs only boxes).
        active = tracker.update(bboxes, confs, None, frame_id=frame_idx)

        # 4. Classify per TRACK, not per detection: a fish's species doesn't change
        #    between frames, so each confirmed track is classified when it appears
        #    and then every --classify_every frames; its label is the majority vote.
        #    t.bbox is the detection the track matched this frame.
        due = [] if args.motion_only else [
            t for t in active
            if frame_idx - last_classified.get(t.track_id, -10**9) >= args.classify_every]
        if due:
            with torch.no_grad(), torch.amp.autocast("cuda"):
                emb_np, _, logits_np = extract_embeddings(
                    backbone, mceam, head, harvester, frame,
                    np.array([t.bbox for t in due]), device,
                    logit_prior=logit_prior, logit_tau=logit_tau,
                )
            for t, emb, lg in zip(due, emb_np, logits_np):
                track_to_species_votes[t.track_id][idx_to_sp.get(int(lg.argmax()), "?")] += 1
                last_classified[t.track_id] = frame_idx
                f, box, _, _ = t.frame_history[-1]     # embedding rides along for Stage 3
                t.frame_history[-1] = (f, box, emb, lg)

        # Resolve species per track via majority vote so far
        track_to_species = {
            tid: max(votes.items(), key=lambda kv: kv[1])[0]
            for tid, votes in track_to_species_votes.items()
        }

        # Lifespan tracking
        for t in active:
            if t.track_id not in track_first_seen:
                track_first_seen[t.track_id] = frame_idx
            track_last_seen[t.track_id] = frame_idx

        active_count_per_frame.append(len(active))

        # 5. Render
        annotated = draw_overlay(frame, active, track_to_species,
                                 frame_idx, len(track_first_seen))
        writer.write(annotated)
        if args.show:
            cv2.imshow("BioReef tracking (q to stop)", cv2.resize(annotated, (1280, int(1280 * H / W))))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Save keyframe
        if frame_idx % args.keyframe_every == 0 and len(active) > 0:
            kf_path = os.path.join(keyframe_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(kf_path, annotated)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    writer.release()
    if debug_writer is not None:
        debug_writer.release()

    # ----------------------------------------------------------------------
    # Descriptive statistics
    # ----------------------------------------------------------------------
    lifespans = [track_last_seen[t] - track_first_seen[t] + 1
                 for t in track_first_seen]

    stats = {
        "video":                  os.path.basename(args.video),
        "frames_processed":       frame_idx,
        "fps":                    round(fps, 2),
        "total_unique_tracks":    len(track_first_seen),
        "max_simultaneous_tracks": int(max(active_count_per_frame)) if active_count_per_frame else 0,
        "mean_active_per_frame":   round(float(np.mean(active_count_per_frame)), 2) if active_count_per_frame else 0,
        "track_lifespan_frames": {
            "min":    int(min(lifespans)) if lifespans else 0,
            "median": int(np.median(lifespans)) if lifespans else 0,
            "mean":   round(float(np.mean(lifespans)), 2) if lifespans else 0,
            "max":    int(max(lifespans)) if lifespans else 0,
        },
        "species_distribution": {
            sp: sum(1 for tid, votes in track_to_species_votes.items()
                    if votes and max(votes.items(), key=lambda kv: kv[1])[0] == sp)
            for sp in set(s for v in track_to_species_votes.values()
                          for s in v)
        },
    }

    stats_path = os.path.join(args.out_dir, "track_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Tracking Summary ===")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sk, sv in (list(v.items())[:8] if k == "species_distribution" else v.items()):
                print(f"  {sk}: {sv}")
        else:
            print(f"{k}: {v}")

    print(f"\nVideo:  {out_path}")
    print(f"Stats:  {stats_path}")
    print(f"Keyframes: {keyframe_dir}/")


if __name__ == "__main__":
    main()
