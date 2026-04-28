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
import colorsys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO

from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester, WaterNetRestorer
from bioreef.tracking import BoTSORTTracker
from infer_stage1 import detect_frame, extract_embeddings, build_species_mapping


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",          required=True, help="Path to raw video file (.mp4/.avi)")
    p.add_argument("--detection_ckpt", default="models/best.pt")
    p.add_argument("--stage1_ckpt",    default="models/bioreef_stage1.pt")
    p.add_argument("--csv_path",       default="data_oz/metadata/frame_metadata.csv")
    p.add_argument("--min_samples",    type=int, default=20)
    p.add_argument("--conf",           type=float, default=0.05)
    p.add_argument("--max_frames",     type=int, default=None, help="Limit frames (None = full video)")
    p.add_argument("--out_dir",        default="results")
    p.add_argument("--keyframe_every", type=int, default=120, help="Save a keyframe every N frames")
    p.add_argument("--no_waternet",    action="store_true",
                   help="Skip WaterNet restoration (NOT recommended — models trained on restored frames)")
    p.add_argument("--save_restored",  action="store_true",
                   help="Also save a side-by-side raw|restored debug video")
    # Tracker tuning
    p.add_argument("--high_thresh",          type=float, default=0.3,  help="Detection conf for primary match")
    p.add_argument("--low_thresh",           type=float, default=None, help="Detection conf for low-conf rescue (defaults to --conf)")
    p.add_argument("--max_lost_age",         type=int,   default=30,   help="Frames a lost track survives before retiring")
    p.add_argument("--min_hits_to_confirm",  type=int,   default=3,    help="Consecutive matches before track is confirmed")
    p.add_argument("--iou_threshold",        type=float, default=0.3,  help="Min IoU for a valid match")
    p.add_argument("--appearance_threshold", type=float, default=0.4,  help="Cosine distance veto threshold")
    p.add_argument("--lambda_iou",           type=float, default=0.98, help="IoU weight in combined cost (lower = more appearance)")
    p.add_argument("--no_cmc",               action="store_true", help="Disable Camera Motion Compensation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stable colour per track ID
# ---------------------------------------------------------------------------

def color_for_id(track_id: int):
    """Deterministic vivid BGR color from track ID via golden-ratio hue."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_overlay(frame, tracks, track_to_species, frame_idx, total_unique):
    out = frame.copy()
    h, w = out.shape[:2]

    for t in tracks:
        x, y, bw, bh = [int(v) for v in t.bbox]
        x2, y2 = x + bw, y + bh
        color = color_for_id(t.track_id)

        cv2.rectangle(out, (x, y), (x2, y2), color, 2)

        species = track_to_species.get(t.track_id, "...")
        label = f"#{t.track_id} {species}"
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
    yolo = YOLO(args.detection_ckpt)
    backbone = ViTBackbone(freeze=True).to(device).eval()
    mceam = MCEAM(embed_dim=768, num_context_levels=3, output_dim=256, num_heads=8).to(device).eval()

    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=True)
    num_classes = ckpt["head"]["weight"].shape[0]
    head = nn.Linear(256, num_classes).to(device).eval()
    mceam.load_state_dict(ckpt["mceam"])
    head.load_state_dict(ckpt["head"])

    _, idx_to_sp = build_species_mapping(args.csv_path, args.min_samples)
    for i in range(len(idx_to_sp), num_classes):
        idx_to_sp[i] = f"unknown_{i}"

    harvester = ContextHarvester(target_resolution=224, small_object_threshold=0.05)
    tracker = BoTSORTTracker(
        high_thresh=args.high_thresh,
        low_thresh=args.low_thresh if args.low_thresh is not None else args.conf,
        max_lost_age=args.max_lost_age,
        min_hits_to_confirm=args.min_hits_to_confirm,
        iou_threshold=args.iou_threshold,
        appearance_threshold=args.appearance_threshold,
        lambda_iou=args.lambda_iou,
        enable_cmc=not args.no_cmc,
    )

    # WaterNet — applied to each frame to match the training distribution.
    # Both detector and classifier were trained on WaterNet-restored frames,
    # so skipping this step causes a distribution shift that degrades accuracy.
    waternet = None
    if not args.no_waternet:
        print("Loading WaterNet (spectral restoration)...")
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
    cap = cv2.VideoCapture(args.video)
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
        bboxes, confs, _ = detect_frame(yolo, frame, args.conf)

        # 3. Embed + classify (on restored frame)
        species_per_det = []
        if len(bboxes) > 0:
            with torch.no_grad(), torch.amp.autocast("cuda"):
                emb_np = extract_embeddings(backbone, mceam, harvester, frame, bboxes, device)
                if len(emb_np) > 0:
                    logits = head(torch.from_numpy(emb_np).float().to(device))
                    pred_idx = logits.argmax(dim=1).cpu().numpy()
                    species_per_det = [idx_to_sp.get(int(i), "?") for i in pred_idx]
        else:
            emb_np = np.empty((0, 256), dtype=np.float64)

        # 3. Track
        active = tracker.update(bboxes, confs, emb_np, frame=frame)

        # 4. Aggregate species votes per track ID (majority over track lifetime)
        # Match active tracks back to detections by IoU to assign species labels
        if active and species_per_det:
            track_boxes = np.array([t.bbox for t in active], dtype=np.float64)
            for ti, t in enumerate(active):
                # Find best-matching detection for this track
                ious = []
                for db in bboxes:
                    ax1, ay1, ax2, ay2 = t.bbox[0], t.bbox[1], t.bbox[0]+t.bbox[2], t.bbox[1]+t.bbox[3]
                    bx1, by1, bx2, by2 = db[0], db[1], db[0]+db[2], db[1]+db[3]
                    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
                    iy = max(0, min(ay2, by2) - max(ay1, by1))
                    inter = ix * iy
                    union = t.bbox[2]*t.bbox[3] + db[2]*db[3] - inter
                    ious.append(inter / union if union > 0 else 0.0)
                if ious and max(ious) > 0.3:
                    best = int(np.argmax(ious))
                    track_to_species_votes[t.track_id][species_per_det[best]] += 1

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

        # Save keyframe
        if frame_idx % args.keyframe_every == 0 and len(active) > 0:
            kf_path = os.path.join(keyframe_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(kf_path, annotated)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
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
