"""
BioReef.ai — Video Demo, drawn from the real pipeline
=====================================================
Runs the pipeline's own stages on one video and renders what they report, so the
video shows exactly what the system outputs (no demo-only logic):

    load_models -> run_stage1 (detect, track, classify each fish every
    --classify_every frames) -> run_stage2 (track + one hierarchical verdict per
    fish) -> render boxes, IDs and verdicts

Frames are read in memory (the pipeline's preprocess step would write every frame
to disk as PNG next to the video). Each fish is labelled with its final verdict
over the whole clip: species, or genus/family when the evidence is weaker.

Writes to --out_dir:
    demo_annotated.mp4   the annotated video
    fish.json            one verdict per fish (track)
    summary.json         per-species MaxN + richness / Shannon / Simpson
    keyframes/*.png      a few stills

Usage:
    python scripts/viz/demo_video.py --video clip.mp4 --out_dir demo_out
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import cv2
from tqdm import tqdm
# --- repo-root bootstrap: this script lives in scripts/<area>/; add the
# repo root (two levels up) to sys.path so `import bioreef` resolves no
# matter the cwd or how the script is invoked. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from bioreef._9_pipeline.config import InferenceConfig
from bioreef._9_pipeline.models import load_models
from bioreef._9_pipeline._92_detect import run_stage1
from bioreef._9_pipeline._93_track import run_stage2
from bioreef._9_pipeline.frame_sources import VideoFrameSource
from bioreef.viz import color_for_id


def parse_args():
    d = InferenceConfig()      # pipeline defaults, so the demo can't drift from them
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", default="demo_out")
    p.add_argument("--stage1_ckpt", default="models/d6a_stage1.pt")
    p.add_argument("--stage1_prior", default="models/d6a_prior.json")
    p.add_argument("--tau", type=float, default=1.0, help="logit adjustment; 1.0 = deployed")
    p.add_argument("--csv_path", default="frame_metadata.csv", help="taxonomy for the genus/family fallback")
    p.add_argument("--conf", type=float, default=d.conf_threshold)
    p.add_argument("--classify_every", type=int, default=d.classify_every)
    p.add_argument("--keyframe_every", type=int, default=30)
    p.add_argument("--show", action="store_true", help="play the result in a window after rendering")
    return p.parse_args()


def label_for(v):
    """Verdict dict -> on-screen label."""
    if v is None or v["level"] == "unidentified":
        return "fish (unidentified)"
    name = {"species": v["name"], "genus": f"{v['name']} sp.",
            "family": f"{v['name']} (family)"}[v["level"]]
    return f"{name} {v['confidence']:.0%}"


def draw(frame, boxes, labels, frame_idx, n_fish):
    out = frame.copy()
    for tid, (x, y, w, h) in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        color = color_for_id(tid)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        text = f"#{tid} {labels[tid]}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ty = max(y - 8, th + 8)
        cv2.rectangle(out, (x, ty - th - 8), (x + tw + 8, ty + 4), color, -1)
        cv2.putText(out, text, (x + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    hud = f"BioReef.ai  |  frame {frame_idx}  |  fish in view: {len(boxes)}  |  fish tracked: {n_fish}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(out, hud, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def biodiversity(tracks, verdicts):
    """Per-species MaxN (most individuals of a species visible in one frame -- the
    BRUVS standard, never double-counts a fish that leaves and comes back) and the
    indices on those counts. Species-level verdicts only."""
    sp = {v["track_id"]: v["name"] for v in verdicts if v["level"] == "species"}
    per_frame = defaultdict(Counter)
    for t in tracks:
        if t["track_id"] in sp:
            for f, _ in t["frames"]:
                per_frame[f][sp[t["track_id"]]] += 1
    maxn = Counter()
    for c in per_frame.values():
        for s, n in c.items():
            maxn[s] = max(maxn[s], n)
    total = sum(maxn.values())
    p = [n / total for n in maxn.values()] if total else []
    return {"maxn": dict(maxn.most_common()), "richness": len(maxn),
            "shannon": round(-sum(x * math.log(x) for x in p), 4),
            "simpson": round(1 - sum(x * x for x in p), 4)}


def main():
    a = parse_args()
    os.makedirs(os.path.join(a.out_dir, "keyframes"), exist_ok=True)
    cfg = InferenceConfig(video=a.video, video_id=os.path.basename(a.video),
                          stage1_ckpt=a.stage1_ckpt, stage1_prior=a.stage1_prior,
                          stage1_logit_tau=a.tau, csv_path=a.csv_path,
                          conf_threshold=a.conf, classify_every=a.classify_every)

    print("Loading models...")
    models = load_models(cfg)
    src = VideoFrameSource(a.video)
    fps, n_frames = src.fps or 30.0, len(src)
    print(f"Stage 1: detect + track + classify ({n_frames} frames)...")
    s1 = run_stage1(tqdm(src, total=n_frames, unit="f"), models, cfg, video_id=cfg.video_id)
    print("Stage 2: track + verdicts...")
    s2 = run_stage2(s1, models, cfg)

    verdicts = s2.track_verdicts or []
    by_id = {v["track_id"]: v for v in verdicts}
    labels = {t["track_id"]: label_for(by_id.get(t["track_id"])) for t in s2.tracks}
    boxes_at = defaultdict(list)
    for t in s2.tracks:
        for f, b in t["frames"]:
            boxes_at[f].append((t["track_id"], b))

    print("Rendering...")
    src = VideoFrameSource(a.video)
    out_path = os.path.join(a.out_dir, "demo_annotated.mp4")
    writer = None
    seen = set()
    for f, frame in tqdm(src, total=n_frames, unit="f"):
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        seen.update(tid for tid, _ in boxes_at[f])
        img = draw(frame, boxes_at[f], labels, f, len(seen))
        writer.write(img)
        if f % a.keyframe_every == 0 and boxes_at[f]:
            cv2.imwrite(os.path.join(a.out_dir, "keyframes", f"frame_{f:06d}.png"), img)
    writer.release()

    summary = {"video": cfg.video_id, "frames": n_frames, "fish_tracked": len(s2.tracks),
               "verdict_levels": dict(Counter(v["level"] for v in verdicts)),
               **biodiversity(s2.tracks, verdicts)}
    json.dump(verdicts, open(os.path.join(a.out_dir, "fish.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Video -> {out_path}")

    if a.show:
        cap = cv2.VideoCapture(out_path)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            cv2.imshow("BioReef.ai (q to quit)", cv2.resize(img, (1280, 720)))
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
