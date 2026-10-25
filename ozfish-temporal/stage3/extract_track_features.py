"""Frozen D6a features for every labelled track box (Stage 3 input) + the frame-60 baseline boxes.

For each clip in labels/tracks.csv, decodes the video once and runs D6a (exact deployment
code: _92_detect.extract_embeddings) on
  * every track box on frames 0, stride, 2*stride, ... and always frame 60
  * the original expert box(es) on frame 60 (phase0/baseline_boxes.csv) -> the D6a baseline
Writes features/<clip>.npz: kind (0 track / 1 baseline), track, frame, emb (N,256) f16,
logits (N,321) f32 raw (no logit adjustment). Resumable: existing .npz are skipped.

No HF token needed: the ViT-H+ architecture is built from its published config and every
weight comes from the D6a checkpoint (strict load), so nothing is downloaded.

    python extract_track_features.py --stride 4
    python extract_track_features.py --clips ~/Desktop/Senior_Group/stage3_clips --ckpt ~/Desktop/Senior_Group/models/d6a_stage1.pt
"""
import argparse
import collections
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CLIPS = REPO / "data" / "results" / "stage3_clips"
CENTRE = 60

# facebook/dinov3-vith16plus-pretrain-lvd1689m config.json (gated repo); shapes are
# re-checked by the strict state_dict load, the rest by the frame-60 sanity number.
VITH16PLUS = dict(hidden_size=1280, num_hidden_layers=32, num_attention_heads=20,
                  intermediate_size=5120, use_gated_mlp=True, hidden_act="silu",
                  num_register_tokens=4, patch_size=16, image_size=224, layer_norm_eps=1e-5,
                  rope_theta=100.0, query_bias=True, key_bias=False, value_bias=True,
                  proj_bias=True, mlp_bias=True, layerscale_value=1.0, drop_path_rate=0.0,
                  pos_embed_rescale=2.0)


def offline_vith():
    """Make AutoModel.from_pretrained build ViT-H+ locally instead of downloading it."""
    import transformers
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    real = transformers.AutoModel.from_pretrained

    def fake(name, *a, **kw):
        if "vith16plus" in name:
            return DINOv3ViTModel(DINOv3ViTConfig(**VITH16PLUS))
        return real(name, *a, **kw)
    transformers.AutoModel.from_pretrained = fake
    import bioreef._2_stage1._25_classifier.backbone as bb
    bb.AutoModel.from_pretrained = fake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=str(HERE / "labels" / "tracks.csv"))
    ap.add_argument("--boxes", default=str(HERE / "phase0" / "baseline_boxes.csv"))
    ap.add_argument("--ckpt", default=str(REPO / "models" / "d6a_stage1.pt"))
    ap.add_argument("--clips", default=str(CLIPS), help="stage3_clips folder")
    ap.add_argument("--out", default=str(HERE / "features"))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(HERE))
    import cv2
    import torch
    offline_vith()
    from d6a_baseline import load_d6a
    from bioreef._9_pipeline._92_detect import extract_embeddings

    tracks = collections.defaultdict(lambda: collections.defaultdict(list))   # path -> frame -> boxes
    for r in csv.DictReader(open(a.tracks, newline="")):
        f = int(r["frame"])
        if r["outside"] == "0" and (f % a.stride == 0 or f == CENTRE):
            tracks[r["path"]][f].append((0, int(r["track"]), r))
    for r in csv.DictReader(open(a.boxes, newline="")):
        if r["path"] in tracks:
            tracks[r["path"]][CENTRE].append((1, int(r["box_index"]), r))
    paths = sorted(tracks)[: a.limit or None]

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    clf, harvester, idx_to_sp, C = load_d6a(a.ckpt, "cpu")
    clf.to(torch.bfloat16).to(dev)
    t0, nbox = time.time(), 0
    for i, p in enumerate(paths):
        dst = Path(a.out) / (Path(p).stem + ".npz")
        if dst.exists():
            continue
        want = tracks[p]
        cap = cv2.VideoCapture(str(Path(a.clips) / p), cv2.CAP_FFMPEG)
        kind, trk, frm, embs, logs = [], [], [], [], []
        f = -1
        last = max(want)
        while f < last:
            ok, img = cap.read()
            f += 1
            if not ok:
                print("short clip", p, "at frame", f, flush=True)
                break
            if f not in want:
                continue
            bs = want[f]
            xywh = np.array([[float(r["x0"]), float(r["y0"]), float(r["x1"]) - float(r["x0"]),
                              float(r["y1"]) - float(r["y0"])] for _, _, r in bs])
            with torch.inference_mode():
                e, _, lg = extract_embeddings(clf.backbone, clf.mceam, clf.head, harvester,
                                              img, xywh, dev)
            for (k, t, _), ee, ll in zip(bs, e, lg):
                kind.append(k); trk.append(t); frm.append(f); embs.append(ee); logs.append(ll)
        cap.release()
        np.savez(dst, kind=np.array(kind, np.int8), track=np.array(trk, np.int32),
                 frame=np.array(frm, np.int16), emb=np.array(embs, np.float16),
                 logits=np.array(logs, np.float32))
        nbox += len(kind)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{len(paths)} clips  {nbox} boxes  {nbox / el:.1f} box/s  {el / 60:.1f} min",
                  flush=True)
    print("done", nbox, "boxes")


if __name__ == "__main__":
    main()
