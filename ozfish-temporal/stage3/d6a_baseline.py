"""
Stage 3, Phase 0 step 6: the D6a single-frame baseline.

D6a (Stage 1, unchanged) classifies the expert's box(es) on frame 60 of every testable clip:
one still frame, no movement. This is the number the temporal model has to beat.

It uses the pipeline's own code (Junior-Project: the D6a loader from _9_pipeline/models.py and
_92_detect.extract_embeddings), so D6a is scored exactly as it runs in deployment.

Scoring
  * Per clip: a clip with several expert boxes counts once (mean over its boxes).
  * Two decision rules:
      all321   argmax over all 321 D6a classes (what deployment does)
      in113    argmax over the 113 Stage 3 species only (the fair comparison for a
               113-class temporal model)
  * Each with and without logit adjustment (tau = 0 and tau = 1, the deployed setting).
  * Macro = mean of per-species clip accuracy over the 113. Also abundant / rare, per fold.

Sanity check built in: the testable clips are D6a's own val/test deployments, so top-1 on
the d6a_split == "test" clips should be close to D6a's test top-1 (0.876, all 321 classes,
tau 0). Much lower means frame decoding or preprocessing differs from training.

Run on the VM (needs the GPU, HF_TOKEN for the gated DINOv3 ViT-H+ config, and a
Junior-Project checkout for the bioreef package):

    export HF_TOKEN=...
    python d6a_baseline.py \\
        --repo     ~/Desktop/Senior_Group/Junior-Project \\
        --clips    ~/Desktop/Senior_Group/stage3_clips \\
        --boxes    ~/Desktop/Senior_Group/stage3/baseline_boxes.csv \\
        --ckpt     ~/Desktop/Senior_Group/models/d6a_stage1.pt \\
        --prior    ~/Desktop/Senior_Group/models/d6a_prior.json \\
        --out      ~/Desktop/Senior_Group/stage3/baseline

Writes <out>/baseline_per_box.csv, <out>/baseline_per_clip.csv and <out>/baseline_summary.txt.
Resumable: clips already in baseline_per_box.csv are skipped.
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np

CENTRE = 60


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Junior-Project checkout (for bioreef)")
    ap.add_argument("--clips", required=True, help="stage3_clips folder")
    ap.add_argument("--boxes", required=True, help="baseline_boxes.csv")
    ap.add_argument("--ckpt", required=True, help="d6a_stage1.pt")
    ap.add_argument("--prior", required=True, help="d6a_prior.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau", type=float, default=1.0, help="logit-adjustment tau to report")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke test)")
    return ap.parse_args()


def load_d6a(ckpt_path, device):
    """Same steps as bioreef._9_pipeline.models.load_models for the research-repo format."""
    import torch
    from bioreef._2_stage1._25_classifier import Classifier, ModelConfig
    from bioreef._1_preprocess._12_context import ContextHarvester

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state, rc = ck["model"], ck.get("run_config", {}) or {}
    num_classes = state["head.weight"].shape[0]
    mcfg = ModelConfig(
        backbone=rc.get("backbone", "dinov3"),
        context_levels=rc.get("context_levels", 3),
        attention_depth=rc.get("attention_depth", 1),
        unfreeze_blocks=0,
        probe=rc.get("probe", "mlp"),
    )
    clf = Classifier(mcfg, num_classes).to(device)
    clf.load_state_dict(state)          # strict
    clf.eval()
    idx_to_sp = {int(k): v for k, v in ck["idx_to_sp"].items()}
    return clf, ContextHarvester(), idx_to_sp, num_classes


def load_prior(path, idx_to_sp, num_classes):
    blob = json.load(open(path, encoding="utf-8"))
    prior = np.asarray(blob["prior"], dtype=np.float64)
    assert prior.shape[0] == num_classes, "prior/head class count differ"
    stored = blob.get("idx_to_sp")
    if stored:
        stored = {int(k): v for k, v in stored.items()}
        bad = [i for i in range(num_classes) if stored.get(i) != idx_to_sp.get(i)]
        assert not bad, "prior class order != checkpoint (first bad index %d)" % bad[0]
    return prior / prior.sum()


def frame60(path):
    import cv2
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)     # MJPEG needs FFMPEG
    if not cap.isOpened():
        return None
    fr = None
    for i in range(CENTRE + 1):                       # read sequentially, 0-based
        ok, f = cap.read()
        if not ok:
            break
        if i == CENTRE:
            fr = f
    cap.release()
    return fr


def summarise(per_clip, groups, folds, tau):
    lines = []
    rules = [("all321", "tau0"), ("all321", "tau%g" % tau), ("in113", "tau0"), ("in113", "tau%g" % tau)]

    def acc(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    def macro(rows, key):
        by = collections.defaultdict(list)
        for r in rows:
            by[r["species"]].append(r[key])
        return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")

    lines.append("D6a single-frame baseline on %d testable clips (per-clip scoring)" % len(per_clip))
    lines.append("")
    hdr = "%-16s %8s %8s %8s %8s %8s" % ("rule", "top1", "macro", "abundant", "rare", "top5")
    lines.append(hdr)
    for space, t in rules:
        k = "%s_%s" % (space, t)
        ab = [r for r in per_clip if groups[r["species"]] == "abundant"]
        ra = [r for r in per_clip if groups[r["species"]] == "rare"]
        lines.append("%-16s %8.4f %8.4f %8.4f %8.4f %8.4f" % (
            k, acc(per_clip, k), macro(per_clip, k), macro(ab, k), macro(ra, k), acc(per_clip, k + "_top5")))
    lines.append("")
    lines.append("sanity: all321_tau0 top1 on D6a 'test' deployments = %.4f (D6a test top1 = 0.876)"
                 % acc([r for r in per_clip if r["d6a_split"] == "test"], "all321_tau0"))
    lines.append("        all321_tau0 top1 on D6a 'val'  deployments = %.4f"
                 % acc([r for r in per_clip if r["d6a_split"] == "val"], "all321_tau0"))
    lines.append("")
    lines.append("per fold, in113_tau0 macro (the Stage 3 comparison number):")
    for f in sorted(folds):
        rows = [r for r in per_clip if r["fold"] == f]
        lines.append("  fold %s: n=%4d  top1 %.4f  macro %.4f" % (
            f, len(rows), acc(rows, "in113_tau0"), macro(rows, "in113_tau0")))
    return "\n".join(lines)


def main():
    a = parse()
    sys.path.insert(0, os.path.expanduser(a.repo))
    import torch
    from bioreef._9_pipeline._92_detect import extract_embeddings

    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    out_box = os.path.join(os.path.expanduser(a.out), "baseline_per_box.csv")

    boxes = list(csv.DictReader(open(os.path.expanduser(a.boxes), newline="")))
    by_clip = collections.OrderedDict()
    for b in boxes:
        by_clip.setdefault(b["path"], []).append(b)
    paths = list(by_clip)
    if a.limit:
        paths = paths[:a.limit]

    device = torch.device(a.device)
    clf, harvester, idx_to_sp, C = load_d6a(os.path.expanduser(a.ckpt), device)
    prior = load_prior(os.path.expanduser(a.prior), idx_to_sp, C)
    log_prior = np.log(prior + 1e-12)
    sp_to_idx = {v: k for k, v in idx_to_sp.items()}
    stage3 = sorted({b["species"] for b in boxes})
    missing = [s for s in stage3 if s not in sp_to_idx]
    assert not missing, "species not in D6a: %s" % missing[:5]
    mask113 = np.full(C, -np.inf)
    mask113[[sp_to_idx[s] for s in stage3]] = 0.0

    done = set()
    if os.path.exists(out_box):
        done = {r["path"] for r in csv.DictReader(open(out_box, newline=""))}
    cols = ["path", "species", "group", "fold", "d6a_split", "box_index"]
    keys = []
    for space in ("all321", "in113"):
        for t in ("tau0", "tau%g" % a.tau):
            keys += ["%s_%s" % (space, t), "%s_%s_top5" % (space, t), "%s_%s_pred" % (space, t)]
    fh = open(out_box, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols + keys + ["p_true_tau0"])
    if not done:
        w.writeheader()

    t0, n = time.time(), 0
    for p in paths:
        if p in done:
            continue
        fr = frame60(os.path.join(os.path.expanduser(a.clips), p))
        if fr is None:
            print("could not decode", p, flush=True)
            continue
        bs = by_clip[p]
        xywh = np.array([[float(b["x0"]), float(b["y0"]),
                          float(b["x1"]) - float(b["x0"]), float(b["y1"]) - float(b["y0"])]
                         for b in bs])
        with torch.inference_mode():
            _, _, logits = extract_embeddings(
                clf.backbone, clf.mceam, clf.head, harvester, fr, xywh, device)
        logits = logits.astype(np.float64)
        for b, lg in zip(bs, logits):
            y = sp_to_idx[b["species"]]
            row = {c: b[c] for c in cols}
            variants = {"tau0": lg, "tau%g" % a.tau: lg - a.tau * log_prior}
            for t, v in variants.items():
                for space, s in (("all321", v), ("in113", v + mask113)):
                    order = np.argsort(-s)
                    k = "%s_%s" % (space, t)
                    row[k] = int(order[0] == y)
                    row[k + "_top5"] = int(y in order[:5])
                    row[k + "_pred"] = idx_to_sp[int(order[0])]
            pr = np.exp(lg - lg.max())
            row["p_true_tau0"] = round(float(pr[y] / pr.sum()), 5)
            w.writerow(row)
        fh.flush()
        n += 1
        if n % 100 == 0:
            print("%d/%d clips  %.0fs" % (n + len(done), len(paths), time.time() - t0), flush=True)
    fh.close()

    # per-clip table (mean over a clip's boxes) and summary
    rows = list(csv.DictReader(open(out_box, newline="")))
    score_keys = [k for k in keys if not k.endswith("_pred")]
    clip = collections.OrderedDict()
    for r in rows:
        clip.setdefault(r["path"], []).append(r)
    per_clip = []
    for p, rs in clip.items():
        d = {c: rs[0][c] for c in ("path", "species", "group", "fold", "d6a_split")}
        d["n_boxes"] = len(rs)
        for k in score_keys:
            d[k] = float(np.mean([float(r[k]) for r in rs]))
        per_clip.append(d)
    with open(os.path.join(os.path.expanduser(a.out), "baseline_per_clip.csv"), "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(per_clip[0]))
        cw.writeheader()
        cw.writerows(per_clip)
    groups = {r["species"]: r["group"] for r in per_clip}
    text = summarise(per_clip, groups, {r["fold"] for r in per_clip}, a.tau)
    open(os.path.join(os.path.expanduser(a.out), "baseline_summary.txt"), "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
