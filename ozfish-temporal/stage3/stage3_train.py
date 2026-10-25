"""Stage 3 on the species labelled so far: D6a baselines vs small heads on frozen D6a features.

Evaluation = the fixed Phase 0 station folds: for fold k, test on fold-k test-eligible clips
(D6a never trained on their deployments), train on every other labelled clip (incl. fold -1).
Scoring is per clip in the space of the labelled species only (D6a's logits sliced to them).
Clip prediction = mean log-prob over its EXPERT tracks (inferred tracks are training-only).
No validation set and no tuning: hyperparameters are fixed up front, so nothing peeks at test.

  d6a_frame60      D6a on the expert box, frame 60 only (the Phase 0 baseline)
  d6a_track_mean   D6a on every sampled frame of the track, log-probs averaged (free pooling)
  s3_frame         MLP on single-frame embeddings, averaged over the track (retrained head, no order)
  s3_gru           BiGRU over windows of consecutive frames (sees movement / order)
  s3_gru_shuffled  same, frames shuffled in train AND test (control: order destroyed)
  fusion           d6a_track_mean + s3_gru log-probs, equal weight

    python stage3_train.py --seeds 3
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
W = 8              # window length in sampled frames (stride 4 -> 32 video frames, ~1.3 s)


def load(feat_dir, tracks_csv, prior_json):
    meta = {}
    for r in csv.DictReader(open(tracks_csv, newline="")):
        meta.setdefault((r["path"], int(r["track"])), r)
    species = sorted({m["species"] for m in meta.values()})
    idx_to_sp = json.load(open(prior_json))["idx_to_sp"]
    sp_to_idx = {v: int(k) for k, v in idx_to_sp.items()}
    cols = np.array([sp_to_idx[s] for s in species])
    prior = np.array(json.load(open(prior_json))["prior"])[cols]
    tracks, base = [], {}
    for (path, t), m in meta.items():
        f = Path(feat_dir) / (Path(path).stem + ".npz")
        if not f.exists():
            continue
        z = np.load(f)
        sel = (z["kind"] == 0) & (z["track"] == t)
        if sel.sum() == 0:
            continue
        o = np.argsort(z["frame"][sel])
        tracks.append(dict(path=path, y=species.index(m["species"]), fold=int(m["fold"]),
                           test=m["test_eligible"] == "1", expert=m["source"] != "inferred",
                           emb=z["emb"][sel][o].astype(np.float32),
                           logit=z["logits"][sel][o][:, cols]))
        b = z["kind"] == 1
        if b.any():
            base[path] = z["logits"][b][:, cols]
    return species, tracks, base, np.log(prior / prior.sum())


class FrameMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, 128),
                                 nn.GELU(), nn.Dropout(0.3), nn.Linear(128, c))

    def forward(self, x, lengths):                     # x (B,T,256) -> mean of per-frame logits
        m = (torch.arange(x.shape[1])[None] < lengths[:, None]).float()
        return (self.net(x) * m[..., None]).sum(1) / m.sum(1, keepdim=True)


class GRUHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, 128), nn.GELU())
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, c))

    def forward(self, x, lengths):
        h = self.inp(x)
        h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.gru(h)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        m = (torch.arange(x.shape[1])[None] < lengths[:, None]).float()
        return self.out((h * m[..., None]).sum(1) / m.sum(1, keepdim=True))


def windows(emb, rng=None, shuffle=False):
    """Random window (train) or all half-overlapping windows (test) of up to W frames."""
    T = len(emb)
    if rng is not None:
        s = rng.integers(0, max(T - W, 0) + 1)
        ws = [emb[s:s + W]]
    else:
        ws = [emb[s:s + W] for s in range(0, max(T - W, 0) + 1, W // 2)]
    if shuffle:
        r = rng or np.random.default_rng(0)
        ws = [w[r.permutation(len(w))] for w in ws]
    return ws


def batchify(ws):
    L = torch.tensor([len(w) for w in ws])
    x = torch.zeros(len(ws), int(L.max()), 256)
    for i, w in enumerate(ws):
        x[i, :len(w)] = torch.from_numpy(w)
    return x, L


def train(model, train_tracks, c, seed, shuffle, epochs=40, per_track=4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    counts = np.bincount([t["y"] for t in train_tracks], minlength=c)
    wts = torch.tensor(len(train_tracks) / (c * np.maximum(counts, 1)), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    model.train()
    for _ in range(epochs):
        items = [t for t in train_tracks for _ in range(per_track)]
        rng.shuffle(items)
        for i in range(0, len(items), 64):
            chunk = items[i:i + 64]
            x, L = batchify([windows(t["emb"], rng, shuffle)[0] for t in chunk])
            y = torch.tensor([t["y"] for t in chunk])
            loss = F.cross_entropy(model(x, L), y, weight=wts)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def track_logprob(model, t, shuffle):
    x, L = batchify(windows(t["emb"], None, shuffle))
    return F.log_softmax(model(x, L), -1).mean(0).numpy()


def lsm(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def scores(pred, y, species):
    acc = (pred == y).astype(float)
    per = collections.defaultdict(list)
    for a, s in zip(acc, y):
        per[s].append(a)
    return acc.mean(), np.mean([np.mean(v) for v in per.values()]), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(HERE / "features"))
    ap.add_argument("--tracks", default=str(HERE / "labels" / "tracks.csv"))
    ap.add_argument("--prior", default=str(HERE.parents[1] / "models" / "d6a_prior.json"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no_inferred", action="store_true", help="ablation: train on expert tracks only")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    torch.set_num_threads(8)

    species, tracks, base, logprior = load(a.features, a.tracks, a.prior)
    c = len(species)
    clips = sorted({t["path"] for t in tracks if t["test"] and t["expert"] and t["path"] in base})
    y = np.array([next(t["y"] for t in tracks if t["path"] == p) for p in clips])
    fold = np.array([next(t["fold"] for t in tracks if t["path"] == p) for p in clips])
    print(f"{c} species, {len(tracks)} tracks, {len(clips)} test clips over folds {sorted(set(fold))}")

    lp = {}                                            # method -> (n_clips, c) log-probs
    lp["d6a_frame60"] = np.stack([lsm(base[p]).mean(0) for p in clips])
    lp["d6a_frame60_tau1"] = np.stack([lsm(base[p] - logprior).mean(0) for p in clips])
    by_clip = collections.defaultdict(list)
    for t in tracks:
        if t["expert"]:
            by_clip[t["path"]].append(t)
    lp["d6a_track_mean"] = np.stack([np.mean([lsm(t["logit"]).mean(0) for t in by_clip[p]], 0)
                                     for p in clips])
    lp["d6a_track_mean_tau1"] = np.stack([np.mean([lsm(t["logit"] - logprior).mean(0)
                                                   for t in by_clip[p]], 0) for p in clips])

    variants = [("s3_frame", FrameMLP, False), ("s3_gru", GRUHead, False),
                ("s3_gru_shuffled", GRUHead, True)]
    for name, cls, shuf in variants:
        out = np.zeros((a.seeds, len(clips), c))
        for seed in range(a.seeds):
            for k in sorted(set(fold)):
                tr = [t for t in tracks if t["fold"] != k and (t["expert"] or not a.no_inferred)]
                m = train(cls(c), tr, c, seed, shuf)
                for i in np.where(fold == k)[0]:
                    out[seed, i] = np.mean([track_logprob(m, t, shuf) for t in by_clip[clips[i]]], 0)
            print(f"  {name} seed {seed} done", flush=True)
        lp[name] = out                                  # keep per-seed
    lp["fusion"] = lp["d6a_track_mean"][None] + lp["s3_gru"]

    Path(a.out).mkdir(exist_ok=True)
    lines = [f"Stage 3 on {c} labelled species: {len(clips)} test-eligible clips, 5 station folds, "
             f"{a.seeds} seeds" + (" (expert tracks only)" if a.no_inferred else ""), "",
             f"{'method':22s} {'top1':>14s} {'macro':>14s}"]
    acc_rows = {}
    for name, v in lp.items():
        v = v if v.ndim == 3 else v[None]
        r = [scores(s.argmax(1), y, species) for s in v]
        t1, mc = np.array([x[0] for x in r]), np.array([x[1] for x in r])
        acc_rows[name] = np.mean([x[2] for x in r], 0)
        pm = lambda q: f"{q.mean():.3f}" + (f" ±{q.std():.3f}" if len(q) > 1 else "      ")
        lines.append(f"{name:22s} {pm(t1):>14s} {pm(mc):>14s}")
    # paired bootstrap over clips: s3_gru vs the two D6a numbers, and order vs shuffled
    rng = np.random.default_rng(0)
    lines.append("")
    for a_, b_ in [("s3_gru", "d6a_frame60"), ("s3_gru", "d6a_track_mean"),
                   ("d6a_track_mean", "d6a_frame60"), ("s3_gru", "s3_gru_shuffled"),
                   ("fusion", "d6a_track_mean"), ("s3_frame", "d6a_track_mean")]:
        d = acc_rows[a_] - acc_rows[b_]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        lines.append(f"{a_} - {b_}: {d.mean():+.3f} top1  (95% CI {np.percentile(bs, 2.5):+.3f} "
                     f"to {np.percentile(bs, 97.5):+.3f})")
    lines.append("")
    lines.append("per species top1 (n test clips): d6a_frame60 / d6a_track_mean / s3_gru / fusion")
    for s in range(c):
        i = y == s
        if i.any():
            lines.append(f"  {species[s]:28s} n={i.sum():3d}  " + " / ".join(
                f"{acc_rows[m][i].mean():.2f}" for m in ("d6a_frame60", "d6a_track_mean", "s3_gru", "fusion")))
    text = "\n".join(lines)
    tag = "_expert_only" if a.no_inferred else ""
    open(Path(a.out) / f"summary{tag}.txt", "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
