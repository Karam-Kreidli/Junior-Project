"""
Export the post-hoc logit-adjustment prior for a Stage-1 checkpoint.

Stage 1 can subtract `tau * log(class_prior)` from its raw logits (Menon 2020 /
Balanced Softmax) to lift rare classes without retraining. This writes the small
JSON that holds that prior, to sit beside the checkpoint and be named by
`inference.stage1_prior` in config.yaml.

The prior is the TRAIN class frequency in CLASS-INDEX order -- the same order as
the logit columns. Getting that order wrong would silently mislabel everything,
so the species mapping is written alongside it and re-checked at load time
(see _9_pipeline/models.py::_load_logit_prior).

Two sources for the counts, in order of preference:

  --dump    the posthoc_dump.npz produced by the research repo's
            scripts/dump_posthoc.py, which already carries `sp_counts`.
            Preferred: no dataset, no CSV, no recomputation.

  --csv     the OzFish metadata CSV; counts are rebuilt from the TRAIN split.
            Use only if no dump exists. Requires the research repo on PYTHONPATH
            since it must reproduce the exact split.

    python scripts/export_logit_prior.py \
        --ckpt models/d6a_stage1.pt \
        --dump posthoc_dump.npz \
        --out models/d6a_prior.json
"""

import argparse
import json
import os

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="the Stage-1 checkpoint the prior belongs to")
    ap.add_argument("--dump", default=None,
                    help="posthoc_dump.npz containing sp_counts (preferred)")
    ap.add_argument("--csv", default=None,
                    help="OzFish CSV, to rebuild counts from the train split "
                         "(fallback; needs the research repo importable)")
    ap.add_argument("--img_dir", default=None, help="frames dir, with --csv")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="tau recorded in the JSON as the recommended default "
                         "(config.yaml's stage1_logit_tau is what actually "
                         "applies; this is documentation)")
    return ap.parse_args()


def counts_from_dump(path):
    d = np.load(path, allow_pickle=True)
    if "sp_counts" not in d:
        raise SystemExit(f"{path} has no 'sp_counts' array (keys: {list(d.keys())})")
    return np.asarray(d["sp_counts"], dtype=np.float64)


def counts_from_csv(csv_path, img_dir, ckpt):
    """Rebuild train counts by reproducing the split. Needs the research repo."""
    try:
        from bioreef.config import BenchmarkConfig
        from bioreef.data import split_from_config
    except ImportError as e:
        raise SystemExit(
            f"--csv needs the research repo importable (bioreef.data.split): {e}\n"
            f"Prefer --dump, which carries sp_counts already.")
    bench_d = ckpt.get("benchmark_config", {}) or {}
    cfg = BenchmarkConfig(**{k: v for k, v in bench_d.items()
                             if k in BenchmarkConfig.__dataclass_fields__})
    *_, sp_counts = split_from_config(csv_path, img_dir, cfg)
    return np.asarray(sp_counts, dtype=np.float64)


def main():
    args = parse_args()
    if not args.dump and not args.csv:
        raise SystemExit("give --dump (preferred) or --csv")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # num_classes from the head itself — never from a stored field that could
    # disagree with the weights.
    if "model" in ckpt:
        num_classes = ckpt["model"]["head.weight"].shape[0]
    elif "head" in ckpt:
        num_classes = ckpt["head"]["weight"].shape[0]
    else:
        raise SystemExit(f"{args.ckpt}: no 'model' or 'head' key — not a Stage-1 "
                         f"checkpoint?")
    idx_to_sp = {int(k): v for k, v in (ckpt.get("idx_to_sp") or {}).items()}

    counts = (counts_from_dump(args.dump) if args.dump
              else counts_from_csv(args.csv, args.img_dir, ckpt))

    if counts.shape[0] != num_classes:
        raise SystemExit(
            f"counts have {counts.shape[0]} classes but the head has "
            f"{num_classes}. The counts must come from the same run as the "
            f"checkpoint.")
    if counts.min() <= 0:
        zero = int((counts <= 0).sum())
        raise SystemExit(
            f"{zero} class(es) have no training examples, so log(prior) is "
            f"undefined. The benchmark guarantees every class has training data "
            f"— this means the counts do not match this checkpoint.")

    prior = counts / counts.sum()
    payload = {
        "tau": args.tau,
        "num_classes": num_classes,
        "source_checkpoint": os.path.basename(args.ckpt),
        "source_counts": os.path.basename(args.dump or args.csv),
        "note": ("Train class frequencies in class-index order (= logit column "
                 "order). Stage 1 applies logits - tau*log(prior). Re-export "
                 "whenever the checkpoint changes."),
        "prior": prior.tolist(),
        "idx_to_sp": {str(i): idx_to_sp[i] for i in sorted(idx_to_sp)},
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    head_n, tail_n = int(counts.max()), int(counts.min())
    print(f"wrote {args.out}")
    print(f"  classes      : {num_classes}")
    print(f"  train counts : max {head_n}  min {tail_n}  (imbalance {head_n/tail_n:.0f}x)")
    print(f"  tau recorded : {args.tau}")
    if not idx_to_sp:
        print("  WARNING: the checkpoint carried no idx_to_sp, so the class-order "
              "cross-check at load time will be skipped.")


if __name__ == "__main__":
    main()
