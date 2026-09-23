"""
Build a single flat record of every test-set metric from every run.

Walks the archived per-seed `metrics.json` files -- every run in the panel, both
the C/A paper configs and the D deployment ladder -- and emits one CSV row per
run per seed, carrying the full metric panel plus the config fields that
distinguish the runs (backbone, lr, loss, unfreeze depth, batch). Per-seed, not
averaged: RESULTS.md already has the mean +/- std view. The chosen post-hoc
operating point is appended as its own row, since it is a real deployment result
that exists in no metrics.json -- it comes from the offline sweep, not a
training run.

    python scripts/build_metrics_record.py

Writes docs/classifier-research/all_test_metrics.csv.
"""

import csv
import glob
import json
import os

ARCHIVE = os.path.join("docs", "classifier-research")
RESULTS = os.path.join(ARCHIVE, "results")
OUT = os.path.join(ARCHIVE, "all_test_metrics.csv")

COLUMNS = [
    "run_id", "slug", "seed", "posthoc", "tau",
    "backbone", "lr", "loss", "unfreeze_blocks", "batch_size", "hslm",
    "layer_decay",
    "macro_accuracy", "top1_accuracy", "top5_accuracy",
    "head", "medium", "tail",
    "mean_hd", "mistake_severity",
    "genus_accuracy", "family_accuracy", "cross_family_error_rate",
    "code_revision", "source",
]

# The deployment operating point: D6a seed 1 with post-hoc logit adjustment at
# tau=1.0. Produced by tail_posthoc_sweep.py --save_json, so it has no
# metrics.json of its own, but it IS the model that ships -- it belongs in the
# record. tau was selected on test, so this is a deployment number, not a clean
# held-out one.
POSTHOC_ROW = {
    "run_id": "D6a", "slug": "D6a_dinov3_huge_lr2p5e6", "seed": 1,
    "posthoc": "logit_adj", "tau": 1.0,
    "backbone": "dinov3_huge", "lr": 2.5e-06, "loss": "ce",
    "unfreeze_blocks": -1, "batch_size": 8, "hslm": True, "layer_decay": "",
    "macro_accuracy": 0.7781128582761858,
    "top1_accuracy": 0.871152314043601,
    "top5_accuracy": 0.9606721228362425,
    "head": 0.880156948771656,
    "medium": 0.7406602757732923,
    "tail": 0.6722222222222222,
    "mean_hd": 0.21300789454624466,
    "mistake_severity": 1.6531759415401912,
    "genus_accuracy": 0.9458970087636706,
    "family_accuracy": 0.9699427826464837,
    "cross_family_error_rate": 0.03005721735351633,
    "code_revision": "",
    "source": "tail_posthoc_sweep.py --method logit_adj --tau 1.0",
}


def row_from_metrics(path):
    with open(path, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    test = m.get("test", {})
    groups = test.get("group_accuracy", {}) or {}
    run_cfg = (m.get("provenance", {}) or {}).get("run_config", {}) or {}
    return {
        "run_id": m.get("run_id", ""),
        "slug": m.get("slug", ""),
        "seed": m.get("seed", ""),
        "posthoc": "",
        "tau": "",
        "backbone": run_cfg.get("backbone", ""),
        "lr": run_cfg.get("lr", ""),
        "loss": run_cfg.get("loss", ""),
        "unfreeze_blocks": run_cfg.get("unfreeze_blocks", ""),
        "batch_size": run_cfg.get("batch_size", ""),
        "hslm": run_cfg.get("hslm", ""),
        "layer_decay": run_cfg.get("layer_decay", "") or "",
        "macro_accuracy": test.get("macro_accuracy"),
        "top1_accuracy": test.get("top1_accuracy"),
        "top5_accuracy": test.get("top5_accuracy"),
        "head": groups.get("head"),
        "medium": groups.get("medium"),
        "tail": groups.get("tail"),
        "mean_hd": test.get("mean_hd"),
        "mistake_severity": test.get("mistake_severity"),
        "genus_accuracy": test.get("genus_accuracy"),
        "family_accuracy": test.get("family_accuracy"),
        "cross_family_error_rate": test.get("cross_family_error_rate"),
        "code_revision": m.get("code_revision", ""),
        "source": path.replace(os.sep, "/"),
    }


def main():
    paths = sorted(glob.glob(os.path.join(RESULTS, "*", "seed*", "metrics.json")))
    if not paths:
        raise SystemExit(f"no metrics.json under {RESULTS}")

    rows = [row_from_metrics(p) for p in paths]
    rows.append(POSTHOC_ROW)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT}  ({len(rows)} rows)")
    for r in rows:
        tag = f" +{r['posthoc']} tau={r['tau']}" if r["posthoc"] else ""
        print(f"  {r['run_id']:5s} seed{r['seed']}{tag:22s} "
              f"macro={r['macro_accuracy']:.4f} top1={r['top1_accuracy']:.4f} "
              f"tail={r['tail']:.4f}")


if __name__ == "__main__":
    main()
