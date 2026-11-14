"""Pre-label the selected frames with RF-DETR CFD (low conf) + Grounding DINO ("fish."), merged,
and pack them as CVAT image tasks of --task_size frames each (#30).

Union, not either alone: each misses fish the other finds, and the reviewer's slow job is ADDING
missed fish, so fewer misses = faster labelling. False boxes are quick to delete.

    python prelabel_frames.py                     # on the VM, after select_frames.py
Writes cvat_tasks/<split>_NNN/{images.zip, annotations.xml}: in CVAT create a task with the zip
as data (label `fish`), then Actions -> Upload annotations -> "CVAT 1.1" with the xml.
Also adds n_rfdetr / n_gdino / n_merged columns to frames/manifest_prelabel.csv.
"""
import argparse
import csv
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

import torch
from PIL import Image
from torchvision.ops import nms

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def gdino_boxes(model, proc, img, thr, device):
    inputs = proc(images=img, text="fish.", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    kw = dict(text_threshold=thr, target_sizes=[img.size[::-1]])
    try:
        res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=thr, **kw)[0]
    except TypeError:  # older transformers call it box_threshold
        res = proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=thr, **kw)[0]
    b, s = res["boxes"].float().cpu(), res["scores"].float().cpu()
    # GDINO likes to box the whole scene / a whole school: drop boxes over 40% of the image
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = area < 0.4 * img.size[0] * img.size[1]
    return b[keep], s[keep]


def cvat_xml(entries, label):
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
    for i, (name, w, h, boxes) in enumerate(entries):
        lines.append(f'  <image id="{i}" name={quoteattr(name)} width="{w}" height="{h}">')
        for x0, y0, x1, y1 in boxes:
            lines.append(f'    <box label={quoteattr(label)} source="auto" occluded="0" '
                         f'xtl="{x0:.2f}" ytl="{y0:.2f}" xbr="{x1:.2f}" ybr="{y1:.2f}" z_order="0"></box>')
        lines.append("  </image>")
    lines.append("</annotations>")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(HERE / "frames"))
    ap.add_argument("--out", default=str(HERE / "cvat_tasks"))
    ap.add_argument("--rfdetr_conf", type=float, default=0.2)
    ap.add_argument("--gdino_thr", type=float, default=0.3)
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--task_size", type=int, default=100)
    ap.add_argument("--label", default="fish")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))
    from bioreef._2_stage1._21_detector import build_detector
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    det = build_detector("rfdetr", weights=str(REPO / "models" / "rfdetr_medium_cfd.pth"), device=args.device)
    proc = AutoProcessor.from_pretrained(args.gdino)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino).to(args.device).eval()

    frames = Path(args.frames)
    rows = list(csv.DictReader(open(frames / "manifest.csv")))
    out = Path(args.out)
    per_split = {}
    for k, r in enumerate(rows, 1):
        img = Image.open(frames / r["split"] / r["image"]).convert("RGB")
        d = det.predict(img, conf=args.rfdetr_conf)
        rb, rs = torch.tensor(d.xyxy, dtype=torch.float32).reshape(-1, 4), torch.tensor(d.conf, dtype=torch.float32)
        gb, gs = gdino_boxes(gd, proc, img, args.gdino_thr, args.device)
        boxes, scores = torch.cat([rb, gb]), torch.cat([rs, gs])
        boxes = boxes[nms(boxes, scores, args.nms_iou)] if len(boxes) else boxes
        r.update(n_rfdetr=len(rb), n_gdino=len(gb), n_merged=len(boxes))
        per_split.setdefault(r["split"], []).append((r["image"], *img.size, boxes.tolist()))
        if k % 50 == 0:
            print(f"  {k}/{len(rows)}")

    for split, entries in per_split.items():
        for t in range(0, len(entries), args.task_size):
            chunk = entries[t:t + args.task_size]
            task = out / f"{split}_{t // args.task_size:03d}"
            task.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(task / "images.zip", "w") as z:  # jpgs: already compressed
                for name, *_ in chunk:
                    z.write(frames / split / name, name)
            (task / "annotations.xml").write_text(cvat_xml(chunk, args.label), encoding="utf-8")
        print(f"{split}: {len(entries)} frames -> {-(-len(entries) // args.task_size)} CVAT tasks")

    with open(frames / "manifest_prelabel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    empty = sum(r["n_merged"] == 0 for r in rows)
    print(f"boxes/frame: rfdetr {sum(r['n_rfdetr'] for r in rows) / len(rows):.1f}, "
          f"gdino {sum(r['n_gdino'] for r in rows) / len(rows):.1f}, "
          f"merged {sum(r['n_merged'] for r in rows) / len(rows):.1f}; "
          f"{empty} frames with no box (expert says a fish is there: check those first)")


if __name__ == "__main__":
    main()
