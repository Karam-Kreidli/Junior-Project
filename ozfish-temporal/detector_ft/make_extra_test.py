"""Second detector test set: Kheiro's frames (train_004/005/006/008) from drops ft1a never saw.

ft1a trained on dataset/ (6 tasks). dataset_all/ is built from all 10 returned tasks. Frames in
dataset_all's train/valid that are not in ft1a's train/valid are Kheiro's; of those, keep only frames whose camera
drop has no frame in ft1a's train/valid. Writes dataset_extra/test/ (images + COCO json) so
eval_detector.py can score it like the main test set:

    python build_dataset.py --labels returned/ --out dataset_all
    python make_extra_test.py
    python eval_detector.py --dataset dataset_extra --weights ../../models/rfdetr_medium_ft1a.pth --name ft1a_extra
    python eval_detector.py --dataset dataset_extra --name cfd_extra
"""
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
A, B, OUT = HERE / "dataset", HERE / "dataset_all", HERE / "dataset_extra" / "test"


def drop(name):
    return re.split(r"_[LR]\.", name)[0]


def coco(split_dir):
    return json.loads((split_dir / "_annotations.coco.json").read_text())


seen = {im["file_name"] for s in ("train", "valid") for im in coco(A / s)["images"]}
seen_drops = {drop(n) for n in seen}

OUT.mkdir(parents=True, exist_ok=True)
images, anns, cats = [], [], None
for s in ("train", "valid"):
    c = coco(B / s)
    cats = c["categories"]
    keep = {im["id"]: im for im in c["images"] if im["file_name"] not in seen and drop(im["file_name"]) not in seen_drops}
    remap = {old: len(images) + i for i, old in enumerate(keep)}
    for old, im in keep.items():
        images.append({**im, "id": remap[old]})
        shutil.copy2(B / s / im["file_name"], OUT / im["file_name"])
    for a in c["annotations"]:
        if a["image_id"] in keep:
            anns.append({**a, "id": len(anns), "image_id": remap[a["image_id"]]})

(OUT / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
assert not {drop(im["file_name"]) for im in images} & seen_drops
print(f"dataset_extra/test: {len(images)} frames, {len(anns)} fish, {len({drop(im['file_name']) for im in images})} drops "
      f"(none shared with ft1a train/valid)")
