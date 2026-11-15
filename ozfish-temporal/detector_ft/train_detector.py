"""Fine-tune RF-DETR Medium from the CFD checkpoint on dataset/ (from build_dataset.py). VM only.

Gentle on purpose: CFD learned from many fish datasets and a small OzFish set can make it forget
that, so low LR and few epochs, keep the checkpoint with the best valid mAP (EMA).

    python train_detector.py                                  # real run
    python train_detector.py --dataset dataset_smoke --epochs 1 --out runs/smoke   # pipeline check
Best weights: <out>/checkpoint_best_ema.pth (or checkpoint_best_regular.pth). Score it with
eval_detector.py before putting it in the pipeline.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset"))
    ap.add_argument("--out", default=str(HERE / "runs" / "ft1"))
    ap.add_argument("--weights", default=str(REPO / "models" / "rfdetr_medium_cfd.pth"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_encoder", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    args = ap.parse_args()

    from rfdetr import RFDETRMedium
    model = RFDETRMedium(pretrain_weights=args.weights, resolution=1024)
    model.train(dataset_dir=args.dataset, output_dir=args.out, epochs=args.epochs, lr=args.lr,
                lr_encoder=args.lr_encoder, batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps, early_stopping=True, early_stopping_patience=5,
                class_names=["fish"])


if __name__ == "__main__":
    main()
