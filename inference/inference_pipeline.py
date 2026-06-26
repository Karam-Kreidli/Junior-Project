"""
BioReef.ai — Inference pipeline entry point.

Runs the full chain on one clip: preprocess -> Stage 1 -> Stage 2 -> (Stage 3
stub) -> outputs. All settings come from the config file; the only argument is
its path.

    python inference/inference_pipeline.py                 # uses ./config.yaml
    python inference/inference_pipeline.py --config my.yaml

Set the clip and knobs under `inference:` (and shared paths under `shared:`) in
config.yaml. Preprocessing is the shared bioreef._1_preprocess._17_preprocess used by both
training and inference.
"""

import argparse
import logging
import os
import sys

# repo-root bootstrap: this file lives in inference/ (one level down).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bioreef._9_pipeline.config import InferenceConfig, DEFAULT_CONFIG_PATH
from bioreef._9_pipeline._95_runner import run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help=f"Pipeline config YAML. Default: {DEFAULT_CONFIG_PATH}")
    args = p.parse_args()

    cfg = InferenceConfig.from_yaml(args.config)
    run_inference(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
