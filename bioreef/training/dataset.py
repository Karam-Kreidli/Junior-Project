"""Stage-1 training dataset: per-sample restore -> augment -> 4-stream crops."""

import numpy as np
from torch.utils.data import Dataset

from bioreef._1_preprocess._12_context import ContextHarvester
from bioreef._1_preprocess._11_restoration import WaterNetRestorer
from bioreef._1_preprocess._13_augmentation import MarineAugmentor
from bioreef.training.ddp import safe_imread


class Stage1Dataset(Dataset):
    def __init__(self, samples, img_dir, is_train=True, use_waternet=False):
        self.samples = samples
        self.img_dir = img_dir
        self.harvester = ContextHarvester(target_resolution=224, small_object_threshold=0.05)
        self.restorer = WaterNetRestorer() if use_waternet else None
        self.augmentor = MarineAugmentor(enabled=is_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frame = safe_imread(s["img_path"])
        if frame is None:
            frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        if self.restorer is not None:
            frame = self.restorer(frame)
        augmented = self.augmentor(frame)
        streams = self.harvester.harvest(augmented, s["bbox"])
        return {"streams": streams, "label": s["class_idx"], "species": s["species"]}
