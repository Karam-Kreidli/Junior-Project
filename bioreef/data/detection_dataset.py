"""
BioReef.ai — Detection Dataset
================================
Frame-level dataset for training the DINO detector + FDR head.

Unlike the Stage 1 classification dataset (one sample = one fish crop),
the detection dataset groups annotations by frame. Each sample is a
full-resolution image with all its bounding boxes and species labels.

Pipeline per sample:
    1. Load frame image (OpenCV BGR)
    2. Resize to detection resolution (default 512×512)
    3. Convert bboxes from pixel (x0, y0, x1, y1) → normalized (cx, cy, w, h)
    4. Apply ImageNet Z-score normalization
    5. Return {'image': Tensor, 'targets': {'labels': Tensor, 'boxes': Tensor}}
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("bioreef.data.detection")

# ImageNet normalization (mandatory for frozen DINOv3)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def safe_imread(path: str) -> Optional[np.ndarray]:
    """Read image while suppressing OpenCV warnings on corrupt files."""
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    finally:
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stderr)
    return img


# =============================================================================
# CSV Parsing — Group Annotations by Frame
# =============================================================================

def load_detection_data(
    csv_path: str,
    img_dirs: List[str],
    filter_labels: Optional[set] = None,
) -> Tuple[List[Dict], Dict[str, int], Dict[int, str]]:
    """
    Parse frame_metadata.csv into frame-level detection samples.

    Args:
        csv_path:      Path to CSV with columns:
                        uid, file_name, x0, y0, x1, y1, family, genus, species
        img_dirs:      Ordered list of directories to search for images.
        filter_labels: Species names to exclude (e.g., 'Unidentified').

    Returns:
        frames:       List of {'img_path': str, 'boxes': [[x0,y0,x1,y1],...],
                       'labels': [class_idx,...]}
        sp_to_idx:    species_name → class index mapping
        idx_to_sp:    class index → species_name mapping
    """
    if filter_labels is None:
        filter_labels = {
            "Unidentified", "Fish", "Unknown", "unidentifiable",
            "fish", "unknown", "unidentified", "other", "Other",
            "spp", "sp1", "sp2", "sp3", "sp6", "sp10",
        }

    df = pd.read_csv(csv_path).dropna(subset=['species'])
    df = df[~df['species'].isin(filter_labels)]

    # Build class mapping
    unique_sp = sorted(df['species'].unique().tolist())
    sp_to_idx = {sp: i for i, sp in enumerate(unique_sp)}
    idx_to_sp = {i: sp for sp, i in sp_to_idx.items()}

    # Resolve image paths and group by frame
    frame_dict: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        fname = row['file_name']
        if fname not in frame_dict:
            # Find image file
            img_path = ""
            for d in img_dirs:
                candidate = os.path.join(d, fname)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if not img_path:
                continue
            frame_dict[fname] = {
                'img_path': img_path,
                'boxes': [],
                'labels': [],
            }

        box = [int(row['x0']), int(row['y0']), int(row['x1']), int(row['y1'])]
        cls_idx = sp_to_idx[row['species']]
        frame_dict[fname]['boxes'].append(box)
        frame_dict[fname]['labels'].append(cls_idx)

    frames = list(frame_dict.values())
    logger.info(
        f"Detection dataset: {len(frames)} frames, "
        f"{sum(len(f['labels']) for f in frames)} annotations, "
        f"{len(unique_sp)} species"
    )
    return frames, sp_to_idx, idx_to_sp


def split_detection_frames(
    frames: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Deterministic train/val/test split at the frame level."""
    import random
    rng = random.Random(seed)
    shuffled = list(frames)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    return shuffled[:n_train], shuffled[n_train:n_val], shuffled[n_val:]


# =============================================================================
# Detection Dataset
# =============================================================================

class DetectionDataset(Dataset):
    """
    Frame-level detection dataset.

    Each sample returns a full image tensor and all GT boxes/labels for
    that frame. The DINO decoder predicts boxes for all objects at once.
    """

    def __init__(
        self,
        frames: List[Dict],
        input_size: int = 512,
        is_train: bool = True,
    ):
        """
        Args:
            frames:     Output of load_detection_data().
            input_size: Resize all images to (input_size, input_size).
            is_train:   Enables random augmentation (flips).
        """
        self.frames = frames
        self.input_size = input_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        info = self.frames[idx]
        img = safe_imread(info['img_path'])
        if img is None:
            img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128

        orig_h, orig_w = img.shape[:2]

        # Convert bboxes: pixel xyxy → normalized cxcywh
        boxes = []
        for x0, y0, x1, y1 in info['boxes']:
            cx = (x0 + x1) / 2.0 / orig_w
            cy = (y0 + y1) / 2.0 / orig_h
            w = (x1 - x0) / orig_w
            h = (y1 - y0) / orig_h
            boxes.append([cx, cy, w, h])

        labels = info['labels']

        # Random horizontal flip (training only)
        if self.is_train and np.random.random() < 0.5:
            img = np.fliplr(img).copy()
            boxes = [[1.0 - cx, cy, w, h] for cx, cy, w, h in boxes]

        # Resize to detection resolution
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB → float → normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        return {
            'image': img_tensor,
            'targets': {
                'labels': torch.tensor(labels, dtype=torch.long),
                'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes
                         else torch.zeros(0, 4, dtype=torch.float32),
            },
        }


def detection_collate(batch: List[Dict]) -> Dict:
    """
    Custom collate for variable-length targets.

    Images are stacked into a batch tensor; targets remain as a list
    of dicts (one per image) since each frame has a different number
    of annotations.
    """
    images = torch.stack([b['image'] for b in batch])
    targets = [b['targets'] for b in batch]
    return {'images': images, 'targets': targets}
