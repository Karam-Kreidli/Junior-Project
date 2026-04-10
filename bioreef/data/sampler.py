"""
BioReef.ai — Weighted Random Sampler (Taxonomic Class Balance)
==============================================================
Handles severe class imbalance by assigning sampling weights inversely
proportional to class frequencies.

Results of Taxonomic Class Balance Audit (4.6K Subset):
    1. CDFW-LakeCam-April-Tules1 (Family A)        : 3848
    2. CDFW-LakeCam-April-SpiderBlocks1 (Family B) :  698
    3. CDFW-LakeCam-April-Tules2 (Family C)        :   71

Imbalance Ratio: 3848 / 71 = 54.2
Status: > 10.0 (CRITICAL IMBALANCE)

Requirement (.agent/rules.md):
    Models must not ignore rare species. This sampler up-weights minority
    classes (like Tules2) during the training epoch to equalize exposure.
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import logging

logger = logging.getLogger("bioreef.data.sampler")

def create_weighted_sampler(dataset_labels: list) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler based on the label distribution.
    
    Args:
        dataset_labels: A list of class IDs (integers) corresponding 
                        to each sample in the dataset.
                        
    Returns:
        WeightedRandomSampler configured to oversample rare classes.
    """
    class_counts = Counter(dataset_labels)
    total_samples = len(dataset_labels)
    num_classes = len(class_counts)
    
    # Weight per class = 1.0 / count (or total / count)
    class_weights = {
        cls: total_samples / float(count) 
        for cls, count in class_counts.items()
    }
    
    # Assign the corresponding weight to every individual sample
    sample_weights = [class_weights[label] for label in dataset_labels]
    
    logger.info(
        f"WeightedRandomSampler built for {num_classes} classes. "
        f"Imbalance ratio (Max/Min): {max(class_counts.values()) / min(class_counts.values()):.2f}"
    )
    
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=total_samples,
        replacement=True
    )
