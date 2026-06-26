"""
BioReef.ai — Evaluation (group 4x).

Metrics over the pipeline's output: HOTA for tracking quality and the
hierarchical-distance evaluator for taxonomy-aware classification error.

    _41_hota/   _42_hd/   (each: evaluator class + metrics.py)
"""

from ._41_hota import HOTAEvaluator
from ._42_hd import HDEvaluator

__all__ = ["HOTAEvaluator", "HDEvaluator"]
