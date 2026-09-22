"""Stage-1 species classifier — the CORRECTED model, ported from the research repo.

This package holds the model that the deployment checkpoint (D6a) was actually
trained with. It is a verbatim copy of the research repo's model code, kept
separate from the older `_22_backbone.py` / `_23_mceam.py` in this directory:
those still serve the legacy `bioreef_stage1.pt`, and their `ViTBackbone`
hardcodes DINOv3 ViT-B (768-d) with a frozen, never-loaded backbone — which
cannot represent a full fine-tuned ViT-H+ (1280-d) model.

Ported from bioreef-classify (now retired; the paper code is public at
github.com/Karam-Kreidli/ozfish-finegrained-classification-benchmark):
    bioreef/model/build.py     -> build.py
    bioreef/model/backbone.py  -> backbone.py
    bioreef/model/mceam.py     -> mceam.py
Unmodified except for this package wrapper — keep them that way, since the
trained weights only load into these exact module/parameter names.

The research repo's own `model/__init__.py` is deliberately NOT copied: it
eagerly imports the training-only loss and the timm baselines, which would drag
`timm` into the inference path for no reason.

Deployment model: D6a seed 1 — DINOv3 ViT-H+/16, full fine-tune, lr 2.5e-6,
321 species. See models/README.md for provenance and metrics.
"""

from .build import Classifier, ModelConfig
from .backbone import ViTBackbone, BACKBONES
from .mceam import MCEAM, CrossAttentionBlock

__all__ = ["Classifier", "ModelConfig", "ViTBackbone", "BACKBONES",
           "MCEAM", "CrossAttentionBlock"]
