"""
ViT backbone wrapper — frozen DINOv3 ViT-B/16 (Darcet et al. 2025).

Per Context-Harvester stream, extracts the [CLS] token (global "fish signature")
and patch embeddings (local "habitat clues") for MCEAM. The backbone is frozen;
only MCEAM trains. At 224×224 / patch 16: 1 CLS + 196 patches + 4 register
tokens, dim 768.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger("bioreef._2_stage1._22_backbone")


class ViTBackbone(nn.Module):
    """Frozen DINOv3 ViT-B/16 backbone extracting [CLS] + patch tokens from each
    of the 4 Context-Harvester streams for MCEAM cross-attention."""

    STREAM_NAMES = ("roi", "social", "habitat", "full_frame")

    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze: bool = True,  # Stage 1 trains only MCEAM
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name

        logger.info(f"Loading DINOv3 backbone: {pretrained_model_name}")
        self.vit = AutoModel.from_pretrained(pretrained_model_name)

        # Extract architecture metadata from HuggingFace config
        cfg = self.vit.config
        self.embed_dim = cfg.hidden_size                              # 768
        self.patch_size = cfg.patch_size                              # 16
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)  # 4
        # For 224×224 input with patch_size=16: (224/16)^2 = 196 patches
        self.num_patches = (224 // self.patch_size) ** 2              # 196

        if freeze:
            self._freeze()

        logger.info(
            f"DINOv3 initialized: embed_dim={self.embed_dim}, "
            f"patch_size={self.patch_size}, num_patches={self.num_patches}, "
            f"num_register_tokens={self.num_register_tokens}, frozen={freeze}"
        )

    def _freeze(self):
        """Freeze all backbone parameters for feature extraction only."""
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()
        logger.info("DINOv3 backbone frozen — gradients disabled.")

    def train(self, mode: bool = True):
        """Override train to keep backbone in eval mode when frozen."""
        super().train(mode)
        if not any(p.requires_grad for p in self.vit.parameters()):
            self.vit.eval()
        return self

    def _extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[CLS] (B, D) + patch tokens (B, num_patches, D) from one stream."""
        # last_hidden_state shape: (B, 1 + num_patches + num_register_tokens, D)
        #   index 0             : CLS token
        #   indices 1..N        : patch tokens
        #   indices N+1..N+R    : register tokens (if any)
        outputs = self.vit(pixel_values=x)
        hidden = outputs.last_hidden_state  # (B, seq_len, D)

        # Compute actual patch count from input resolution
        num_patches = hidden.shape[1] - 1 - self.num_register_tokens

        cls_token = hidden[:, 0]                                              # (B, D)
        patch_tokens = hidden[:, 1 + self.num_register_tokens:]              # (B, num_patches, D)

        return cls_token, patch_tokens

    def extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch tokens only (B, num_patches, embed_dim) — for the detection head."""
        _, patch_tokens = self._extract_features(x)
        return patch_tokens

    def unfreeze_blocks(self, n: int = 2):
        """Domain adaptation: unfreeze the final N transformer blocks (+ final
        layer-norm) so the backbone adapts to marine fin/scale biology."""
        # HuggingFace ViT uses encoder.layer; timm/torch.hub uses blocks
        if hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
            blocks = self.vit.encoder.layer
        elif hasattr(self.vit, "blocks"):
            blocks = self.vit.blocks
        else:
            logger.warning("Could not map ViT blocks. Unfreezing entire network (DANGER).")
            for param in self.vit.parameters():
                param.requires_grad = True
            self.vit.train()
            return

        total = len(blocks)
        self.vit.train()
        for i, block in enumerate(blocks):
            if i >= total - n:
                for param in block.parameters():
                    param.requires_grad = True

        # Unfreeze final layer-norm for numeric stability during adaptation
        layernorm = (
            getattr(self.vit, "layernorm", None)  # HuggingFace
            or getattr(self.vit, "norm", None)     # timm
        )
        if layernorm is not None:
            for param in layernorm.parameters():
                param.requires_grad = True

        logger.info(
            f"DOMAIN ADAPTATION ENABLED: Unfrozen final {n}/{total} "
            "DINOv3 transformer blocks!"
        )

    def forward(
        self, streams: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Run all 4 streams -> {name: (cls (B,768), patches (B,196,768))}. The
        ROI [CLS] is MCEAM's query; context patches are its keys/values."""
        features = {}

        for name in self.STREAM_NAMES:
            if name in streams:
                cls_tok, patch_tok = self._extract_features(streams[name])
                features[name] = (cls_tok, patch_tok)
            else:
                logger.warning(f"Stream '{name}' not found in input dict.")

        return features

    @property
    def output_dim(self) -> int:
        """Output embedding dimension (768 for ViT-B/16)."""
        return self.embed_dim
