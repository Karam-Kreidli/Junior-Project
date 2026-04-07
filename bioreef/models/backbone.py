"""
BioReef.ai — DINOv2 Backbone Wrapper
=====================================
Thin wrapper around a frozen DINOv2 Vision Transformer (ViT-B/14) providing
both [CLS] token extraction and dense patch embeddings for each of the
4 Context Harvester streams.

Architecture Logic:
    The DINOv2 backbone was self-supervised on 142M images and produces
    rich, general-purpose visual features. For BioReef.ai, the backbone
    is frozen — only the downstream MCEAM fusion module is trained.

    Each of the 4 streams (ROI, 3x Social, 5x Habitat, Full Frame) is
    independently processed to extract:
        - [CLS] token 'g': Global class token — the fish "signature"
        - Patch embeddings 'P': Local spatial features — the "habitat clues"

    At 224×224 input with patch_size=14, DINOv2 produces:
        - 1 [CLS] token of dimension 768
        - 16×16 = 256 patch tokens of dimension 768

Guardrails (.agent/rules.md):
    - PyTorch for all model definitions.
    - No generic CNN detectors — DINOv2 ViT is the mandatory backbone.

Reference:
    Oquab et al. (2023), "DINOv2: Learning Robust Visual Features
    without Supervision."
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("bioreef.models.backbone")


class DINOv2Backbone(nn.Module):
    """
    Frozen DINOv2 ViT-B/14 backbone for multi-stream feature extraction.

    Extracts [CLS] tokens and patch embeddings from each of the 4
    Context Harvester streams, preparing them for MCEAM cross-attention.

    Attributes:
        embed_dim (int): Hidden dimension of the ViT (768 for ViT-B/14).
        num_patches (int): Number of spatial patches per stream (256 for 224×224).
        patch_size (int): Size of each image patch (14 for ViT-B/14).

    Ecological note:
        The self-supervised pretraining of DINOv2 on diverse imagery means
        it has already learned to attend to texture boundaries — ideal for
        distinguishing scale patterns of closely related reef fish species
        (e.g., separating Lutjanus ehrenbergii from L. kasmira by spot
        distribution).
    """

    STREAM_NAMES = ("roi", "social", "habitat", "full_frame")

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        hub_repo: str = "facebookresearch/dinov2",
        freeze: bool = True,
    ):
        """
        Args:
            model_name: DINOv2 variant identifier.
                        Options: dinov2_vits14 (21M), dinov2_vitb14 (86M),
                                 dinov2_vitl14 (300M), dinov2_vitg14 (1.1B).
            hub_repo:   PyTorch Hub repository for DINOv2 weights.
            freeze:     Whether to freeze all backbone parameters.
                        Per architecture spec, Stage 1 trains only the MCEAM.
        """
        super().__init__()
        self.model_name = model_name

        logger.info(f"Loading DINOv2 backbone: {model_name} from {hub_repo}")
        self.vit = torch.hub.load(hub_repo, model_name, pretrained=True)

        # Extract architecture metadata
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_size if hasattr(self.vit, "patch_size") else 14
        # For 224×224 input with patch_size=14: (224/14)^2 = 256 patches
        self.num_patches = (224 // self.patch_size) ** 2

        # Freeze backbone (only MCEAM is trained)
        if freeze:
            self._freeze()

        logger.info(
            f"DINOv2 initialized: embed_dim={self.embed_dim}, "
            f"patch_size={self.patch_size}, num_patches={self.num_patches}, "
            f"frozen={freeze}"
        )

    def _freeze(self):
        """Freeze all backbone parameters for feature extraction only."""
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()
        logger.info("DINOv2 backbone frozen — gradients disabled.")

    def train(self, mode: bool = True):
        """Override train to keep backbone in eval mode when frozen."""
        super().train(mode)
        # Always keep the backbone in eval mode
        if not any(p.requires_grad for p in self.vit.parameters()):
            self.vit.eval()
        return self

    def _extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract [CLS] token and patch embeddings from a single stream.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            cls_token: (B, embed_dim) — global "fish signature"
            patch_tokens: (B, num_patches, embed_dim) — spatial features
        """
        # DINOv2 forward_features returns all tokens: [CLS] + patches
        # Use get_intermediate_layers for full access
        features = self.vit.forward_features(x)

        # features is a dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
        # or a tensor depending on the version
        if isinstance(features, dict):
            cls_token = features.get("x_norm_clstoken", features.get("x_cls_token"))
            patch_tokens = features.get("x_norm_patchtokens", features.get("x_patch_tokens"))
            if cls_token is None or patch_tokens is None:
                # Fallback: manual split
                all_tokens = features.get("x_prenorm", features.get("x_norm"))
                cls_token = all_tokens[:, 0]
                patch_tokens = all_tokens[:, 1:]
        else:
            # Tensor output: first token is [CLS], rest are patches
            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]

        return cls_token, patch_tokens

    def unfreeze_blocks(self, n=2):
        """
        [Phase 10: Domain Adaptation]
        Surgically unfreezes the final N transformer blocks so DINOv2 can rewrite
        its geometric features specifically for marine fin/scale biology, 
        giving you genuine intelligence over artificial inflation!
        """
        self.vit.train()  # Force unlocked layers to compute gradients
        if hasattr(self.vit, 'blocks'):
            total = len(self.vit.blocks)
            for i, block in enumerate(self.vit.blocks):
                if i >= total - n:
                    for param in block.parameters():
                        param.requires_grad = True
            
            # Unfreeze the final layer-norm for numeric stability during adaptation
            if hasattr(self.vit, 'norm'):
                for param in self.vit.norm.parameters():
                    param.requires_grad = True
            logger.info(f"DOMAIN ADAPTATION ENABLED: Unfrozen final {n}/{total} DINOv2 transformer blocks!")
        else:
            logger.warning("Could not map ViT blocks. Unfreezing entire Network (DANGER).")
            for param in self.vit.parameters():
                param.requires_grad = True

    def forward(
        self, streams: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process all 4 Context Harvester streams through the backbone.

        Args:
            streams: Dict with keys from STREAM_NAMES, each a tensor of
                     shape (B, 3, 224, 224).

        Returns:
            Dict mapping stream names to (cls_token, patch_tokens) tuples:
                - cls_token:    (B, 768) — global class embedding
                - patch_tokens: (B, 256, 768) — grid of spatial features

            The ROI stream's [CLS] token serves as the Query (g) for the
            MCEAM, while context stream patches serve as Keys and Values.
        """
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
        """Output embedding dimension (768 for ViT-B/14)."""
        return self.embed_dim
