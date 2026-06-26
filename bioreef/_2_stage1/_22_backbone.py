"""
BioReef.ai — ViT Backbone Wrapper (DINOv3)
===========================================
Thin wrapper around a frozen DINOv3 Vision Transformer (ViT-B/16) providing
both [CLS] token extraction and dense patch embeddings for each of the
4 Context Harvester streams.

Architecture Logic:
    The DINOv3 backbone was self-supervised on 1.689B images (LVD-1689M)
    using Gram Anchoring, which enforces dense feature stability across
    training. For BioReef.ai, the backbone is frozen — only the downstream
    MCEAM fusion module is trained.

    Each of the 4 streams (ROI, 3x Social, 5x Habitat, Full Frame) is
    independently processed to extract:
        - [CLS] token 'g': Global class token — the fish "signature"
        - Patch embeddings 'P': Local spatial features — the "habitat clues"

    At 224×224 input with patch_size=16, DINOv3 produces:
        - 1 [CLS] token of dimension 768
        - 14×14 = 196 patch tokens of dimension 768
        - 4 register tokens (absorbed into sequence, not used downstream)

    Gram Anchoring ensures patch tokens are semantically consistent across
    viewpoint changes — key for MCEAM cross-attention over reef habitat clues
    and for EMA Re-ID stability in Stage 2.

Guardrails (.agent/rules.md):
    - PyTorch for all model definitions.
    - No generic CNN detectors — DINOv3 ViT is the mandatory backbone.

Reference:
    Darcet et al. (2025), "DINOv3: A 7B-Parameter Vision Foundation Model
    with Gram-Anchored Dense Features." arXiv:2508.10104v1.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger("bioreef._2_stage1._22_backbone")


class ViTBackbone(nn.Module):
    """
    Frozen DINOv3 ViT-B/16 backbone for multi-stream feature extraction.

    Extracts [CLS] tokens and patch embeddings from each of the 4
    Context Harvester streams, preparing them for MCEAM cross-attention.

    Attributes:
        embed_dim (int):        Hidden dimension of the ViT (768 for ViT-B/16).
        num_patches (int):      Number of spatial patches per stream (196 for 224×224).
        patch_size (int):       Size of each image patch (16 for ViT-B/16).
        num_register_tokens (int): Register tokens in the sequence (4 for DINOv3).

    Ecological note:
        DINOv3's Gram Anchoring produces denser, more stable patch features than
        DINOv2, improving MCEAM's ability to distinguish substrate textures and
        scale patterns of closely related reef fish species.
    """

    STREAM_NAMES = ("roi", "social", "habitat", "full_frame")

    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze: bool = True,
    ):
        """
        Args:
            pretrained_model_name: HuggingFace model identifier for DINOv3.
            freeze: Whether to freeze all backbone parameters.
                    Per architecture spec, Stage 1 trains only the MCEAM.
        """
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
        """
        Extract [CLS] token and patch embeddings from a single stream.

        Args:
            x: Input tensor of shape (B, 3, H, W).
               H and W must be divisible by patch_size.

        Returns:
            cls_token:    (B, embed_dim) — global "fish signature"
            patch_tokens: (B, num_patches, embed_dim) — spatial features
        """
        # HuggingFace DINOv3 forward pass
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
        """
        Extract patch tokens only (for the detection head).

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            patch_tokens: (B, num_patches, embed_dim)
        """
        _, patch_tokens = self._extract_features(x)
        return patch_tokens

    def unfreeze_blocks(self, n: int = 2):
        """
        [Phase 10: Domain Adaptation]
        Surgically unfreezes the final N transformer blocks so the backbone
        can adapt to marine fin/scale biology. Backbone is set to train mode
        for the unfrozen blocks only.
        """
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
        """
        Process all 4 Context Harvester streams through the backbone.

        Args:
            streams: Dict with keys from STREAM_NAMES, each a tensor of
                     shape (B, 3, 224, 224).

        Returns:
            Dict mapping stream names to (cls_token, patch_tokens) tuples:
                - cls_token:    (B, 768) — global class embedding
                - patch_tokens: (B, 196, 768) — grid of spatial features

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
        """Output embedding dimension (768 for ViT-B/16)."""
        return self.embed_dim
