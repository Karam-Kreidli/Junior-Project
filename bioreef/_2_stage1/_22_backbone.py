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
        # DINOv3/DINOv2 last_hidden_state layout: [CLS, registers, patches]
        #   index 0                       : CLS token
        #   indices 1 .. R                : register tokens (if any) — come BEFORE patches
        #   indices 1+R .. 1+R+num_patches: patch tokens
        # (Defect B: an earlier comment placed registers AFTER the patches, which
        # is wrong for these backbones. The slice below is correct — it skips CLS
        # and the R register tokens — but the comment must match so a future edit
        # doesn't "fix" it the wrong way.)
        outputs = self.vit(pixel_values=x)
        hidden = outputs.last_hidden_state  # (B, seq_len, D)

        # Actual patch count from the real sequence length (robust to input size).
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
        # KNOWN_BUGS #7: block ModuleList lives at different paths per backbone
        # (HF DINOv2 = encoder.layer; DINOv3 tf-4.56 = bare layer; timm = blocks).
        # A miss must NOT silently unfreeze the WHOLE network — that turns a
        # "last-N blocks" adaptation into a full fine-tune. Probe the known paths
        # and RAISE if none match, rather than guessing.
        blocks = None
        for owner, attr in (("encoder", "layer"), (None, "layer"), (None, "blocks")):
            obj = getattr(self.vit, owner) if owner else self.vit
            if obj is not None and hasattr(obj, attr):
                blocks = getattr(obj, attr)
                break
        if blocks is None:
            raise RuntimeError(
                "unfreeze_blocks: could not locate the transformer-block "
                "ModuleList on this backbone (tried encoder.layer, layer, blocks). "
                "Refusing to silently unfreeze the whole network — add this "
                "backbone's block path before enabling domain adaptation."
            )

        total = len(blocks)
        self.vit.train()
        if n >= total:
            # True full fine-tune: unfreeze every backbone param, not just blocks.
            for param in self.vit.parameters():
                param.requires_grad = True
        else:
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
        # Every stream is required downstream (MCEAM uses roi as the query and each
        # context stream as keys/values). Silently dropping a missing one just
        # defers the failure to an opaque shape error in MCEAM's fusion (Bug A);
        # fail here at the source with the clearest message instead.
        missing = [name for name in self.STREAM_NAMES if name not in streams]
        if missing:
            raise KeyError(
                f"ViTBackbone: input stream(s) {missing} missing from the harvested "
                f"streams (have {sorted(streams)}). The ContextHarvester must emit "
                f"all of {list(self.STREAM_NAMES)}."
            )

        features = {}
        for name in self.STREAM_NAMES:
            cls_tok, patch_tok = self._extract_features(streams[name])
            features[name] = (cls_tok, patch_tok)

        return features

    @property
    def output_dim(self) -> int:
        """Output embedding dimension (768 for ViT-B/16)."""
        return self.embed_dim
