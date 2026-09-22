"""
ViT backbone wrapper — frozen DINOv3 ViT-B/16 (Darcet et al. 2025).

Per context stream, extracts the [CLS] token (global "fish signature") and patch
embeddings (local "habitat clues") for MCEAM. The backbone is frozen by default;
only MCEAM + the head train. At 224x224 / patch 16: 1 CLS + 196 patches + 4
register tokens, dim 768.

The model name is a constructor arg so the DINOv2-vs-DINOv3 ablation (paper A1)
is a flag, not a code edit.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger("bioreef.model.backbone")

# Backbone presets for the ablation panel. `name` -> HuggingFace model id.
# The panel uses dinov3 (ViT-B/16); dinov2 (ViT-B/14) is the backbone-generation
# ablation. dinov3_large (ViT-L/16, ~300M, embed 1024) is a DEPLOYMENT-only
# experiment (not a paper run) — the L40S can fine-tune it where the benchmark GPUs
# could not. embed_dim is read from the backbone (build.py), so a bigger backbone
# sizes MCEAM/head automatically. All are gated HF repos (need the HF token).
BACKBONES = {
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov2": "facebook/dinov2-base",
    "dinov3_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    # dinov3_huge (ViT-H+/16, ~840M, embed 1280, 32 blocks) — next DEPLOYMENT rung
    # after ViT-L trended up (D3a). Same DINOv3 layout as ViT-L, so _BLOCK_PATHS and
    # the embed_dim auto-read already cover it; unfreeze_blocks(-1) full-FTs all 32.
    "dinov3_huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


class ViTBackbone(nn.Module):
    """Frozen DINOv3/DINOv2 ViT backbone extracting [CLS] + patch tokens from
    each context stream for MCEAM cross-attention."""

    STREAM_NAMES = ("roi", "social", "habitat", "full_frame")

    def __init__(
        self,
        pretrained_model_name: str = BACKBONES["dinov3"],
        freeze: bool = True,
        attn_implementation: str = None,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name

        logger.info(f"Loading ViT backbone: {pretrained_model_name}")
        # attn_implementation="eager" is REQUIRED for attention_rollout: the SDPA
        # / flash kernels (the transformers default) never materialize the
        # attention matrix, so output_attentions=True silently returns nothing.
        # Training/eval leave this None (fast default); only visualize.py asks for
        # eager, and only for the frozen model it builds to render saliency.
        kw = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
        self.vit = AutoModel.from_pretrained(pretrained_model_name, **kw)

        cfg = self.vit.config
        self.embed_dim = cfg.hidden_size                              # 768
        self.patch_size = cfg.patch_size                              # 16 / 14
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)
        self.num_patches = (224 // self.patch_size) ** 2

        if freeze:
            self._freeze()

        logger.info(
            f"Backbone ready: embed_dim={self.embed_dim}, "
            f"patch_size={self.patch_size}, num_patches={self.num_patches}, "
            f"num_register_tokens={self.num_register_tokens}, frozen={freeze}"
        )

    def _freeze(self):
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()
        logger.info("Backbone frozen — gradients disabled.")

    def train(self, mode: bool = True):
        """Keep the backbone in eval mode when frozen."""
        super().train(mode)
        if not any(p.requires_grad for p in self.vit.parameters()):
            self.vit.eval()
        return self

    def _extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[CLS] (B, D) + patch tokens (B, num_patches, D) from one stream."""
        outputs = self.vit(pixel_values=x)
        hidden = outputs.last_hidden_state  # (B, 1 + num_patches + num_reg, D)
        cls_token = hidden[:, 0]
        patch_tokens = hidden[:, 1 + self.num_register_tokens:]
        return cls_token, patch_tokens

    # Where the transformer-block ModuleList lives, per HF backbone layout:
    #   DINOv3 (transformers 4.56): vit.layer   (blocks are a bare `.layer`
    #                               ModuleList directly on the model root)
    #   DINOv3 (other layouts):     vit.model.layer
    #   DINOv2:                     vit.encoder.layer
    #   timm-style ViT:             vit.blocks
    # Order matters: probe the most specific paths first. If none match we MUST
    # fail — silently unfreezing the whole network turns a last-N-block ablation
    # into a full fine-tune (a different experiment), the bug this replaces.
    _BLOCK_PATHS = (("model", "layer"), ("encoder", "layer"),
                    ("layer",), ("blocks",))

    def _find_blocks(self):
        for path in self._BLOCK_PATHS:
            obj = self.vit
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            else:
                import torch.nn as nn
                if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                    return obj
        return None

    def unfreeze_blocks(self, n: int = 2):
        """Domain adaptation: unfreeze the final N transformer blocks (+ final
        layer-norm). Optional last-N-block path mentioned in the paper (4.1).

        n = -1 means FULL fine-tune regardless of the block count — use this for a
        full-FT recipe on a backbone whose depth may differ (e.g. ViT-L has 24
        blocks, ViT-B has 12). Passing a hardcoded 12 on a 24-block model would
        silently unfreeze only the last half — a partial FT wearing a full-FT
        config. -1 resolves to the model's real total so the recipe is portable."""
        blocks = self._find_blocks()
        if blocks is None:
            raise RuntimeError(
                f"unfreeze_blocks({n}): could not locate the transformer-block "
                f"ModuleList on backbone '{self.pretrained_model_name}'. Tried "
                f"{self._BLOCK_PATHS}. Refusing to unfreeze the whole network "
                f"silently — add this backbone's block path to _BLOCK_PATHS."
            )

        total = len(blocks)
        if n == -1:                      # full fine-tune, any depth
            n = total
        if not 0 < n <= total:
            raise ValueError(f"unfreeze_blocks: n={n} out of range for {total} blocks")

        self.vit.train()
        if n == total:
            # A11 is reported as the FULL fine-tune endpoint, so it must actually
            # be one. Unfreezing only the blocks would leave the patch embedding,
            # CLS/register tokens and positional embeddings frozen — a partial
            # fine-tune wearing the name of a full one.
            for param in self.vit.parameters():
                param.requires_grad = True
            n_tr = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            logger.info(f"Unfroze the ENTIRE backbone: all {total} blocks + patch "
                        f"embedding, tokens and positional embeddings "
                        f"({n_tr:,} backbone params now trainable).")
            return

        for i, block in enumerate(blocks):
            if i >= total - n:
                for param in block.parameters():
                    param.requires_grad = True

        layernorm = getattr(self.vit, "layernorm", None) or getattr(self.vit, "norm", None)
        if layernorm is None:
            # Not fatal (the blocks still adapt), but the final norm staying
            # frozen changes what "unfreeze last N" means — say so.
            logger.warning(
                "unfreeze_blocks(%d): no final layer-norm found at .layernorm or "
                ".norm on '%s'; the last-N blocks are unfrozen but the final norm "
                "remains frozen.", n, self.pretrained_model_name,
            )
        else:
            for param in layernorm.parameters():
                param.requires_grad = True

        n_tr = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        logger.info(f"Unfroze final {n}/{total} transformer blocks "
                    f"({n_tr:,} backbone params now trainable).")

    def forward(
        self, streams: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Run all streams -> {name: (cls (B,D), patches (B,N,D))}. The ROI
        [CLS] is MCEAM's query; context patches are its keys/values."""
        features = {}
        for name in self.STREAM_NAMES:
            if name in streams:
                features[name] = self._extract_features(streams[name])
            else:
                logger.warning(f"Stream '{name}' not found in input dict.")
        return features

    @torch.no_grad()
    def attention_rollout(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone saliency for one stream via attention rollout (Abnar &
        Zuidema, 2020): multiply the per-layer, head-averaged self-attention
        (with residual, re-normalized) down the depth, then read the [CLS] row.

        x: (B, 3, H, W). Returns (B, gh, gw) CLS->patch saliency on the patch
        grid (register tokens dropped), each map min-max normalized to [0,1].
        Answers 'what does the frozen ViT look at on the crop' — separate from
        MCEAM's cross-attention.
        """
        outputs = self.vit(pixel_values=x, output_attentions=True)
        attns = outputs.attentions  # tuple(L) of (B, H, T, T)
        if not attns:
            raise RuntimeError(
                f"{self.pretrained_model_name} did not return attentions; "
                "this backbone does not support rollout."
            )

        B, _, T, _ = attns[0].shape
        device = attns[0].device
        result = torch.eye(T, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        for a in attns:
            a = a.mean(dim=1)                      # head-average -> (B, T, T)
            a = a + torch.eye(T, device=device)    # add residual connection
            a = a / a.sum(dim=-1, keepdim=True)    # renormalize rows
            result = torch.bmm(a, result)          # accumulate down the depth

        start = 1 + self.num_register_tokens       # drop CLS + register tokens
        cls_to_patches = result[:, 0, start:]      # (B, num_patches)

        grid = int(self.num_patches ** 0.5)
        sal = cls_to_patches.reshape(B, grid, grid)
        flat = sal.flatten(1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        sal = ((flat - lo) / (hi - lo + 1e-8)).reshape(B, grid, grid)
        return sal

    @property
    def output_dim(self) -> int:
        return self.embed_dim
