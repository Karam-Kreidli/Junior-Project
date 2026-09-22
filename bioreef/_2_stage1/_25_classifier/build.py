"""Assemble the classifier (backbone -> MCEAM -> head) from flat config so the
trainer and evaluator build identical models and every ablation is a flag.

The reference model (paper C09): DINOv3 frozen, MCEAM 1 block / 3 scales, linear
head. Ablations flip one field:
    A1  backbone="dinov2"
    A2  context_levels=0                 (no MCEAM — head on pooled ROI)
    A3  context_levels=1                 (single context stream: social only)
    A4  attention_depth=2                (A4b: 4)
"""

from dataclasses import dataclass

import torch.nn as nn

from .backbone import ViTBackbone, BACKBONES
from .mceam import MCEAM


@dataclass
class ModelConfig:
    backbone: str = "dinov3"          # dinov3 | dinov2
    context_levels: int = 3           # # of context streams the ROI attends to,
                                      # from (social, habitat, full_frame):
                                      # 0=none, 1=social only, 3=all three
    attention_depth: int = 1          # cross-attention blocks per stream
    embed_dim: int = None             # None -> read from the backbone (its real
                                      # hidden size); set only to assert a value
    output_dim: int = 256
    num_heads: int = 8
    unfreeze_blocks: int = 0          # 0 = fully frozen backbone
    # Head used when context_levels == 0 (no MCEAM):
    #   "mlp"    LayerNorm -> Linear(D,out) -> GELU -> Linear(out,C)   (A2)
    #   "linear" Linear(D, C) only — a STRICT linear probe (C01)
    # C01 is the paper's linear-probe floor, and reviewers read "linear probe"
    # strictly: any hidden layer + nonlinearity makes it an MLP probe and a
    # stronger baseline than claimed. Ignored when context_levels > 0.
    probe: str = "mlp"


class Classifier(nn.Module):
    """Frozen backbone (no grad) + optional MCEAM + linear head. The backbone is
    held but never returned as a trainable param; only MCEAM+head (and optional
    last-N blocks) train.

    When context_levels==0 there is no MCEAM: the ROI [CLS] is projected straight
    to output_dim, giving the 'context off' ablation a clean path.
    """

    def __init__(self, cfg: ModelConfig, num_classes: int, attn_implementation: str = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = ViTBackbone(
            pretrained_model_name=BACKBONES[cfg.backbone], freeze=True,
            attn_implementation=attn_implementation,
        )
        # >0 = unfreeze last N; -1 = full fine-tune (any depth). 0 = fully frozen.
        if cfg.unfreeze_blocks != 0:
            self.backbone.unfreeze_blocks(cfg.unfreeze_blocks)

        # Take the embedding width from the ACTUAL backbone, not the hardcoded
        # ModelConfig.embed_dim (768). The two agree for DINOv2/v3-base, but a
        # small/large backbone (e.g. DINOv2-large = 1024) would silently mismatch
        # the MCEAM/head sizing against the real feature dim. If a config sets an
        # explicit embed_dim that disagrees with the backbone, that is a config
        # error — fail loudly rather than build a mis-sized head.
        embed_dim = self.backbone.embed_dim
        if cfg.embed_dim is not None and cfg.embed_dim != embed_dim:
            raise ValueError(
                f"ModelConfig.embed_dim={cfg.embed_dim} disagrees with the "
                f"{cfg.backbone} backbone's actual hidden size {embed_dim}. Leave "
                f"embed_dim as the default (it is read from the backbone) or set it "
                f"to match.")

        if cfg.context_levels > 0:
            self.mceam = MCEAM(
                embed_dim=embed_dim, num_heads=cfg.num_heads,
                output_dim=cfg.output_dim, num_context_levels=cfg.context_levels,
                attention_depth=cfg.attention_depth, use_checkpointing=True,
            )
            head_in = cfg.output_dim
        else:
            # No context: project the ROI [CLS] directly (linear-probe-like, but
            # keeps HSLM + long-tail handling — this is ablation A2, not C01).
            self.mceam = None
            if getattr(cfg, "probe", "mlp") == "linear":
                # Strict linear probe (C01): the ONLY trainable transform of the
                # frozen [CLS] is a single Linear. No LayerNorm, no nonlinearity.
                self.roi_only_proj = nn.Identity()
                head_in = embed_dim
            else:
                self.roi_only_proj = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, cfg.output_dim),
                    nn.GELU(),
                )
                head_in = cfg.output_dim

        self.head = nn.Linear(head_in, num_classes)

    def trainable_modules(self):
        """The modules whose parameters the optimizer should step (everything
        except the frozen backbone)."""
        mods = [self.head]
        mods.append(self.mceam if self.mceam is not None else self.roi_only_proj)
        return mods

    def embed(self, streams):
        """streams dict -> (B, output_dim) embedding (no head)."""
        feats = self.backbone(streams)
        if self.mceam is not None:
            return self.mceam(feats)["embedding"]
        roi_cls, _ = feats["roi"]
        return self.roi_only_proj(roi_cls)

    def forward(self, streams):
        return self.head(self.embed(streams))

    def forward_with_attention(self, streams):
        """Like forward(), but also return MCEAM's cross-attention maps for
        visualization. Returns (logits, attentions) where attentions is a
        {stream_name: (B, H, 1, N)} dict — the ROI [CLS] query's attention onto
        each context stream's patches. Empty dict when context is off (A2), where
        no MCEAM exists. Never used in training/eval; keeps forward() untouched."""
        feats = self.backbone(streams)
        if self.mceam is None:
            return self.head(self.roi_only_proj(feats["roi"][0])), {}
        out = self.mceam(feats, return_attention=True)
        return self.head(out["embedding"]), out.get("attentions", {})
