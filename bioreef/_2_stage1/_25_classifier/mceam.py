"""
MCEAM — Multi-Context Environmental Attention Module.

The ROI [CLS] token (the fish) cross-attends to the patch embeddings of each
context stream (the environment). Each stream yields ONE attended vector; these
are CONCATENATED with the ROI [CLS], projected by a fusion FFN, and blended with a
projected ROI via a learned gate into the context-aware embedding z. Inspired by
MATANet's multi-context attention, with the learned gate as the addition (paper 4.2).

Per context stream j (attended vector, then combined by concat+FFN — NOT summed):
    a_j = softmax((W_q·g)·(W_k·P_j)^T / sqrt(d)) · (W_v·P_j)
    z_ctx = FFN( concat[ g, a_1, ..., a_C ] )
    z     = gate·z_ctx + (1 - gate)·(W_roi·g)

NOTE on attention_depth > 1: the CrossAttentionBlock is a cross-attention +
output-projection layer with LayerNorm, but NO residual around attention and NO
FFN sublayer — so it is not a standard Transformer block. Stacking depth>1 refines
the query through each block in turn (the query after block 1 is the attended
output, not the original ROI [CLS]); the original ROI query is not re-used
independently at every layer.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("bioreef.model.mceam")


class CrossAttentionBlock(nn.Module):
    """One cross-attention level: ROI [CLS] query attends to a context stream's
    patch embeddings (keys/values)."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ROI query (B,1,D)/(B,D) attends to context (B,N,D) -> attended (B,D)
        and optional attn map (B,H,1,N)."""
        if query.dim() == 2:
            query = query.unsqueeze(1)

        B, N_q, D = query.shape
        _, N_kv, _ = context.shape

        query = self.norm_q(query)
        context = self.norm_kv(context)

        Q = self.W_q(query)
        K = self.W_k(context)
        V = self.W_v(context)

        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,H,1,N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)  # (B,H,1,d)
        attended = attended.transpose(1, 2).contiguous().view(B, N_q, D)
        attended = self.out_proj(attended)
        attended = self.out_dropout(attended)
        attended = attended.squeeze(1)  # (B,D)

        if return_attention:
            return attended, attn_weights
        return attended, None


class MCEAM(nn.Module):
    """Fuse the ROI with all context streams via cross-attention + a gated FFN
    into the embedding z (morphology + social + habitat + environment).

    num_context_levels controls how many context scales feed the attention:
    3 = ROI/social/habitat (paper reference), 0 = none (ablation A2/A3).
    attention_depth stacks N cross-attention blocks per stream (ablation A4).
    """

    CONTEXT_STREAMS = ("social", "habitat", "full_frame")

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 256,
        num_context_levels: int = 3,
        attention_depth: int = 1,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        # Only 1 or 3 are valid: the fusion layer is sized for exactly
        # num_context_levels streams, and CONTEXT_STREAMS has length 3, so e.g.
        # 4 would build 3 attention blocks but a fusion layer expecting 4 vectors
        # -> a shape error later. 0 context must bypass MCEAM entirely (build.py).
        if attention_depth <= 0:
            raise ValueError(
                f"MCEAM attention_depth must be > 0, got {attention_depth}: "
                "a depth of 0 builds empty attention stacks and silently ignores "
                "all context (the attended output is just the ROI query)."
            )
        if num_context_levels not in (1, 3):
            raise ValueError(
                f"MCEAM num_context_levels must be 1 or 3, got {num_context_levels}. "
                "Use context_levels: 0 (no MCEAM) for the ROI-only path."
            )
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_context_levels = num_context_levels
        self.attention_depth = attention_depth
        self.use_checkpointing = use_checkpointing

        # attention_depth cross-attention blocks per context level (A4 ablation).
        self.cross_attention_blocks = nn.ModuleDict({
            name: nn.ModuleList([
                CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(attention_depth)
            ])
            for name in self.CONTEXT_STREAMS[:num_context_levels]
        })

        fusion_input_dim = embed_dim * (1 + num_context_levels)
        self.fusion_ffn = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Residual gate: learned weighting between ROI-only and context-enriched.
        self.gate = nn.Sequential(nn.Linear(embed_dim + output_dim, 1), nn.Sigmoid())
        self.roi_proj = nn.Linear(embed_dim, output_dim)

        logger.info(
            f"MCEAM: {num_context_levels} context levels x depth {attention_depth}, "
            f"{num_heads} heads, embed_dim={embed_dim} -> output_dim={output_dim}"
        )

    def _attend(self, attn_blocks, query, context, return_attention):
        """Run a stack of cross-attention blocks; the ROI query is refined by
        each block in turn. Returns (attended, attn_of_last_block)."""
        attended = query
        attn_weights = None
        for block in attn_blocks:
            if self.use_checkpointing and self.training:
                import torch.utils.checkpoint as cp
                def block_forward(q, c, _b=block):
                    out, _ = _b(query=q, context=c, return_attention=False)
                    return out
                attended = cp.checkpoint(block_forward, attended, context, use_reentrant=False)
                attn_weights = None
            else:
                attended, attn_weights = block(
                    query=attended, context=context, return_attention=return_attention
                )
        return attended, attn_weights

    def forward(
        self,
        backbone_features: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Fuse ROI with multi-scale context (ViTBackbone output) -> dict with
        'embedding' z (B, output_dim), 'roi_cls' (B, embed_dim), and optional
        'attentions'."""
        roi_cls, _ = backbone_features["roi"]  # (B, D)

        attended_features = []
        attention_maps = {}

        # Every configured context stream is REQUIRED. Skipping a missing one
        # would make `concat` narrower than fusion_ffn expects and fail with an
        # opaque matmul shape error; name the missing stream instead.
        missing = [s for s in self.cross_attention_blocks if s not in backbone_features]
        if missing:
            raise KeyError(
                f"MCEAM: required context stream(s) {missing} absent from backbone "
                f"features (have {sorted(backbone_features)}). The dataset/backbone "
                "must emit every configured stream."
            )
        for stream_name, attn_blocks in self.cross_attention_blocks.items():
            _, context_patches = backbone_features[stream_name]  # (B, N, D)
            attended, attn_weights = self._attend(
                attn_blocks, roi_cls, context_patches, return_attention
            )
            attended_features.append(attended)
            if return_attention and attn_weights is not None:
                attention_maps[stream_name] = attn_weights

        concat = torch.cat([roi_cls] + attended_features, dim=-1)  # (B, D*(1+C))
        z_context = self.fusion_ffn(concat)  # (B, output_dim)

        # Gated residual: blend ROI-only with context-enriched.
        roi_projected = self.roi_proj(roi_cls)
        gate_weight = self.gate(torch.cat([roi_cls, z_context], dim=-1))  # (B, 1)
        z = gate_weight * z_context + (1 - gate_weight) * roi_projected

        result = {"embedding": z, "roi_cls": roi_cls}
        if return_attention:
            result["attentions"] = attention_maps
        return result
