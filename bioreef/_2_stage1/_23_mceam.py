"""
MCEAM — Multi-Context Environmental Attention Module (MATANet, Lee et al. 2026).

Stage-1 fusion: the ROI [CLS] token (the fish) cross-attends to the patch
embeddings of each context stream (the environment), then a gated FFN fuses
them into the context-aware embedding z.

    F_attn = Σ_j softmax((W_q·g)·(W_k·P_j)ᵀ / √d) · (W_v·P_j)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("bioreef._2_stage1._23_mceam")


class CrossAttentionBlock(nn.Module):
    """One-level cross-attention: ROI [CLS] query attends to a context stream's
    patch embeddings (keys/values)."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Learned projection matrices W_q, W_k, W_v
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Layer normalization for stable training
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
        # Ensure query has sequence dimension
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, 1, D)

        B, N_q, D = query.shape
        _, N_kv, _ = context.shape

        # Pre-norm
        query = self.norm_q(query)
        context = self.norm_kv(context)

        # Project to Q, K, V
        Q = self.W_q(query)    # (B, 1, D)
        K = self.W_k(context)  # (B, N, D)
        V = self.W_v(context)  # (B, N, D)

        # Reshape for multi-head attention
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, 1, d)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, 1, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attended = torch.matmul(attn_weights, V)  # (B, H, 1, d)

        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(B, N_q, D)  # (B, 1, D)
        attended = self.out_proj(attended)
        attended = self.out_dropout(attended)

        # Squeeze sequence dimension
        attended = attended.squeeze(1)  # (B, D)

        if return_attention:
            return attended, attn_weights
        return attended, None


class MCEAM(nn.Module):
    """
    Multi-Context Environmental Attention Module — fuses the ROI with all context
    streams via cross-attention + a gated FFN into the embedding z:

        roi_cls ─→ CrossAttn(social) ─┐
                ─→ CrossAttn(habitat)─┤
                ─→ CrossAttn(full)  ──┴→ Concat+FFN ─→ z  (morphology + social +
                                                           habitat + environment)
    """

    CONTEXT_STREAMS = ("social", "habitat", "full_frame")

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 256,
        num_context_levels: int = 3,
        use_checkpointing: bool = False,  # torch.utils.checkpoint to save VRAM
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_context_levels = num_context_levels
        self.use_checkpointing = use_checkpointing

        # One cross-attention block per context level
        self.cross_attention_blocks = nn.ModuleDict({
            name: CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for name in self.CONTEXT_STREAMS[:num_context_levels]
        })

        # Fusion FFN: projects concatenated attended features + ROI cls
        # Input: ROI cls (D) + attended features (D × num_context_levels)
        fusion_input_dim = embed_dim * (1 + num_context_levels)

        self.fusion_ffn = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Residual gate: learned weighting between ROI-only and context-enriched
        self.gate = nn.Sequential(
            nn.Linear(embed_dim + output_dim, 1),
            nn.Sigmoid(),
        )

        # ROI projection to match output_dim for gated residual
        self.roi_proj = nn.Linear(embed_dim, output_dim)

        logger.info(
            f"MCEAM initialized: {num_context_levels} context levels, "
            f"{num_heads} heads, embed_dim={embed_dim} → output_dim={output_dim}"
        )

    def forward(
        self,
        backbone_features: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Fuse ROI with multi-scale context (ViTBackbone output) -> dict with
        'embedding' z (B, output_dim), 'roi_cls' (B, embed_dim), and optional
        'attentions'."""
        # Extract ROI [CLS] token as the Query
        roi_cls, _ = backbone_features["roi"]  # (B, D)

        # Cross-attend to each context level
        attended_features = []
        attention_maps = {}

        for stream_name, attn_block in self.cross_attention_blocks.items():
            if stream_name not in backbone_features:
                logger.warning(
                    f"Context stream '{stream_name}' not in backbone features. "
                    "Skipping."
                )
                continue

            _, context_patches = backbone_features[stream_name]  # (B, N, D)

            if self.use_checkpointing and self.training:
                import torch.utils.checkpoint as cp
                # Custom forward wrapper to handle the optional return_attention argument
                def block_forward(q, c):
                    out, _ = attn_block(query=q, context=c, return_attention=False)
                    return out
                attended = cp.checkpoint(block_forward, roi_cls, context_patches, use_reentrant=False)
                attn_weights = None
            else:
                attended, attn_weights = attn_block(
                    query=roi_cls,
                    context=context_patches,
                    return_attention=return_attention,
                )

            attended_features.append(attended)  # (B, D)

            if return_attention and attn_weights is not None:
                attention_maps[stream_name] = attn_weights

        # Concatenate ROI cls + all attended features
        concat = torch.cat([roi_cls] + attended_features, dim=-1)  # (B, D*(1+C))

        # Fusion FFN → context-aware embedding z
        z_context = self.fusion_ffn(concat)  # (B, output_dim)

        # Gated residual: blend ROI-only with context-enriched
        roi_projected = self.roi_proj(roi_cls)  # (B, output_dim)
        gate_input = torch.cat([roi_cls, z_context], dim=-1)
        gate_weight = self.gate(gate_input)  # (B, 1)

        z = gate_weight * z_context + (1 - gate_weight) * roi_projected  # (B, output_dim)

        result = {
            "embedding": z,
            "roi_cls": roi_cls,
        }

        if return_attention:
            result["attentions"] = attention_maps

        return result
