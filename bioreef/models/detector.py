"""
BioReef.ai — DINO Detector + FDR Head
======================================
End-to-end object detector combining:
    - DINO decoder (Zhang et al., 2022) with Contrastive DeNoising anchors
    - FDR bbox head (D-FINE, Peng et al., 2024) for distribution-based regression
    - DINOv3 ViT-B/16 patch tokens as visual features

Architecture:
    DINOv3 patch tokens (B, S, 768)
        ↓
    Feature Projection (768 → 256)
        ↓
    Transformer Decoder (6 layers):
        - N=100 learnable object queries
        - Contrastive DeNoising (CDN) during training
        - Self-attention among queries (with CDN isolation mask)
        - Cross-attention to patch features
        - Iterative box refinement per layer via FDR
        ↓
    Classification Head → (B, N, num_classes + 1)   [softmax + focal]
    FDR Bbox Head       → (B, N, 4)                 [distribution → cx,cy,w,h]

Key innovations over vanilla DETR:
    1. CDN: GT-derived noised queries provide dense supervision → 12-epoch
       convergence instead of 50+.
    2. FDR: Bbox coordinates predicted as probability distributions over
       discrete bins, capturing localization uncertainty from turbidity.
    3. Iterative refinement: each decoder layer refines the previous layer's
       box predictions rather than predicting from scratch.

Reference:
    Zhang et al. (2022), "DINO: DETR with Improved DeNoising Anchor Boxes"
    Peng et al. (2024), "D-FINE: Redefine Regression Task in DETRs as FDR"
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("bioreef.models.detector")


# =============================================================================
# Utilities
# =============================================================================

def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid (logit function)."""
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# =============================================================================
# Decoder Layer
# =============================================================================

class DINODecoderLayer(nn.Module):
    """
    Single transformer decoder layer (pre-norm):
        Self-Attention → Cross-Attention → FFN

    Self-attention operates among object queries (with CDN isolation mask).
    Cross-attention allows queries to attend to patch-token memory from DINOv3.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries:       (B, N, D)  object query content
            memory:        (B, S, D)  projected DINOv3 patch tokens
            query_pos:     (B, N, D)  positional embeddings for queries
            self_attn_mask:(N, N)     bool mask — True blocks attention (CDN isolation)
        """
        # --- Self-attention (with position added to Q, K) ---
        q = k = self.norm1(queries) + query_pos
        sa_out = self.self_attn(q, k, self.norm1(queries), attn_mask=self_attn_mask)[0]
        queries = queries + self.dropout1(sa_out)

        # --- Cross-attention (queries attend to memory) ---
        q = self.norm2(queries) + query_pos
        ca_out = self.cross_attn(q, memory, memory)[0]
        queries = queries + self.dropout2(ca_out)

        # --- FFN ---
        queries = queries + self.dropout3(self.ffn(self.norm3(queries)))
        return queries


# =============================================================================
# FDR Head — Fine-grained Distribution Refinement
# =============================================================================

class FDRHead(nn.Module):
    """
    Fine-grained Distribution Refinement head for bounding-box regression.

    Instead of directly regressing 4 coordinates, predicts a probability
    distribution over discrete bins for each coordinate (cx, cy, w, h).
    The expected value of each distribution yields the final coordinate.

    This captures localization uncertainty — critical for underwater scenes
    where object boundaries are often ambiguous due to turbidity and
    color-cast attenuation.

    Reference: Peng et al. (2024), D-FINE (ICLR 2025 Spotlight)
    """

    def __init__(self, hidden_dim: int = 256, num_bins: int = 17):
        super().__init__()
        self.num_bins = num_bins
        self.bbox_proj = MLP(hidden_dim, hidden_dim, 4 * num_bins, num_layers=3)

    def forward(
        self,
        query_features: torch.Tensor,
        reference_boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_features: (B, N, D) decoder output per query
            reference_boxes: (B, N, 4) previous layer's box prediction (sigmoid),
                             or None for the first decoder layer.

        Returns:
            distributions: (B, N, 4, num_bins) — distribution logits per coordinate
            boxes:         (B, N, 4)            — decoded [cx, cy, w, h] in [0, 1]
        """
        B, N, _ = query_features.shape
        raw = self.bbox_proj(query_features)                # (B, N, 4·num_bins)
        distributions = raw.view(B, N, 4, self.num_bins)

        # Expected value of softmax distribution → coordinate in [0, 1]
        weights = F.softmax(distributions, dim=-1)
        bins = torch.arange(self.num_bins, dtype=weights.dtype, device=weights.device)
        coords = (weights * bins).sum(dim=-1) / (self.num_bins - 1)  # (B, N, 4)

        if reference_boxes is not None:
            # Iterative refinement: add predicted offset to previous reference
            ref_inv = _inverse_sigmoid(reference_boxes)
            coords = (ref_inv + _inverse_sigmoid(coords)).sigmoid()

        return distributions, coords


# =============================================================================
# Contrastive DeNoising (CDN)
# =============================================================================

class ContrastiveDenoising(nn.Module):
    """
    Contrastive DeNoising training from the DINO detector.

    During training, generates noised copies of GT boxes/labels as extra
    decoder queries. This provides dense supervision and accelerates
    convergence from ~50 epochs (vanilla DETR) to ~12 epochs.

    - Positive queries: small noise on GT box → supervised to reconstruct GT
    - Negative queries: large noise on GT box → supervised as background

    An attention mask prevents information leakage between DN groups and
    between DN queries and the learnable queries.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 497,
        num_dn_groups: int = 5,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_dn_groups = num_dn_groups
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.label_enc = nn.Embedding(num_classes, hidden_dim)

    @torch.no_grad()
    def forward(
        self,
        targets: List[Dict[str, torch.Tensor]],
        num_queries: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Generate denoising queries from ground truth.

        Args:
            targets:     List[B] of {'labels': (M,), 'boxes': (M, 4)}
            num_queries: number of learnable object queries

        Returns:
            dn_queries: (B, num_dn, D)  embedded denoising queries
            dn_boxes:   (B, num_dn, 4)  noised reference boxes
            attn_mask:  (N_total, N_total) bool mask for self-attention
            dn_meta:    bookkeeping dict for loss computation
        """
        device = self.label_enc.weight.device
        batch_size = len(targets)
        num_gts = [len(t['labels']) for t in targets]
        max_gt = max(num_gts) if num_gts else 0

        if max_gt == 0:
            return (
                torch.zeros(batch_size, 0, self.hidden_dim, device=device),
                torch.zeros(batch_size, 0, 4, device=device),
                None,
                {'dn_num_split': [num_queries, 0], 'max_gt': 0,
                 'dn_per_group': 0, 'num_groups': 0, 'targets': targets},
            )

        # Each group: max_gt positive + max_gt negative
        dn_per_group = 2 * max_gt
        total_dn = dn_per_group * self.num_dn_groups

        dn_queries = torch.zeros(batch_size, total_dn, self.hidden_dim, device=device)
        dn_boxes = torch.zeros(batch_size, total_dn, 4, device=device)

        for b_idx in range(batch_size):
            n_gt = num_gts[b_idx]
            if n_gt == 0:
                continue

            gt_labels = targets[b_idx]['labels'].to(device)
            gt_boxes = targets[b_idx]['boxes'].to(device)

            for g in range(self.num_dn_groups):
                base = g * dn_per_group
                pos_slice = slice(base, base + n_gt)
                neg_slice = slice(base + max_gt, base + max_gt + n_gt)

                # Noised labels (shared by pos and neg)
                noised = gt_labels.clone()
                flip = torch.rand(n_gt, device=device) < self.label_noise_ratio
                noised[flip] = torch.randint(
                    0, self.num_classes, (flip.sum(),), device=device
                )

                label_embed = self.label_enc(noised)
                dn_queries[b_idx, pos_slice] = label_embed
                dn_queries[b_idx, neg_slice] = label_embed

                # Positive boxes: small noise (within 0.5× box size)
                pos_noise = (torch.rand(n_gt, 4, device=device) * 2 - 1) * 0.5
                pos_noise = pos_noise * self.box_noise_scale
                wh = gt_boxes[:, 2:].repeat(1, 2)
                dn_boxes[b_idx, pos_slice] = (gt_boxes + pos_noise * wh).clamp(0, 1)

                # Negative boxes: large noise (1–2× box size)
                neg_noise = (torch.rand(n_gt, 4, device=device) * 2 - 1) * 2.0
                neg_noise = neg_noise * self.box_noise_scale
                dn_boxes[b_idx, neg_slice] = (gt_boxes + neg_noise * wh).clamp(0, 1)

        # --- Attention mask: isolate learnable queries from each DN group ---
        total = num_queries + total_dn
        attn_mask = torch.ones(total, total, dtype=torch.bool, device=device)
        # Learnable queries attend to each other
        attn_mask[:num_queries, :num_queries] = False
        # Each DN group attends only within itself
        for g in range(self.num_dn_groups):
            s = num_queries + g * dn_per_group
            e = s + dn_per_group
            attn_mask[s:e, s:e] = False

        dn_meta = {
            'dn_num_split': [num_queries, total_dn],
            'max_gt': max_gt,
            'dn_per_group': dn_per_group,
            'num_groups': self.num_dn_groups,
            'targets': targets,
        }
        return dn_queries, dn_boxes, attn_mask, dn_meta


# =============================================================================
# BioReef Detector — Full Module
# =============================================================================

class BioReefDetector(nn.Module):
    """
    DINO-style object detector with FDR bounding-box regression.

    Takes DINOv3 patch tokens and outputs per-query class predictions
    and bounding boxes with distribution-based iterative refinement.
    """

    def __init__(
        self,
        backbone_dim: int = 768,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_classes: int = 497,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_fdr_bins: int = 17,
        num_dn_groups: int = 5,
        dn_label_noise_ratio: float = 0.5,
        dn_box_noise_scale: float = 1.0,
    ):
        """
        Args:
            backbone_dim:        DINOv3 embedding dim (768 for ViT-B/16).
            hidden_dim:          Internal dimension for decoder and heads.
            num_queries:         Learnable object queries (100 standard).
            num_classes:         Number of real species classes.
                                 Internally uses num_classes + 1 (+ background).
            num_decoder_layers:  Transformer decoder depth.
            num_heads:           Attention heads per layer.
            dim_feedforward:     FFN intermediate width.
            dropout:             Dropout rate (0 recommended for detection).
            num_fdr_bins:        Bins per coordinate for FDR distribution.
            num_dn_groups:       CDN denoising groups during training.
            dn_label_noise_ratio:Fraction of DN labels randomly flipped.
            dn_box_noise_scale:  Scale factor for DN box noise.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_output_classes = num_classes + 1  # +1 for "no object"
        self.num_decoder_layers = num_decoder_layers

        # Project backbone features (768 → hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Learnable queries: content + position (split at forward time)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DINODecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Classification head (shared across decoder layers)
        self.class_head = nn.Linear(hidden_dim, self.num_output_classes)

        # FDR bbox head (shared across decoder layers, iterative refinement)
        self.bbox_head = FDRHead(hidden_dim, num_fdr_bins)

        # Initial reference point prediction from query position
        self.ref_point_head = MLP(hidden_dim, hidden_dim, 4, num_layers=2)

        # Contrastive DeNoising
        self.cdn = ContrastiveDenoising(
            hidden_dim, num_classes, num_dn_groups,
            dn_label_noise_ratio, dn_box_noise_scale,
        )

        self._init_weights()
        logger.info(
            f"BioReefDetector: queries={num_queries}, layers={num_decoder_layers}, "
            f"hidden={hidden_dim}, classes={num_classes}(+bg), "
            f"FDR bins={num_fdr_bins}, DN groups={num_dn_groups}"
        )

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Bias init for rare-positive focal-loss regime
        prior_prob = 0.01
        nn.init.constant_(
            self.class_head.bias,
            -math.log((1 - prior_prob) / prior_prob),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, S, backbone_dim) from frozen DINOv3
            targets:      List[B] of {'labels': (M,), 'boxes': (M, 4)}
                          Required during training for CDN; None at inference.

        Returns:
            Dict with:
                pred_logits:        (B, N, num_classes+1) final class logits
                pred_boxes:         (B, N, 4)             final [cx,cy,w,h]
                pred_distributions: (B, N, 4, num_bins)   final FDR dists
                aux_outputs:        list of intermediate-layer outputs
                dn_outputs / dn_meta: CDN outputs (training only)
        """
        B = patch_tokens.shape[0]

        # --- Project backbone features ---
        memory = self.input_proj(patch_tokens)  # (B, S, hidden_dim)

        # --- Initialize learnable queries ---
        qe = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        content, pos = qe.split(self.hidden_dim, dim=-1)
        queries = content                                   # (B, N, D)
        query_pos = pos                                     # (B, N, D)
        ref_points = self.ref_point_head(pos).sigmoid()     # (B, N, 4)

        # --- Contrastive DeNoising (training only) ---
        attn_mask = None
        dn_meta = None

        if targets is not None and self.training:
            dn_queries, dn_boxes, attn_mask, dn_meta = self.cdn(
                targets, self.num_queries
            )
            if dn_queries.shape[1] > 0:
                queries = torch.cat([queries, dn_queries], dim=1)
                query_pos = torch.cat(
                    [query_pos, torch.zeros_like(dn_queries)], dim=1
                )
                ref_points = torch.cat([ref_points, dn_boxes], dim=1)

        # --- Decode through all layers ---
        all_class_preds: List[torch.Tensor] = []
        all_box_preds: List[torch.Tensor] = []
        all_distributions: List[torch.Tensor] = []

        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            queries = decoder_layer(
                queries, memory, query_pos, self_attn_mask=attn_mask,
            )

            class_pred = self.class_head(queries)
            distributions, box_pred = self.bbox_head(
                queries,
                reference_boxes=ref_points if layer_idx > 0 else None,
            )

            # Detach reference for next layer (stabilizes training)
            ref_points = box_pred.detach()

            all_class_preds.append(class_pred)
            all_box_preds.append(box_pred)
            all_distributions.append(distributions)

        # --- Split learnable vs. DN predictions ---
        nq = self.num_queries
        result: Dict[str, object] = {
            'pred_logits':        all_class_preds[-1][:, :nq],
            'pred_boxes':         all_box_preds[-1][:, :nq],
            'pred_distributions': all_distributions[-1][:, :nq],
            'aux_outputs': [
                {
                    'pred_logits':        cp[:, :nq],
                    'pred_boxes':         bp[:, :nq],
                    'pred_distributions': dp[:, :nq],
                }
                for cp, bp, dp in zip(
                    all_class_preds[:-1],
                    all_box_preds[:-1],
                    all_distributions[:-1],
                )
            ],
        }

        if dn_meta is not None and dn_meta['dn_num_split'][1] > 0:
            result['dn_outputs'] = {
                'pred_logits':        [cp[:, nq:] for cp in all_class_preds],
                'pred_boxes':         [bp[:, nq:] for bp in all_box_preds],
                'pred_distributions': [dp[:, nq:] for dp in all_distributions],
            }
            result['dn_meta'] = dn_meta

        return result
