"""
BioReef.ai — Detection Loss Functions
=======================================
Combines:
    1. Hungarian matching (Kuhn, 1955) for bipartite query-to-GT assignment
    2. Focal Loss (Lin et al., 2017) for foreground classification
    3. GIoU Loss (Rezatofighi et al., 2019) for box regression
    4. Distribution Focal Loss (Li et al., 2020) for FDR distributions
    5. CDN denoising loss for accelerated convergence

The "no object" (background) class is the last class index (num_classes).
Unmatched queries are trained to predict this class with a down-weighted
loss factor (eos_coef) to prevent gradient dominance.

Reference:
    Carion et al. (2020), "End-to-End Object Detection with Transformers"
    Li et al. (2020), "Generalized Focal Loss"
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("bioreef.losses.detection")


# =============================================================================
# Box Utilities
# =============================================================================

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x0, y0, x1, y1)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise Generalized IoU between two sets of boxes (xyxy format).

    Args:
        boxes1, boxes2: (..., 4) in (x0, y0, x1, y1) format

    Returns:
        giou: (...,) values in [-1, 1]
    """
    inter_x0 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y0 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x1 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y1 = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter = (inter_x1 - inter_x0).clamp(min=0) * (inter_y1 - inter_y0).clamp(min=0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter

    iou = inter / union.clamp(min=1e-6)

    enc_x0 = torch.min(boxes1[..., 0], boxes2[..., 0])
    enc_y0 = torch.min(boxes1[..., 1], boxes2[..., 1])
    enc_x1 = torch.max(boxes1[..., 2], boxes2[..., 2])
    enc_y1 = torch.max(boxes1[..., 3], boxes2[..., 3])
    enc_area = (enc_x1 - enc_x0) * (enc_y1 - enc_y0)

    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)


def pairwise_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise GIoU matrix.

    Args:
        boxes1: (N, 4)  boxes2: (M, 4)   both in xyxy format

    Returns:
        giou_matrix: (N, M)
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    b1 = boxes1.unsqueeze(1).expand(N, M, 4)
    b2 = boxes2.unsqueeze(0).expand(N, M, 4)
    return generalized_iou(b1, b2)


# =============================================================================
# Hungarian Matcher
# =============================================================================

class HungarianMatcher(nn.Module):
    """
    Bipartite matching between predictions and ground truth using
    the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

    Cost = λ_cls · cls_cost + λ_box · L1_cost + λ_giou · giou_cost
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            pred_logits: (B, N, num_classes+1) — softmax classification logits
            pred_boxes:  (B, N, 4)             — predicted [cx, cy, w, h] in [0,1]
            targets:     List[B] of {'labels': (M,), 'boxes': (M, 4)}

        Returns:
            List[B] of (pred_indices, target_indices) matched pairs
        """
        B, N = pred_logits.shape[:2]
        indices = []

        for b in range(B):
            n_gt = len(targets[b]['labels'])
            if n_gt == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long),
                ))
                continue

            # Classification cost: negative probability of correct class
            out_prob = pred_logits[b].softmax(-1)                   # (N, C+1)
            cost_class = -out_prob[:, targets[b]['labels']]         # (N, M)

            # L1 box cost
            tgt_boxes = targets[b]['boxes'].to(pred_boxes.device)
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)  # (N, M)

            # GIoU cost
            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            cost_giou = -pairwise_giou(pred_xyxy, tgt_xyxy)        # (N, M)

            C = (self.cost_class * cost_class
                 + self.cost_bbox * cost_bbox
                 + self.cost_giou * cost_giou)

            row, col = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.tensor(row, dtype=torch.long),
                torch.tensor(col, dtype=torch.long),
            ))

        return indices


# =============================================================================
# Distribution Focal Loss (DFL)
# =============================================================================

def distribution_focal_loss(
    pred_dist: torch.Tensor,
    target_val: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """
    Distribution Focal Loss from Generalized Focal Loss (Li et al., 2020).

    Supervises the FDR distribution to peak around the continuous target
    value by computing weighted CE to the two nearest bin indices.

    Args:
        pred_dist:  (*, num_bins) distribution logits
        target_val: (*,)          target values in [0, 1]
        num_bins:   number of distribution bins
    """
    target_scaled = target_val.float() * (num_bins - 1)
    left = target_scaled.long().clamp(0, num_bins - 2)
    right = left + 1

    weight_right = target_scaled - left.float()
    weight_left = 1.0 - weight_right

    loss_left = F.cross_entropy(pred_dist, left, reduction='none')
    loss_right = F.cross_entropy(pred_dist, right, reduction='none')

    return (loss_left * weight_left + loss_right * weight_right).mean()


# =============================================================================
# Combined Detection Loss
# =============================================================================

class DetectionLoss(nn.Module):
    """
    Combined detection loss with Hungarian matching.

    Components:
        - Weighted CE for classification (eos_coef for background)
        - L1 + GIoU for box regression (matched queries only)
        - DFL for FDR distributions (matched queries only)
        - Auxiliary losses from intermediate decoder layers
        - CDN denoising losses (training only)
    """

    def __init__(
        self,
        num_classes: int,
        num_fdr_bins: int = 17,
        eos_coef: float = 0.1,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        weight_cls: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        weight_dfl: float = 1.5,
        weight_dn: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_output_classes = num_classes + 1
        self.num_fdr_bins = num_fdr_bins
        self.weight_cls = weight_cls
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_dfl = weight_dfl
        self.weight_dn = weight_dn

        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)

        # Down-weight background class to prevent gradient dominance
        empty_weight = torch.ones(self.num_output_classes)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _loss_for_layer(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_dist: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute per-layer losses given matched indices."""
        device = pred_logits.device
        B, N = pred_logits.shape[:2]

        # --- Classification loss (all queries) ---
        # Default target: background (= num_classes)
        target_classes = torch.full(
            (B, N), self.num_classes, dtype=torch.long, device=device,
        )
        for b, (pi, ti) in enumerate(indices):
            if len(pi) > 0:
                target_classes[b, pi.to(device)] = targets[b]['labels'][ti].to(device)

        loss_cls = F.cross_entropy(
            pred_logits.transpose(1, 2),  # (B, C+1, N)
            target_classes,               # (B, N)
            weight=self.empty_weight,
        ) * self.weight_cls

        # --- Box losses (matched queries only) ---
        matched_pred_boxes = []
        matched_tgt_boxes = []
        matched_pred_dist = []
        matched_tgt_vals = []

        for b, (pi, ti) in enumerate(indices):
            if len(pi) == 0:
                continue
            matched_pred_boxes.append(pred_boxes[b, pi.to(device)])
            matched_tgt_boxes.append(targets[b]['boxes'][ti].to(device))
            matched_pred_dist.append(pred_dist[b, pi.to(device)])

        if matched_pred_boxes:
            all_pred = torch.cat(matched_pred_boxes)
            all_tgt = torch.cat(matched_tgt_boxes)
            all_dist = torch.cat(matched_pred_dist)

            # L1 loss
            loss_bbox = F.l1_loss(all_pred, all_tgt) * self.weight_bbox

            # GIoU loss
            giou = generalized_iou(
                box_cxcywh_to_xyxy(all_pred),
                box_cxcywh_to_xyxy(all_tgt),
            )
            loss_giou = (1 - giou).mean() * self.weight_giou

            # DFL loss (flatten coords → per-coord distribution)
            flat_dist = all_dist.reshape(-1, self.num_fdr_bins)
            flat_tgt = all_tgt.reshape(-1)
            loss_dfl = distribution_focal_loss(
                flat_dist, flat_tgt, self.num_fdr_bins,
            ) * self.weight_dfl
        else:
            z = torch.tensor(0.0, device=device)
            loss_bbox, loss_giou, loss_dfl = z, z, z

        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'loss_dfl': loss_dfl,
        }

    def _dn_loss(
        self,
        outputs: Dict,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute CDN denoising loss (no matching — direct supervision)."""
        dn = outputs['dn_outputs']
        meta = outputs['dn_meta']
        device = dn['pred_logits'][0].device
        max_gt = meta['max_gt']
        dn_per_group = meta['dn_per_group']
        num_groups = meta['num_groups']

        # Use last decoder layer's DN predictions
        dn_logits = dn['pred_logits'][-1]   # (B, total_dn, C+1)
        dn_boxes = dn['pred_boxes'][-1]     # (B, total_dn, 4)

        cls_losses = []
        box_losses = []
        B = len(targets)

        for b in range(B):
            n_gt = len(targets[b]['labels'])
            if n_gt == 0:
                continue

            gt_labels = targets[b]['labels'].to(device)
            gt_boxes = targets[b]['boxes'].to(device)

            for g in range(num_groups):
                base = g * dn_per_group
                # Positive queries → predict GT
                pos = dn_logits[b, base:base + n_gt]
                cls_losses.append(F.cross_entropy(pos, gt_labels))
                box_losses.append(F.l1_loss(dn_boxes[b, base:base + n_gt], gt_boxes))

                # Negative queries → predict background
                neg = dn_logits[b, base + max_gt:base + max_gt + n_gt]
                bg = torch.full((n_gt,), self.num_classes, dtype=torch.long, device=device)
                cls_losses.append(F.cross_entropy(neg, bg))

        if cls_losses:
            return {
                'loss_dn_cls': torch.stack(cls_losses).mean(),
                'loss_dn_box': torch.stack(box_losses).mean(),
            }
        z = torch.tensor(0.0, device=device)
        return {'loss_dn_cls': z, 'loss_dn_box': z}

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total detection loss.

        Args:
            outputs: from BioReefDetector.forward()
            targets: List[B] of {'labels': (M,), 'boxes': (M, 4)}

        Returns:
            Dict of loss components + 'total_loss'
        """
        # --- Main loss (last decoder layer) ---
        indices = self.matcher(
            outputs['pred_logits'], outputs['pred_boxes'], targets,
        )
        losses = self._loss_for_layer(
            outputs['pred_logits'],
            outputs['pred_boxes'],
            outputs['pred_distributions'],
            targets, indices,
        )

        # --- Auxiliary losses (intermediate layers) ---
        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                aux_idx = self.matcher(aux['pred_logits'], aux['pred_boxes'], targets)
                aux_l = self._loss_for_layer(
                    aux['pred_logits'], aux['pred_boxes'],
                    aux['pred_distributions'], targets, aux_idx,
                )
                for k, v in aux_l.items():
                    losses[f'{k}_aux{i}'] = v

        # --- CDN denoising loss ---
        if 'dn_outputs' in outputs and 'dn_meta' in outputs:
            dn_l = self._dn_loss(outputs, targets)
            for k, v in dn_l.items():
                losses[k] = v * self.weight_dn

        losses['total_loss'] = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        return losses
