"""
BioReef.ai — HSLM Loss (Hierarchical Separation-Induced Learning)
==================================================================
Wires the hierarchical family/genus/species supervision described in the
project config (`stage1.yaml: hslm`) into an actual training loss.

The species classifier emits 260-way logits. HSLM keeps that head as-is
and adds two auxiliary objectives by **marginalizing** the species
distribution up the taxonomy tree:

    p_genus[b, g]  = Σ_{s : genus(s)=g}  p_species[b, s]
    p_family[b, f] = Σ_{s : family(s)=f} p_species[b, s]

The total loss is a weighted sum of three terms:

    L = w_species · L_species  +  w_genus · L_genus  +  w_family · L_family

  - L_species : CB-Focal loss (Cui et al. 2019) — identical to the existing
                CBFocalLoss, so HSLM with w_genus=w_family=0 is a strict
                no-op equivalent. This keeps the long-tail class balancing
                where the imbalance actually lives (species level).
  - L_genus / L_family : negative log-likelihood on the marginalized
                distributions. Coarser levels are far less imbalanced, so
                plain NLL is sufficient.

Why this helps the long tail (issue / workstream W3):
    A rare species with only ~20 samples cannot be learned reliably at the
    species level. But its genus and family have plenty of samples. The
    marginalized losses give the model gradient signal at those coarser
    levels — and crucially, a within-genus confusion (predicting the wrong
    species but the right genus) barely moves L_genus, so the model is not
    punished for a biologically near-miss. A cross-family mistake, on the
    other hand, drives L_family hard.

Weight direction:
    `stage1.yaml` sets family=3, genus=2, species=1 — coarse errors are
    penalised more heavily than fine errors. This deliberately trades a few
    points of raw species Top-1 for a better Hierarchical Distance (HD),
    which is the project's primary classifier metric. The weights are
    exposed as constructor args so they can be tuned / ablated.

Reference:
    Lee et al. (2026), "MATANet" — Hierarchical Separation-Induced
    Learning Module.
    Cui et al. (2019), "Class-Balanced Loss Based on Effective Number
    of Samples."
"""

import logging
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("bioreef.losses.hslm")


class HSLMLoss(nn.Module):
    """
    Hierarchical loss over species / genus / family.

    Args:
        samples_per_class: Per-species sample counts (long-tail) — used for
                           the CB-Focal weighting of the species term.
        species_to_genus:  Length-S sequence mapping each species class
                           index to its genus index.
        species_to_family: Length-S sequence mapping each species class
                           index to its family index.
        num_genera:        Total number of distinct genera.
        num_families:      Total number of distinct families.
        family_weight:     Weight on the family term (default 3.0).
        genus_weight:      Weight on the genus term (default 2.0).
        species_weight:    Weight on the species term (default 1.0).
        beta:              CB-Focal effective-number smoothing.
        gamma:             CB-Focal focal modulation exponent.
        eps:               Floor for log() on marginalized probabilities.
        device:            Device for the registered buffers.
    """

    def __init__(
        self,
        samples_per_class: Sequence[int],
        species_to_genus: Sequence[int],
        species_to_family: Sequence[int],
        num_genera: int,
        num_families: int,
        family_weight: float = 3.0,
        genus_weight: float = 2.0,
        species_weight: float = 1.0,
        beta: float = 0.9999,
        gamma: float = 2.0,
        eps: float = 1e-8,
        device: str = "cuda",
    ):
        super().__init__()

        # --- CB-Focal class-balanced weights for the species term ---------
        spc = np.asarray(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, spc)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
        weights = weights / np.sum(weights) * len(spc)
        self.register_buffer(
            "cb_weights",
            torch.tensor(weights, dtype=torch.float32, device=device),
        )

        # --- Taxonomy index maps (species idx → genus / family idx) -------
        s2g = torch.as_tensor(species_to_genus, dtype=torch.long, device=device)
        s2f = torch.as_tensor(species_to_family, dtype=torch.long, device=device)
        assert s2g.numel() == len(spc) == s2f.numel(), (
            "species_to_genus / species_to_family must have one entry per "
            f"species ({len(spc)}); got {s2g.numel()} and {s2f.numel()}"
        )
        self.register_buffer("species_to_genus", s2g)
        self.register_buffer("species_to_family", s2f)

        self.num_genera = int(num_genera)
        self.num_families = int(num_families)
        self.gamma = gamma
        self.eps = eps
        self.w_species = species_weight
        self.w_genus = genus_weight
        self.w_family = family_weight

        # Per-component values from the most recent forward() — for logging.
        self.last_components: Dict[str, float] = {}

        logger.info(
            "HSLMLoss: %d species → %d genera → %d families | "
            "weights species=%.1f genus=%.1f family=%.1f",
            len(spc), self.num_genera, self.num_families,
            species_weight, genus_weight, family_weight,
        )

    def _marginalize(
        self,
        p_species: torch.Tensor,
        mapping: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        """
        Sum species probabilities into their parent taxonomic group.

        Args:
            p_species:  (B, S) softmax probabilities over species.
            mapping:    (S,) parent group index for each species.
            num_groups: Number of parent groups.

        Returns:
            (B, num_groups) probabilities over the parent level.
        """
        B = p_species.shape[0]
        p_group = p_species.new_zeros(B, num_groups)
        idx = mapping.unsqueeze(0).expand(B, -1)  # (B, S)
        p_group.scatter_add_(1, idx, p_species)
        return p_group

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, S) species classifier logits.
            targets: (B,) species class indices.

        Returns:
            Scalar total loss. Per-level components are stashed in
            `self.last_components` for logging.
        """
        # --- Species term: CB-Focal (identical to CBFocalLoss) ------------
        ce = F.cross_entropy(
            logits, targets, weight=self.cb_weights, reduction="none"
        )
        pt = torch.exp(-ce)
        species_loss = ((1.0 - pt) ** self.gamma * ce).mean()

        # --- Marginalize species probabilities up the taxonomy -----------
        p_species = F.softmax(logits, dim=1)
        p_genus = self._marginalize(
            p_species, self.species_to_genus, self.num_genera
        )
        p_family = self._marginalize(
            p_species, self.species_to_family, self.num_families
        )

        genus_targets = self.species_to_genus[targets]
        family_targets = self.species_to_family[targets]

        # NLL on the marginalized distributions (log of clamped probs).
        genus_loss = F.nll_loss(
            torch.log(p_genus.clamp_min(self.eps)), genus_targets
        )
        family_loss = F.nll_loss(
            torch.log(p_family.clamp_min(self.eps)), family_targets
        )

        total = (
            self.w_species * species_loss
            + self.w_genus * genus_loss
            + self.w_family * family_loss
        )

        self.last_components = {
            "species": float(species_loss.detach()),
            "genus": float(genus_loss.detach()),
            "family": float(family_loss.detach()),
            "total": float(total.detach()),
        }
        return total
