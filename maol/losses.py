"""MAOL loss composition.

All terms follow the denominators in the updated theory document:

* ``L_CE`` and ``L_intra``      -> mean over the batch
* ``L_sep``                     -> mean over valid sibling pairs
* ``L_cov`` and ``L_center``    -> mean over leaf classes present in the batch
* ``L_nest``                    -> mean over non-root tree nodes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from maol.head import PrototypeHead


@dataclass(frozen=True)
class MAOLLossTerms:
    total: torch.Tensor
    ce: torch.Tensor
    intra: torch.Tensor
    sep: torch.Tensor
    nest: torch.Tensor
    cov: torch.Tensor
    center: torch.Tensor
    logits: torch.Tensor

    def named(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "ce": self.ce,
            "intra": self.intra,
            "sep": self.sep,
            "nest": self.nest,
            "cov": self.cov,
            "center": self.center,
        }


def _cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if class_weights is None:
        return F.cross_entropy(logits, labels, reduction="mean")
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    weights = class_weights.to(device=logits.device, dtype=logits.dtype)[labels]
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)


def compute_maol_losses(
    head: PrototypeHead,
    h: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambdas: Mapping[str, float],
    gamma: float = 0.5,
    eta: float = 0.1,
    class_weights: torch.Tensor | None = None,
) -> MAOLLossTerms:
    """Compute the six MAOL terms and their weighted total for one batch."""
    if h.ndim != 2:
        raise ValueError("MAOL expects a 2-D embedding batch")
    if labels.ndim != 1 or labels.shape[0] != h.shape[0]:
        raise ValueError("labels must have shape (batch,) and match h")

    logits = head.logits(h)
    ce = _cross_entropy(logits, labels, class_weights)
    intra = head.intra_loss(h, labels, gamma=gamma)
    sep = head.separation_loss(h, labels, eta=eta)
    nest = head.nest_loss()
    cov = head.coverage_loss(labels)
    center = head.center_loss(labels)

    total = (
        ce
        + float(lambdas["intra"]) * intra
        + float(lambdas["sep"]) * sep
        + float(lambdas["nest"]) * nest
        + float(lambdas["cov"]) * cov
        + float(lambdas["center"]) * center
    )
    return MAOLLossTerms(
        total=total,
        ce=ce,
        intra=intra,
        sep=sep,
        nest=nest,
        cov=cov,
        center=center,
        logits=logits,
    )


__all__ = ["MAOLLossTerms", "compute_maol_losses"]
