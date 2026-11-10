"""Warm-up statistics and MAOL margin calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from maol.head import PrototypeHead
from maol.tree import ClassTree


@dataclass(frozen=True)
class WarmupStats:
    """Everything measured at the end of MAOL warm-up."""

    c_ema: torch.Tensor
    q: torch.Tensor
    node_weights: torch.Tensor
    d_intra: float
    d_inter: float
    margin_cap: float
    m_base: float
    m_max: float
    k0: float
    alpha: float
    tau0: float
    tau_warm: float
    confusion: torch.Tensor
    w_vu: torch.Tensor
    margins: torch.Tensor
    sep_weights: torch.Tensor

    def to_metadata(self) -> dict[str, object]:
        return {
            "d_intra": float(self.d_intra),
            "d_inter": float(self.d_inter),
            "margin_cap": float(self.margin_cap),
            "m_base": float(self.m_base),
            "m_max": float(self.m_max),
            "k0": float(self.k0),
            "alpha": float(self.alpha),
            "tau0": float(self.tau0),
            "tau_warm": float(self.tau_warm),
            "q": [float(value) for value in self.q.detach().cpu().tolist()],
            "c_ema": self.c_ema.detach().cpu().tolist(),
            "node_weights": [float(value) for value in self.node_weights.detach().cpu().tolist()],
            "confusion": self.confusion.detach().cpu().tolist(),
            "w_vu": [float(value) for value in self.w_vu.detach().cpu().tolist()],
            "margins": [float(value) for value in self.margins.detach().cpu().tolist()],
            "sep_weights": [float(value) for value in self.sep_weights.detach().cpu().tolist()],
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "WarmupStats":
        def tensor(key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            value = metadata[key]
            return torch.as_tensor(value, dtype=dtype)

        return cls(
            c_ema=tensor("c_ema"),
            q=tensor("q"),
            node_weights=tensor("node_weights"),
            d_intra=float(metadata["d_intra"]),
            d_inter=float(metadata["d_inter"]),
            margin_cap=float(metadata["margin_cap"]),
            m_base=float(metadata["m_base"]),
            m_max=float(metadata["m_max"]),
            k0=float(metadata["k0"]),
            alpha=float(metadata["alpha"]),
            tau0=float(metadata["tau0"]),
            tau_warm=float(metadata.get("tau_warm", metadata["tau0"])),
            confusion=tensor("confusion", dtype=torch.long),
            w_vu=tensor("w_vu"),
            margins=tensor("margins"),
            sep_weights=tensor("sep_weights"),
        )


def _prepare_input(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device=device, dtype=torch.float32)


@torch.no_grad()
def compute_class_means(
    forward_embedding: Callable[[torch.Tensor], torch.Tensor],
    loader,
    device: torch.device,
    num_classes: int,
    embedding_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return class means, counts, and mean squared distance to the frozen center.

    This is the single pass used at the start of each warm-up epoch.  It also
    provides ``tau_warm`` for the part of warm-up after the first 100 steps.
    The prototype head classifies against unit-sphere centers, so the distance
    is measured to the normalized class mean, not to the raw mean.
    """
    sums = torch.zeros((num_classes, embedding_dim), device=device, dtype=torch.float32)
    sum_squares = torch.zeros((num_classes,), device=device, dtype=torch.float32)
    counts = torch.zeros((num_classes,), device=device, dtype=torch.float32)

    for x_cpu, y_cpu in loader:
        x = _prepare_input(x_cpu, device)
        y = y_cpu.to(device=device, dtype=torch.long)
        h = forward_embedding(x).detach().float()
        for class_index in torch.unique(y).tolist():
            class_index = int(class_index)
            mask = y == class_index
            class_h = h[mask]
            sums[class_index] += class_h.sum(dim=0)
            sum_squares[class_index] += torch.sum(class_h * class_h)
            counts[class_index] += float(class_h.shape[0])

    means = sums / counts.clamp_min(1.0)[:, None]
    # tau_warm must be the mean squared distance to the centers actually used by
    # the head (normalized class means).  With unit-norm centers:
    #   sum_i ||h_i - m_hat_{y_i}||^2
    #   = sum_i ||h_i||^2 - 2 * sum_i h_i . m_hat_{y_i} + N
    # and sum_i h_i . m_hat_{y_i} = sum_c ||sum_c|| because m_hat_c = sum_c / ||sum_c||.
    total_count = counts.sum().clamp_min(1.0)
    projection = torch.linalg.vector_norm(sums, dim=-1).sum()
    total_sq = sum_squares.sum()
    tau_warm = float(((total_sq - 2.0 * projection + total_count) / total_count).clamp_min(0.0).item())
    return means, counts, tau_warm


@torch.no_grad()
def compute_warmup_statistics(
    forward_embedding: Callable[[torch.Tensor], torch.Tensor],
    head: PrototypeHead,
    train_loader,
    val_loader,
    device: torch.device,
    tree: ClassTree,
    *,
    margin_scale: float = 1.25,
    mbase_scale: float = 0.9,
    mmax_scale: float = 0.8,
    n_min: int = 5,
) -> WarmupStats:
    """Measure radii, distances, confusion weights, and margins after warm-up.

    Distances/radii come from the warm-up training set; the confusion matrix comes
    from the validation set, as required by the margin definition.
    """
    del n_min  # q is initialized directly from the warm-up distribution.
    means, counts, tau_warm = compute_class_means(
        forward_embedding,
        train_loader,
        device,
        tree.num_leaves,
        head.embedding_dim,
    )
    c_ema = F.normalize(means, p=2, dim=-1, eps=head.eps)
    head.set_centers(c_ema)
    head.tau.fill_(1.0)

    distances: list[list[float]] = [[] for _ in range(tree.num_leaves)]
    total_distance = 0.0
    total_samples = 0

    for x_cpu, y_cpu in train_loader:
        x = _prepare_input(x_cpu, device)
        y = y_cpu.to(device=device, dtype=torch.long)
        h = forward_embedding(x).detach().float()
        class_means = c_ema.to(device=h.device)
        distances_per_sample = torch.linalg.vector_norm(h - class_means[y], dim=-1)
        total_distance += float(distances_per_sample.sum().item())
        total_samples += int(y.numel())
        for class_index in torch.unique(y).tolist():
            class_index = int(class_index)
            mask = y == class_index
            distances[class_index].extend(distances_per_sample[mask].cpu().tolist())

    confusion = torch.zeros((tree.num_leaves, tree.num_leaves), dtype=torch.long)
    confusion_loader = val_loader if val_loader is not None else train_loader
    for x_cpu, y_cpu in confusion_loader:
        x = _prepare_input(x_cpu, device)
        y = y_cpu.to(device=device, dtype=torch.long)
        h = forward_embedding(x).detach().float()
        logits = head.logits(h, tau=1.0)
        predictions = torch.argmax(logits, dim=1)
        for label, prediction in zip(y.tolist(), predictions.tolist()):
            confusion[int(label), int(prediction)] += 1

    q_values = []
    for class_distances in distances:
        if not class_distances:
            q_values.append(0.0)
        elif len(class_distances) == 1:
            q_values.append(float(class_distances[0]))
        else:
            q_values.append(
                float(
                    torch.quantile(
                        torch.tensor(class_distances, dtype=torch.float32),
                        0.95,
                        interpolation="linear",
                    ).item()
                )
            )
    q = torch.tensor(q_values, dtype=torch.float32, device=device)

    d_intra = float(total_distance / max(1, total_samples))
    leaf_centers = c_ema.detach().cpu()
    leaf_pairs = tree.leaf_sibling_pairs()
    if leaf_pairs:
        pair_distances = [
            float(
                torch.linalg.vector_norm(
                    leaf_centers[int(tree.nodes[left].leaf_index)]
                    - leaf_centers[int(tree.nodes[right].leaf_index)]
                ).item()
            )
            for left, right in leaf_pairs
        ]
    else:
        pair_distances = [
            float(torch.linalg.vector_norm(leaf_centers[left] - leaf_centers[right]).item())
            for left in range(tree.num_leaves)
            for right in range(left + 1, tree.num_leaves)
        ]
    d_inter = float(np.mean(pair_distances)) if pair_distances else max(d_intra, 1e-3)

    margin_cap = max(float(margin_scale) * d_inter, 1e-3)
    m_base = max(float(mbase_scale) * d_intra, 1e-3)
    m_max = max(float(mmax_scale) * margin_cap, m_base + 1e-3)
    k0 = m_base * float(np.exp(m_base / margin_cap))
    alpha = max(m_max * float(np.exp(m_max / margin_cap)) - k0, 0.0)

    row_sums = confusion.sum(dim=1, keepdim=True).clamp_min(1)
    confusion_prob = confusion.float() / row_sums

    sibling_pairs = tree.sibling_pairs()
    w_bar: list[float] = []
    for left, right in sibling_pairs:
        left_leaves = list(tree.subtree_leaf_indices(left))
        right_leaves = list(tree.subtree_leaf_indices(right))
        block = confusion_prob[left_leaves][:, right_leaves]
        w_bar.append(float(block.mean().item()) if block.numel() else 0.0)
    w_bar_tensor = torch.tensor(w_bar, dtype=torch.float32)
    if w_bar_tensor.numel() > 0:
        q90 = float(torch.quantile(w_bar_tensor, 0.90, interpolation="linear").item())
        w_vu = torch.clamp(w_bar_tensor / (q90 + 1e-12), 0.0, 1.0)
    else:
        w_vu = torch.zeros((0,), dtype=torch.float32)

    if w_vu.numel() > 0:
        k_vu = (k0 + alpha * w_vu) / margin_cap
        margins = margin_cap * lambert_w(k_vu.numpy())
        margins = torch.tensor(margins, dtype=torch.float32)
        sep_weights = w_vu.clone()
    else:
        margins = torch.zeros((0,), dtype=torch.float32)
        sep_weights = torch.zeros((0,), dtype=torch.float32)

    node_weights = torch.tensor(
        tree.node_weights_from_labels(
            [int(value) for value in torch.repeat_interleave(
                torch.arange(tree.num_leaves),
                counts.detach().cpu().long(),
            ).tolist()]
        ),
        dtype=torch.float32,
    )

    return WarmupStats(
        c_ema=c_ema.detach().cpu(),
        q=q.detach().cpu(),
        node_weights=node_weights,
        d_intra=d_intra,
        d_inter=d_inter,
        margin_cap=margin_cap,
        m_base=m_base,
        m_max=m_max,
        k0=k0,
        alpha=alpha,
        tau0=max(d_intra * d_intra, 1e-6),
        tau_warm=tau_warm,
        confusion=confusion.cpu(),
        w_vu=w_vu.cpu(),
        margins=margins.cpu(),
        sep_weights=sep_weights.cpu(),
    )


def lambert_w(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Principal Lambert-W branch, Newton iteration (the document's method)."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    safe = np.maximum(array, 0.0)
    result = np.log1p(safe)
    for _ in range(8):
        exp_result = np.exp(result)
        denominator = exp_result * (result + 1.0)
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        result = result - (result * exp_result - safe) / denominator
    return result


def apply_warmup_stats(head: PrototypeHead, stats: WarmupStats) -> None:
    head.initialize_from_stats(stats.c_ema, stats.q, tau=stats.tau0)
    head.set_margins(stats.margins, stats.sep_weights)
    head.set_node_weights([float(value) for value in stats.node_weights])


__all__ = [
    "WarmupStats",
    "apply_warmup_stats",
    "compute_class_means",
    "compute_warmup_statistics",
    "lambert_w",
]
