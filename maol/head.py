"""MAOL prototype head: leaf wells, deterministic parent geometry, and losses."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from maol.tree import ClassTree


class PrototypeHead(nn.Module):
    """Prototype classifier and well geometry for one MAOL class tree.

    The model passes a unit-norm embedding ``h``; centers are free parameters
    normalized on read.  Only leaf centers and leaf radii are learnable;
    internal centers/radii are derived bottom-up from child centers and the
    training sample counts ``n_u``.
    """

    def __init__(
        self,
        tree: ClassTree,
        embedding_dim: int,
        *,
        node_weights: Sequence[float] | None = None,
        tau: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        embedding_dim = int(embedding_dim)
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        self.tree = tree
        self.embedding_dim = embedding_dim
        self.eps = float(eps)
        self.num_leaves = tree.num_leaves
        self.num_nodes = tree.num_nodes

        self.centers = nn.Parameter(torch.zeros(self.num_leaves, embedding_dim), requires_grad=False)
        self.raw_radii = nn.Parameter(torch.zeros(self.num_leaves), requires_grad=False)
        self.register_buffer("c_ema", torch.zeros(self.num_leaves, embedding_dim))
        self.register_buffer("q", torch.full((self.num_leaves,), 0.1))
        weights = (
            torch.ones(self.num_nodes, dtype=torch.float32)
            if node_weights is None
            else torch.as_tensor(list(node_weights), dtype=torch.float32)
        )
        if weights.numel() != self.num_nodes:
            raise ValueError("node_weights must contain one value per tree node")
        self.register_buffer("node_weights", weights.clamp_min(0.0))
        self.register_buffer("tau", torch.tensor(float(tau), dtype=torch.float32))

        pairs = tree.sibling_pairs()
        self.register_buffer("margins", torch.zeros(len(pairs), dtype=torch.float32))
        self.register_buffer("sep_weights", torch.ones(len(pairs), dtype=torch.float32))

        # Static tree indexing.  Non-persistent buffers do not appear in checkpoints.
        max_ancestors = max((len(tree.ancestor_nodes(i)) for i in range(self.num_leaves)), default=1)
        ancestor_nodes = torch.full((self.num_leaves, max_ancestors), -1, dtype=torch.long)
        ancestor_exponents = torch.zeros((self.num_leaves, max_ancestors), dtype=torch.float32)
        for leaf_index in range(self.num_leaves):
            for position, node_index in enumerate(tree.ancestor_nodes(leaf_index)):
                ancestor_nodes[leaf_index, position] = int(node_index)
                ancestor_exponents[leaf_index, position] = float(
                    tree.depth[tree.leaf_indices[leaf_index]] - tree.depth[node_index]
                )
        self.register_buffer("_ancestor_nodes", ancestor_nodes, persistent=False)
        self.register_buffer("_ancestor_exponents", ancestor_exponents, persistent=False)

        leaf_to_node = torch.tensor(
            [int(tree.leaf_indices[i]) for i in range(self.num_leaves)], dtype=torch.long
        )
        self.register_buffer("_leaf_to_node", leaf_to_node, persistent=False)

        if pairs:
            pair_tensor = torch.tensor(pairs, dtype=torch.long)
        else:
            pair_tensor = torch.empty((0, 2), dtype=torch.long)
        self.register_buffer("_sibling_pairs", pair_tensor, persistent=False)

        subtree_mask = torch.zeros((self.num_nodes, self.num_leaves), dtype=torch.float32)
        for node_index in range(self.num_nodes):
            for leaf_index in tree.subtree_leaf_indices(node_index):
                subtree_mask[node_index, leaf_index] = 1.0
        self.register_buffer("_subtree_mask", subtree_mask, persistent=False)
        self._compute_order = tuple(
            sorted(range(self.num_nodes), key=lambda index: tree.depth[index], reverse=True)
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @property
    def radii(self) -> torch.Tensor:
        return F.softplus(self.raw_radii) + self.eps

    @property
    def centers_hat(self) -> torch.Tensor:
        """Unit-sphere centers used by the prototype head and all geometry losses."""
        return F.normalize(self.centers, p=2, dim=-1, eps=self.eps)

    def set_trainable(self, trainable: bool) -> None:
        requires_grad = bool(trainable)
        self.centers.requires_grad_(requires_grad)
        self.raw_radii.requires_grad_(requires_grad)

    def set_centers(self, means: torch.Tensor) -> None:
        if tuple(means.shape) != tuple(self.centers.shape):
            raise ValueError(f"Expected centers of shape {tuple(self.centers.shape)}, got {tuple(means.shape)}")
        with torch.no_grad():
            normalized = F.normalize(
                means.detach().to(device=self.centers.device, dtype=self.centers.dtype),
                p=2,
                dim=-1,
                eps=self.eps,
            )
            self.centers.copy_(normalized)

    def initialize_from_stats(
        self,
        c_ema: torch.Tensor,
        q: torch.Tensor,
        *,
        tau: float,
    ) -> None:
        c_ema = c_ema.detach().to(device=self.centers.device, dtype=self.centers.dtype)
        q = q.detach().to(device=self.q.device, dtype=self.q.dtype).clamp_min(self.eps)
        with torch.no_grad():
            self.c_ema.copy_(c_ema)
            self.q.copy_(q)
            self.centers.copy_(F.normalize(c_ema, p=2, dim=-1, eps=self.eps))
            self.raw_radii.copy_(torch.log(torch.expm1(q)))
            self.tau.fill_(float(max(tau, self.eps)))

    def set_margins(self, margins: torch.Tensor, sep_weights: torch.Tensor) -> None:
        if margins.numel() != self.margins.numel() or sep_weights.numel() != self.sep_weights.numel():
            raise ValueError("Margin tensors do not match the number of sibling pairs")
        with torch.no_grad():
            self.margins.copy_(margins.detach().to(self.margins.device, self.margins.dtype))
            self.sep_weights.copy_(sep_weights.detach().to(self.sep_weights.device, self.sep_weights.dtype))

    def set_node_weights(self, weights: Sequence[float]) -> None:
        tensor = torch.as_tensor(list(weights), dtype=self.node_weights.dtype, device=self.node_weights.device)
        if tensor.numel() != self.num_nodes:
            raise ValueError("node_weights must contain one value per tree node")
        with torch.no_grad():
            self.node_weights.copy_(tensor.clamp_min(0.0))

    def node_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(centers, radii)`` for every node, with gradient paths to leaves."""
        dtype = self.centers.dtype
        device = self.centers.device
        leaf_centers = self.centers_hat
        all_centers: list[torch.Tensor | None] = [None] * self.num_nodes
        all_radii: list[torch.Tensor | None] = [None] * self.num_nodes

        for node_index in self._compute_order:
            node = self.tree.nodes[node_index]
            if node.leaf_index is not None:
                all_centers[node_index] = leaf_centers[node.leaf_index]
                all_radii[node_index] = self.radii[node.leaf_index]
                continue

            child_indices = list(node.children)
            if not child_indices:
                raise RuntimeError(f"Internal node {node.path!r} has no children")
            child_centers = torch.stack([all_centers[child] for child in child_indices])  # type: ignore[list-item]
            child_radii = torch.stack([all_radii[child] for child in child_indices])  # type: ignore[list-item]
            weights = self.node_weights[child_indices].to(dtype=dtype, device=device)
            denominator = weights.sum().clamp_min(self.eps)
            center = (weights[:, None] * child_centers).sum(dim=0) / denominator
            distances = torch.linalg.vector_norm(child_centers - center, dim=-1)
            radius = torch.max(distances + child_radii) + self.eps
            all_centers[node_index] = center
            all_radii[node_index] = radius

        return (
            torch.stack([center for center in all_centers if center is not None]),
            torch.stack([radius for radius in all_radii if radius is not None]),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def logits(self, h: torch.Tensor, *, tau: float | None = None) -> torch.Tensor:
        temperature = self.tau if tau is None else torch.as_tensor(float(tau), device=h.device, dtype=h.dtype)
        centers = self.centers_hat.to(dtype=h.dtype)
        h2 = torch.sum(h * h, dim=-1, keepdim=True)
        c2 = torch.sum(centers * centers, dim=-1).unsqueeze(0)
        cross = h @ centers.transpose(0, 1)
        distance_sq = (h2 + c2 - 2.0 * cross).clamp_min(0.0)
        return -distance_sq / temperature.clamp_min(self.eps)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.logits(h)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def intra_loss(self, h: torch.Tensor, labels: torch.Tensor, *, gamma: float = 0.5) -> torch.Tensor:
        leaf_centers = self.centers_hat[labels]
        loss = torch.sum((h - leaf_centers) ** 2, dim=-1)

        ancestor_nodes = self._ancestor_nodes[labels].clamp_min(0)
        ancestor_beta = torch.pow(
            torch.as_tensor(float(gamma), device=h.device, dtype=h.dtype),
            self._ancestor_exponents[labels].to(device=h.device, dtype=h.dtype),
        )
        mask = (self._ancestor_nodes[labels] >= 0).to(dtype=h.dtype)
        all_centers, _ = self.node_geometry()
        ancestor_centers = all_centers[ancestor_nodes]
        ancestor_distance_sq = torch.sum((h[:, None, :] - ancestor_centers) ** 2, dim=-1)
        loss = loss + torch.sum(ancestor_beta * ancestor_distance_sq * mask, dim=-1)
        return loss.mean()

    def separation_loss(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        *,
        eta: float = 0.1,
    ) -> torch.Tensor:
        pairs = self._sibling_pairs
        if pairs.numel() == 0:
            return torch.zeros((), device=h.device, dtype=h.dtype)

        one_hot = F.one_hot(labels, num_classes=self.num_leaves).to(dtype=h.dtype)
        leaf_counts = one_hot.sum(dim=0)
        leaf_sums = one_hot.transpose(0, 1) @ h
        node_sums = self._subtree_mask.to(dtype=h.dtype) @ leaf_sums
        node_counts = self._subtree_mask.to(dtype=h.dtype) @ leaf_counts
        node_means = node_sums / node_counts.clamp_min(1.0)[:, None]

        left, right = pairs[:, 0], pairs[:, 1]
        valid = (node_counts[left] > 0) & (node_counts[right] > 0)
        if not bool(valid.any()):
            return torch.zeros((), device=h.device, dtype=h.dtype)

        all_centers, _ = self.node_geometry()
        batch_distance = torch.linalg.vector_norm(node_means[left] - node_means[right], dim=-1)
        center_distance = torch.linalg.vector_norm(all_centers[left] - all_centers[right], dim=-1)
        margins = self.margins.to(dtype=h.dtype)
        weights = self.sep_weights.to(dtype=h.dtype)
        term = weights * (
            F.relu(margins - batch_distance) ** 2
            + float(eta) * F.relu(margins - center_distance) ** 2
        )
        return term[valid].mean()

    def nest_loss(self) -> torch.Tensor:
        all_centers, all_radii = self.node_geometry()
        values: list[torch.Tensor] = []
        for node_index in self.tree.nonroot_nodes:
            parent = self.tree.parent[node_index]
            parent_center = all_centers[parent].detach()
            parent_radius = all_radii[parent].detach()
            child_center = all_centers[node_index]
            child_radius = all_radii[node_index]
            gap = torch.linalg.vector_norm(child_center - parent_center) + child_radius - parent_radius
            values.append(F.relu(gap) ** 2)
        if not values:
            return torch.zeros((), device=all_centers.device, dtype=all_centers.dtype)
        return torch.stack(values).mean()

    def coverage_loss(self, labels: torch.Tensor) -> torch.Tensor:
        present = torch.unique(labels)
        if present.numel() == 0:
            return torch.zeros((), device=labels.device, dtype=self.centers.dtype)
        radii = self.radii[present]
        gaps = self.q.to(device=radii.device, dtype=radii.dtype)[present] - radii
        return (F.relu(gaps) ** 2).mean()

    def center_loss(self, labels: torch.Tensor) -> torch.Tensor:
        present = torch.unique(labels)
        if present.numel() == 0:
            return torch.zeros((), device=labels.device, dtype=self.centers.dtype)
        gaps = self.centers_hat[present] - self.c_ema.to(device=self.centers.device, dtype=self.centers.dtype)[present]
        return torch.sum(gaps * gaps, dim=-1).mean()

    @torch.no_grad()
    def update_ema_stats(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        *,
        m: float = 0.9,
        m_q: float = 0.9,
        n_min: int = 5,
    ) -> None:
        """Update ``c_EMA`` and the 95th-percentile radius statistic for classes in this batch."""
        for class_index in torch.unique(labels).tolist():
            class_index = int(class_index)
            mask = labels == class_index
            count = int(mask.sum().item())
            if count < int(n_min):
                continue
            class_h = h[mask].detach().float()
            batch_mean = class_h.mean(dim=0)
            distances = torch.linalg.vector_norm(class_h - self.c_ema[class_index].float(), dim=-1)
            batch_q = torch.quantile(distances, 0.95, interpolation="linear")
            self.q[class_index] = float(m_q) * self.q[class_index] + (1.0 - float(m_q)) * batch_q
            updated_mean = (
                float(m) * self.c_ema[class_index] + (1.0 - float(m)) * batch_mean
            )
            self.c_ema[class_index] = F.normalize(
                updated_mean.unsqueeze(0),
                p=2,
                dim=-1,
                eps=self.eps,
            )[0]

    def extra_repr(self) -> str:
        return (
            f"num_leaves={self.num_leaves}, num_nodes={self.num_nodes}, "
            f"embedding_dim={self.embedding_dim}, tau={float(self.tau):.4f}"
        )


__all__ = ["PrototypeHead"]
