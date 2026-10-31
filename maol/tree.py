"""Class-well tree used by MAOL.

The tree is derived from the nested dataset layout (``Dataset/<superclass>/<class>/*``)
or rebuilt from checkpoint metadata.  Only leaf nodes own learnable centers and
radii; internal nodes are deterministic, weighted combinations of their children.

This module deliberately has no PyTorch dependency so it can be used by dataset
discovery, checkpoint metadata, tests, and the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TreeNode:
    """One MAOL well-tree node."""

    index: int
    name: str
    path: str
    parent: int
    children: tuple[int, ...]
    depth: int
    leaf_index: int | None = None


class ClassTree:
    """Immutable MAOL class tree.

    ``nodes[0]`` is always the root.  Leaf classes are indexed in the same order
    as ``class_names``; that order is the model's classification order.
    """

    def __init__(self, nodes: Sequence[TreeNode], class_names: Sequence[str]) -> None:
        if not nodes:
            raise ValueError("A class tree needs at least a root node")
        if nodes[0].parent != -1 or nodes[0].depth != 0:
            raise ValueError("The first tree node must be the root with parent=-1 and depth=0")

        self.nodes: tuple[TreeNode, ...] = tuple(nodes)
        self.class_names: tuple[str, ...] = tuple(str(name) for name in class_names)
        if not self.class_names:
            raise ValueError("A class tree needs at least one leaf class")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Leaf class names must be unique")

        self.root_index = 0
        self.parent: tuple[int, ...] = tuple(node.parent for node in self.nodes)
        self.children: tuple[tuple[int, ...], ...] = tuple(node.children for node in self.nodes)
        self.depth: tuple[int, ...] = tuple(node.depth for node in self.nodes)
        self.leaf_indices: tuple[int, ...] = tuple(
            node.index for node in self.nodes if node.leaf_index is not None
        )
        self.leaf_node_to_class: dict[int, int] = {
            node.index: int(node.leaf_index)
            for node in self.nodes
            if node.leaf_index is not None
        }
        self.nonroot_nodes: tuple[int, ...] = tuple(
            node.index for node in self.nodes if node.index != self.root_index
        )

        if len(self.leaf_indices) != len(self.class_names):
            raise ValueError("Leaf node count does not match class name count")
        if len(set(self.leaf_indices)) != len(self.leaf_indices):
            raise ValueError("A leaf node is assigned to more than one class")

        self._validate_structure()
        self._ancestor_chains: tuple[tuple[int, ...], ...] = tuple(
            self._build_ancestor_chain(leaf_index) for leaf_index in range(len(self.class_names))
        )
        self._subtree_leaves: tuple[tuple[int, ...], ...] = tuple(
            self._build_subtree_leaves(node.index) for node in self.nodes
        )
        self._sibling_pairs: tuple[tuple[int, int], ...] = self._build_sibling_pairs()
        self._leaf_sibling_pairs: tuple[tuple[int, int], ...] = tuple(
            (left, right)
            for left, right in self._sibling_pairs
            if self.nodes[left].leaf_index is not None
            and self.nodes[right].leaf_index is not None
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_leaf_paths(cls, leaf_paths: Iterable[str]) -> "ClassTree":
        """Build a tree from relative leaf directory paths such as ``animals/cat``."""
        normalized: list[tuple[str, ...]] = []
        for raw_path in leaf_paths:
            path = str(raw_path).replace("\\", "/").strip("/")
            parts = tuple(part for part in path.split("/") if part)
            if not parts:
                raise ValueError("Leaf path cannot be empty")
            normalized.append(parts)
        if not normalized:
            raise ValueError("At least one leaf path is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate leaf paths are not allowed")

        nodes: list[TreeNode] = [
            TreeNode(index=0, name="root", path="", parent=-1, children=(), depth=0)
        ]
        path_to_index: dict[tuple[str, ...], int] = {(): 0}
        children_by_parent: dict[int, list[int]] = {0: []}

        for parts in normalized:
            parent = 0
            for depth in range(1, len(parts) + 1):
                prefix = parts[:depth]
                index = path_to_index.get(prefix)
                if index is None:
                    index = len(nodes)
                    nodes.append(
                        TreeNode(
                            index=index,
                            name=prefix[-1],
                            path="/".join(prefix),
                            parent=parent,
                            children=(),
                            depth=depth,
                        )
                    )
                    path_to_index[prefix] = index
                    children_by_parent.setdefault(index, [])
                    children_by_parent[parent].append(index)
                parent = index

        # A path that is both a leaf and an ancestor of another leaf is ambiguous.
        leaf_path_set = set(normalized)
        for parts in normalized:
            for depth in range(1, len(parts)):
                if parts[:depth] in leaf_path_set:
                    raise ValueError(
                        "A directory cannot contain both images and nested class directories: "
                        + "/".join(parts[:depth])
                    )

        sorted_leaves = sorted(normalized, key=lambda parts: "/".join(parts))
        class_names = [parts[-1] for parts in sorted_leaves]
        duplicates: dict[str, list[str]] = {}
        for parts in sorted_leaves:
            duplicates.setdefault(parts[-1], []).append("/".join(parts))
        repeated = {name: paths for name, paths in duplicates.items() if len(paths) > 1}
        if repeated:
            detail = "; ".join(f"{name}: {', '.join(paths)}" for name, paths in repeated.items())
            raise ValueError(
                "Leaf directory names must be unique because they are used as class labels. "
                f"Rename one of: {detail}"
            )

        class_by_path = {
            "/".join(parts): class_index for class_index, parts in enumerate(sorted_leaves)
        }
        leaf_nodes: dict[str, int] = {
            "/".join(parts): path_to_index[parts] for parts in sorted_leaves
        }

        # Rebuild nodes with deterministic child ordering and leaf indices.
        rebuilt: list[TreeNode] = []
        for node in nodes:
            children = tuple(
                sorted(children_by_parent.get(node.index, []), key=lambda child: nodes[child].name)
            )
            leaf_index = (
                class_by_path[node.path] if node.index in set(leaf_nodes.values()) else None
            )
            rebuilt.append(
                TreeNode(
                    index=node.index,
                    name=node.name,
                    path=node.path,
                    parent=node.parent,
                    children=children,
                    depth=node.depth,
                    leaf_index=leaf_index,
                )
            )
        return cls(rebuilt, class_names)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ClassTree":
        """Rebuild a tree from checkpoint metadata."""
        if not isinstance(payload, Mapping):
            raise ValueError("Tree metadata must be a mapping")
        raw_nodes = payload.get("nodes")
        raw_class_names = payload.get("class_names")
        if not isinstance(raw_nodes, (list, tuple)) or not raw_nodes:
            raise ValueError("Tree metadata is missing `nodes`")
        if not isinstance(raw_class_names, (list, tuple)) or not raw_class_names:
            raise ValueError("Tree metadata is missing `class_names`")
        nodes: list[TreeNode] = []
        for raw in raw_nodes:
            if not isinstance(raw, Mapping):
                raise ValueError("Tree node metadata must be a mapping")
            nodes.append(
                TreeNode(
                    index=int(raw["index"]),
                    name=str(raw["name"]),
                    path=str(raw.get("path", "")),
                    parent=int(raw["parent"]),
                    children=tuple(int(child) for child in raw.get("children", ())),
                    depth=int(raw["depth"]),
                    leaf_index=(
                        None if raw.get("leaf_index") is None else int(raw["leaf_index"])
                    ),
                )
            )
        return cls(nodes, [str(name) for name in raw_class_names])

    # ------------------------------------------------------------------
    # Validation and derived lookups
    # ------------------------------------------------------------------
    def _validate_structure(self) -> None:
        for node in self.nodes:
            if node.index != self.nodes.index(node):
                raise ValueError("Tree node indices must match their list position")
            if node.index == self.root_index:
                continue
            if not (0 <= node.parent < len(self.nodes)):
                raise ValueError(f"Node {node.path!r} has an invalid parent")
            if node.index not in self.children[node.parent]:
                raise ValueError(f"Node {node.path!r} is missing from its parent's children")
            if self.depth[node.parent] + 1 != node.depth:
                raise ValueError(f"Node {node.path!r} has an inconsistent depth")
            if self.nodes[node.parent].leaf_index is not None:
                raise ValueError(f"Leaf node {self.nodes[node.parent].path!r} cannot have children")
        if self.nodes[0].parent != -1:
            raise ValueError("Root must not have a parent")

    def _build_ancestor_chain(self, leaf_index: int) -> tuple[int, ...]:
        node_index = self.leaf_indices[leaf_index]
        chain: list[int] = []
        parent = self.parent[node_index]
        while parent != -1:
            chain.append(parent)
            parent = self.parent[parent]
        return tuple(chain)

    def _build_subtree_leaves(self, node_index: int) -> tuple[int, ...]:
        node = self.nodes[node_index]
        if node.leaf_index is not None:
            return (int(node.leaf_index),)
        leaves: list[int] = []
        for child in node.children:
            leaves.extend(self._build_subtree_leaves(child))
        return tuple(sorted(leaves))

    def _build_sibling_pairs(self) -> tuple[tuple[int, int], ...]:
        pairs: list[tuple[int, int]] = []
        for node in self.nodes:
            for left, right in combinations(node.children, 2):
                pairs.append((int(left), int(right)))
        return tuple(pairs)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def ancestor_nodes(self, leaf_index: int) -> tuple[int, ...]:
        """Parent-to-root node indices for one leaf class."""
        return self._ancestor_chains[int(leaf_index)]

    def subtree_leaf_indices(self, node_index: int) -> tuple[int, ...]:
        return self._subtree_leaves[int(node_index)]

    def sibling_pairs(self, *, leaves_only: bool = False) -> tuple[tuple[int, int], ...]:
        """Unique unordered sibling node pairs, at every tree level by default."""
        return self._leaf_sibling_pairs if leaves_only else self._sibling_pairs

    def leaf_sibling_pairs(self) -> tuple[tuple[int, int], ...]:
        return self._leaf_sibling_pairs

    def node_weights_from_labels(self, labels: Sequence[int] | Iterable[int]) -> list[float]:
        """Return ``n_u`` for every node from training labels."""
        counts = [0] * len(self.class_names)
        for label in labels:
            label = int(label)
            if not (0 <= label < len(self.class_names)):
                raise ValueError(f"Label {label} is outside the class range")
            counts[label] += 1
        weights: list[float] = []
        for node in self.nodes:
            weights.append(float(sum(counts[leaf] for leaf in self._subtree_leaves[node.index])))
        return weights

    def class_depth(self, leaf_index: int) -> int:
        return int(self.depth[self.leaf_indices[int(leaf_index)]])

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "class_names": list(self.class_names),
            "nodes": [
                {
                    "index": node.index,
                    "name": node.name,
                    "path": node.path,
                    "parent": node.parent,
                    "children": list(node.children),
                    "depth": node.depth,
                    "leaf_index": node.leaf_index,
                }
                for node in self.nodes
            ],
        }

    @property
    def num_leaves(self) -> int:
        return len(self.class_names)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def describe(self) -> str:
        lines: list[str] = []
        for node in self.nodes:
            indent = "  " * node.depth
            suffix = "" if node.leaf_index is None else f"  [class {node.leaf_index}: {self.class_names[node.leaf_index]}]"
            lines.append(f"{indent}{node.name}{suffix}")
        return "\n".join(lines)


__all__ = ["ClassTree", "TreeNode"]
