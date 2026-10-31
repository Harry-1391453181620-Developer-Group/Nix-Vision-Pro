"""Tests for the MAOL class tree."""

from __future__ import annotations

from maol.tree import ClassTree


def _tree() -> ClassTree:
    return ClassTree.from_leaf_paths(
        [
            "living-matters/animals/cat",
            "living-matters/animals/dog",
            "living-matters/plants/tree",
            "non-living-matters/rock",
        ]
    )


def test_tree_builds_expected_hierarchy() -> None:
    tree = _tree()
    assert tree.class_names == ("cat", "dog", "tree", "rock")
    assert tree.num_nodes == 9
    assert max(tree.depth) == 3
    cat_node = tree.leaf_indices[0]
    assert tuple(tree.ancestor_nodes(0)) == (
        tree.nodes[cat_node].parent,
        tree.nodes[tree.nodes[cat_node].parent].parent,
        tree.root_index,
    )
    assert tree.subtree_leaf_indices(tree.root_index) == (0, 1, 2, 3)
    assert tree.sibling_pairs() == ((1, 7), (2, 5), (3, 4))
    assert tree.leaf_sibling_pairs() == ((3, 4),)


def test_tree_weights_follow_subtree_counts() -> None:
    tree = _tree()
    weights = tree.node_weights_from_labels([0, 0, 1, 2, 3, 3, 3])
    # root=7, living=4, animals=3, cat=2, dog=1, plants=1, tree=1, non-living=3, rock=3
    assert weights[0] == 7
    assert weights[1] == 4
    assert weights[2] == 3
    assert weights[3] == 2
    assert weights[4] == 1
    assert weights[8] == 3


def test_tree_round_trip_metadata() -> None:
    tree = _tree()
    restored = ClassTree.from_dict(tree.to_dict())
    assert restored.to_dict() == tree.to_dict()
    assert restored.class_names == tree.class_names
