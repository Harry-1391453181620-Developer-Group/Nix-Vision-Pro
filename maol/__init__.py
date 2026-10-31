"""MAOL: Multi-Attractor Optimized Learning."""

from maol.head import PrototypeHead
from maol.losses import MAOLLossTerms, compute_maol_losses
from maol.tree import ClassTree, TreeNode

__all__ = [
    "ClassTree",
    "TreeNode",
    "PrototypeHead",
    "MAOLLossTerms",
    "compute_maol_losses",
]
