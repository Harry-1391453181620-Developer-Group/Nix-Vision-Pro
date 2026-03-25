"""Neural network primitives: layers, activations, losses, optimizers. NumPy only."""

from nn.activations import gelu, gelu_backward, relu, relu_backward, softmax, softplus, softplus_backward
from nn.layers import (
    BatchNorm2D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePool2D,
    LayerNorm,
    MaxPool2D,
    SqueezeExcitation,
)
from nn.losses import cross_entropy_loss, cross_entropy_loss_backward
from nn.mamba import MambaBlock
from nn.optimizers import AdamW, SGD

__all__ = [
    "relu",
    "relu_backward",
    "gelu",
    "gelu_backward",
    "softmax",
    "softplus",
    "softplus_backward",
    "Conv2D",
    "BatchNorm2D",
    "DepthwiseConv2D",
    "Dense",
    "Dropout",
    "GlobalAveragePool2D",
    "LayerNorm",
    "MambaBlock",
    "MaxPool2D",
    "SqueezeExcitation",
    "cross_entropy_loss",
    "cross_entropy_loss_backward",
    "AdamW",
    "SGD",
]
