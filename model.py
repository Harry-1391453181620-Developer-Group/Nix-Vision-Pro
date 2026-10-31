"""Active model facade for the CNN-only MAOL implementation."""

from backends.torch.model import CNN, TorchCNN

__all__ = ["CNN", "TorchCNN"]
