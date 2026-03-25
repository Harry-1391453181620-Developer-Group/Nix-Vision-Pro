"""
Active model facade.

Keeps the original import path while allowing the project to prefer the PyTorch
backend when it is available. The legacy NumPy model remains available for
comparison and backward compatibility.
"""

from backends.numpy.model import CNN as NumpyCNN

try:
    from backends.torch.model import TorchCNN
except Exception:
    TorchCNN = None

CNN = TorchCNN or NumpyCNN

__all__ = ["CNN", "NumpyCNN", "TorchCNN"]
