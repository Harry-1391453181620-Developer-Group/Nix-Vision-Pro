"""Optimizers. NumPy only."""

from typing import List, Tuple

import numpy as np


class SGD:
    """SGD with optional momentum and weight decay."""

    def __init__(
        self,
        parameters: List[Tuple[np.ndarray, np.ndarray]],
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        """
        Args:
            parameters: List of (param, grad) tuples. grad may be None if not computed yet.
            lr: Learning rate.
            momentum: Momentum factor in [0, 1).
            nesterov: Whether to use Nesterov momentum.
            weight_decay: L2 regularization factor.
        """
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must satisfy 0 <= momentum < 1")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self._velocity = [np.zeros_like(param) for param, _ in self.parameters]

    def step(self) -> None:
        """Update each parameter using SGD (+ momentum/weight decay when enabled)."""
        for idx, (param, grad) in enumerate(self.parameters):
            if grad is not None:
                update = grad
                if self.weight_decay > 0.0:
                    update = update + self.weight_decay * param
                if self.momentum > 0.0:
                    v = self._velocity[idx]
                    v *= self.momentum
                    v += update
                    if self.nesterov:
                        update = update + self.momentum * v
                    else:
                        update = v
                param -= self.lr * update

    def zero_grad(self, parameters: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """No-op for SGD (gradients are overwritten each backward). Kept for API compatibility."""
        pass


class AdamW:
    """AdamW optimizer (decoupled weight decay)."""

    def __init__(
        self,
        parameters: List[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("betas must satisfy 0 <= beta < 1")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(param) for param, _ in self.parameters]
        self._v = [np.zeros_like(param) for param, _ in self.parameters]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        bias_correction1 = 1.0 - self.beta1**self._t
        bias_correction2 = 1.0 - self.beta2**self._t
        for idx, (param, grad) in enumerate(self.parameters):
            if grad is None:
                continue
            if self.weight_decay > 0.0:
                param *= 1.0 - self.lr * self.weight_decay
            m = self._m[idx]
            v = self._v[idx]
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * (grad * grad)
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, parameters: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """No-op for AdamW (gradients are overwritten each backward)."""
        pass
