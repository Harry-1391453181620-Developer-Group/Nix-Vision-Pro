"""
Mamba block (selective state-space model) in NumPy only.
O(L * E) recurrence; no O(L^2) attention. Used in hybrid CNN+Mamba.
"""

import numpy as np

from nn.activations import softplus, softplus_backward


def _log_spaced_A(state_dim: int) -> np.ndarray:
    """Log-spaced negative values for SSM A (HiPPO-style). Returns shape (state_dim,)."""
    return -np.exp(np.linspace(0, np.log(state_dim + 1), state_dim + 1, dtype=np.float64)[1:])


class MambaBlock:
    """
    Selective SSM block: input (N, L, D) -> output (N, L, D).
    State recurrence is O(N*L*E); no quadratic attention.
    """

    def __init__(self, in_dim: int, state_dim: int, seed: int | None = None):
        if in_dim < 1 or state_dim < 1:
            raise ValueError("in_dim and state_dim must be >= 1")
        if seed is not None:
            np.random.seed(seed)
        self.in_dim = in_dim
        self.state_dim = state_dim
        scale = 1.0 / np.sqrt(in_dim)
        # Input projections (all from x): delta, B, C, u
        self.W_delta = np.random.randn(in_dim, 1).astype(np.float64) * scale
        self.W_B = np.random.randn(in_dim, state_dim).astype(np.float64) * scale
        self.W_C = np.random.randn(in_dim, state_dim).astype(np.float64) * scale
        self.W_u = np.random.randn(in_dim, state_dim).astype(np.float64) * scale
        self.W_res = np.eye(in_dim, dtype=np.float64) + np.random.randn(in_dim, in_dim).astype(np.float64) * (0.01 * scale)
        self.v_out = np.random.randn(in_dim).astype(np.float64) * (0.1 * scale)
        # Fixed A (diagonal), negative for stability
        self.A = _log_spaced_A(state_dim)
        self._x = None
        self._z_delta = None
        self._delta = None
        self._B = None
        self._C = None
        self._u = None
        self._x_res = None
        self._A_bar = None
        self._F = None
        self._B_bar = None
        self._h_all = None
        self._h_prev_all = None
        self._ssm_out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, L, D). Returns (N, L, D)."""
        self._x = x
        N, L, D = x.shape
        E = self.state_dim
        # Projections: O(N*L*D*E) total
        self._z_delta = x @ self.W_delta
        self._delta = softplus(self._z_delta)
        self._B = x @ self.W_B
        self._C = x @ self.W_C
        self._u = x @ self.W_u
        self._x_res = x @ self.W_res
        # Discretize: A_bar = exp(delta * A), B_bar = ((A_bar - 1)/A) * B
        delta_expand = np.maximum(self._delta, 1e-6)
        self._A_bar = np.exp(delta_expand * self.A)
        self._F = (self._A_bar - 1.0) / (self.A + 1e-12)
        self._B_bar = self._F * self._B
        # Recurrence: h_t = A_bar_t * h_{t-1} + B_bar_t * u_t; y_t = (C_t * h_t).sum(-1). O(N*L*E)
        h = np.zeros((N, E), dtype=np.float64)
        ssm_out = np.zeros((N, L), dtype=np.float64)
        h_all = np.zeros((N, L, E), dtype=np.float64)
        h_prev_all = np.zeros((N, L, E), dtype=np.float64)
        for t in range(L):
            h_prev_all[:, t, :] = h
            h = self._A_bar[:, t, :] * h + self._B_bar[:, t, :] * self._u[:, t, :]
            h_all[:, t, :] = h
            ssm_out[:, t] = np.sum(self._C[:, t, :] * h, axis=-1)
        self._h_all = h_all
        self._h_prev_all = h_prev_all
        self._ssm_out = ssm_out
        # Identity residual keeps a strong gradient path through the block.
        out = self._x_res + ssm_out[:, :, np.newaxis] * self.v_out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (N, L, D). Returns (N, L, D)."""
        N, L, D = dout.shape
        E = self.state_dim
        d_residual = dout.copy()
        d_ssm = np.sum(dout * self.v_out, axis=-1)
        # Backward through recurrence (reverse time): O(N*L*E)
        dh = np.zeros((N, E), dtype=np.float64)
        dA_bar = np.zeros_like(self._A_bar)
        dB_bar = np.zeros_like(self._B_bar)
        dC = np.zeros_like(self._C)
        du = np.zeros_like(self._u)
        for t in range(L - 1, -1, -1):
            dC[:, t, :] = d_ssm[:, t, np.newaxis] * self._h_all[:, t, :]
            dh += d_ssm[:, t, np.newaxis] * self._C[:, t, :]
            dB_bar[:, t, :] = dh * self._u[:, t, :]
            du[:, t, :] = dh * self._B_bar[:, t, :]
            dA_bar[:, t, :] = dh * self._h_prev_all[:, t, :]
            dh = dh * self._A_bar[:, t, :]
        # Backward through discretization: B_bar = F * B, F = (A_bar - 1)/A.
        dB = dB_bar * self._F
        dF = dB_bar * self._B
        dA_bar += dF / (self.A + 1e-12)
        d_delta = np.sum(dA_bar * self._A_bar * self.A, axis=-1, keepdims=True)
        dz_delta = softplus_backward(d_delta, self._z_delta)
        # Gradients to parameters
        self._dW_delta = np.tensordot(self._x, dz_delta, axes=([0, 1], [0, 1]))
        self._dW_B = np.tensordot(self._x, dB, axes=([0, 1], [0, 1]))
        self._dW_C = np.tensordot(self._x, dC, axes=([0, 1], [0, 1]))
        self._dW_u = np.tensordot(self._x, du, axes=([0, 1], [0, 1]))
        self._dW_res = np.tensordot(self._x, d_residual, axes=([0, 1], [0, 1]))
        self._dv_out = np.sum(dout * self._ssm_out[:, :, np.newaxis], axis=(0, 1))
        # Gradient to input
        dx = d_residual @ self.W_res.T
        dx += dz_delta @ self.W_delta.T
        dx += dB @ self.W_B.T
        dx += dC @ self.W_C.T
        dx += du @ self.W_u.T
        return dx

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (self.W_delta, getattr(self, "_dW_delta", None)),
            (self.W_B, getattr(self, "_dW_B", None)),
            (self.W_C, getattr(self, "_dW_C", None)),
            (self.W_u, getattr(self, "_dW_u", None)),
            (self.W_res, getattr(self, "_dW_res", None)),
            (self.v_out, getattr(self, "_dv_out", None)),
        ]
