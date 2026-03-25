"""
PyTorch Mamba block that mirrors the current NumPy selective SSM structure.

The math intentionally follows the legacy implementation so the project keeps the
same high-level architecture while moving the active backend to PyTorch.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _log_spaced_A(state_dim: int) -> torch.Tensor:
    if state_dim < 1:
        raise ValueError("state_dim must be >= 1")
    values = torch.linspace(0.0, math.log(state_dim + 1), state_dim + 1, dtype=torch.float32)[1:]
    return -torch.exp(values)


class TorchMambaBlock(nn.Module):
    """Selective SSM block: input (N, L, D) -> output (N, L, D)."""

    def __init__(self, in_dim: int, state_dim: int, seed: int | None = None):
        super().__init__()
        if in_dim < 1 or state_dim < 1:
            raise ValueError("in_dim and state_dim must be >= 1")
        if seed is not None:
            torch.manual_seed(seed)
        self.in_dim = int(in_dim)
        self.state_dim = int(state_dim)
        scale = 1.0 / math.sqrt(in_dim)

        self.delta_proj = nn.Linear(in_dim, 1, bias=False)
        self.B_proj = nn.Linear(in_dim, state_dim, bias=False)
        self.C_proj = nn.Linear(in_dim, state_dim, bias=False)
        self.u_proj = nn.Linear(in_dim, state_dim, bias=False)
        self.res_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.v_out = nn.Parameter(torch.randn(in_dim, dtype=torch.float32) * (0.1 * scale))

        self.register_buffer("A", _log_spaced_A(state_dim))
        self._reset_parameters(scale)

    def _reset_parameters(self, scale: float) -> None:
        for layer in (self.delta_proj, self.B_proj, self.C_proj, self.u_proj):
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
        eye = torch.eye(self.in_dim, dtype=torch.float32)
        noise = torch.randn(self.in_dim, self.in_dim, dtype=torch.float32) * (0.01 * scale)
        with torch.no_grad():
            self.res_proj.weight.copy_(eye + noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TorchMambaBlock expects (N, L, D), got {tuple(x.shape)}")
        z_delta = self.delta_proj(x)
        delta = F.softplus(z_delta)
        B = self.B_proj(x)
        C = self.C_proj(x)
        u = self.u_proj(x)
        x_res = self.res_proj(x)

        # Keep the discretization in the current tensor dtype/device and avoid
        # slice writes into an autograd-tracked output tensor. The previous
        # version was mathematically correct on CPU, but this formulation is
        # more stable and avoids a CUDA failure mode around recurrent
        # copy-slice updates.
        A = self.A.to(device=x.device, dtype=x.dtype).view(1, 1, self.state_dim)
        delta_expand = delta.clamp_min(1e-6)
        discretized = delta_expand * A
        A_bar = torch.exp(discretized)
        F_term = torch.expm1(discretized) / (A + 1e-12)
        B_bar = F_term * B

        batch_size, seq_len, _ = x.shape
        hidden = torch.zeros(batch_size, self.state_dim, dtype=x.dtype, device=x.device)
        ssm_terms: list[torch.Tensor] = []
        for index in range(seq_len):
            hidden = A_bar[:, index, :] * hidden + B_bar[:, index, :] * u[:, index, :]
            ssm_terms.append(torch.sum(C[:, index, :] * hidden, dim=-1))
        ssm_out = torch.stack(ssm_terms, dim=1)
        return x_res + ssm_out.unsqueeze(-1) * self.v_out
