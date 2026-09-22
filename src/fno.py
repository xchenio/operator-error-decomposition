"""Small dependency-free 1D Fourier neural operator used for cross-architecture controls."""
from __future__ import annotations

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes,
                                                        dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, X)
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :n_modes] = torch.einsum("bim,iom->bom", x_ft[..., :n_modes],
                                              self.weight[..., :n_modes])
        return torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)


class FNO1d(nn.Module):
    """Maps a 1D initial condition to one later solution field on the same grid."""
    def __init__(self, width: int = 32, modes: int = 16, layers: int = 4):
        super().__init__()
        self.lift = nn.Conv1d(2, width, 1)  # value plus normalized coordinate
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, u0: torch.Tensor, coordinates=None) -> torch.Tensor:
        if u0.ndim != 2:
            raise ValueError(f"expected (batch, grid), got {tuple(u0.shape)}")
        x = (torch.linspace(0, 1, u0.shape[-1], device=u0.device, dtype=u0.dtype)
             if coordinates is None else torch.as_tensor(coordinates, device=u0.device, dtype=u0.dtype).flatten())
        if x.numel() != u0.shape[-1]:
            raise ValueError('coordinate length differs from field length')
        x = x.expand(u0.shape[0], 1, -1)
        h = self.lift(torch.cat([u0[:, None, :], x], dim=1))
        for spec, local in zip(self.spectral, self.local):
            h = torch.nn.functional.gelu(spec(h) + local(h))
        return self.project(h).squeeze(1)
