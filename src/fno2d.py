"""Small dependency-free 2D Fourier neural operator, for the steady Darcy control.

Why a separate module
---------------------
`src/fno.py` is one of the files hashed by `source_hash()` in every existing run identity,
so adding a class to it would change every fingerprint and break code-identity comparison
with the finished families. A new module leaves those hashes untouched.

The interface deliberately mirrors `FNO1d`: `forward(u0, coordinates)` maps a flattened
field of shape `(B, H*W)` to another flattened field on the same grid.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes, modes,
                                                       dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        m = min(self.modes, x_ft.shape[-2], x_ft.shape[-1])
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :m, :m] = torch.einsum("bixy,ioxy->boxy", x_ft[..., :m, :m],
                                           self.weight[..., :m, :m])
        return torch.fft.irfft2(out_ft, s=x.shape[-2:], dim=(-2, -1))


class FNO2d(nn.Module):
    """Maps a 2D coefficient field to a 2D solution field on the same grid."""

    def __init__(self, grid_shape=(64, 64), width: int = 32, modes: int = 12, layers: int = 4):
        super().__init__()
        if len(grid_shape) != 2 or any(int(s) <= 0 for s in grid_shape):
            raise ValueError(f"bad grid shape {grid_shape}")
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.lift = nn.Conv2d(3, width, 1)  # value plus two normalized coordinates
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                     nn.Conv2d(width, 1, 1))

    def forward(self, u0: torch.Tensor, coordinates: torch.Tensor | None = None) -> torch.Tensor:
        h, w = self.grid_shape
        if u0.ndim != 2 or u0.shape[-1] != h * w:
            raise ValueError(f"expected (batch, {h * w}), got {tuple(u0.shape)}")
        b = u0.shape[0]
        if coordinates is None:
            ys = torch.linspace(0, 1, h, device=u0.device, dtype=u0.dtype)
            xs = torch.linspace(0, 1, w, device=u0.device, dtype=u0.dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)
        else:
            coords = torch.as_tensor(coordinates, device=u0.device, dtype=u0.dtype)
            if coords.shape != (h * w, 2):
                raise ValueError(f"expected coordinates ({h * w}, 2), got {tuple(coords.shape)}")
        grid = coords.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        v = u0.reshape(b, h, w).unsqueeze(1)
        z = self.lift(torch.cat([v, grid], dim=1))
        for spec, local in zip(self.spectral, self.local):
            z = torch.nn.functional.gelu(spec(z) + local(z))
        return self.project(z).squeeze(1).reshape(b, h * w)


def square_grid_size(n_points: int) -> tuple[int, int]:
    """Infer a square grid from a flattened point count; used only to validate caches."""
    side = int(round(math.sqrt(n_points)))
    if side * side != n_points:
        raise ValueError(f"{n_points} is not a square grid")
    return side, side
