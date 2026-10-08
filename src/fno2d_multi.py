"""2D FNO for a multi-channel input field, used for the reaction-diffusion system control.

Why a separate module
---------------------
`FNO2d` lifts a single value channel plus two coordinate channels, so it cannot take the
two-component initial state of the reaction-diffusion system. `src/fno2d.py` is one of the
files hashed by `source_hash()` in the Darcy run identities, so it is not edited; this module
is new and leaves those hashes untouched.

Interface mirrors `FNO2d`: `forward(u0, coordinates)` maps a flattened field of shape
`(B, C*H*W)` to a single output field of shape `(B, H*W)`.
"""
from __future__ import annotations

import torch
from torch import nn

from .fno2d import SpectralConv2d


class FNO2dMulti(nn.Module):
    """Maps a C-channel initial 2D state to one component of the later state."""

    def __init__(self, grid_shape=(32, 32), channels: int = 2, width: int = 32,
                 modes: int = 12, layers: int = 4):
        super().__init__()
        h, w = int(grid_shape[0]), int(grid_shape[1])
        if h <= 0 or w <= 0 or channels <= 0:
            raise ValueError(f"bad grid_shape {grid_shape} or channels {channels}")
        self.grid_shape = (h, w)
        self.channels = int(channels)
        # C value channels plus two coordinates
        self.lift = nn.Conv2d(channels + 2, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                     nn.Conv2d(width, 1, 1))

    def forward(self, u0: torch.Tensor, coordinates: torch.Tensor | None = None) -> torch.Tensor:
        h, w = self.grid_shape
        c = self.channels
        n = h * w
        if u0.ndim != 2 or u0.shape[-1] != c * n:
            raise ValueError(f"expected (batch, {c * n}), got {tuple(u0.shape)}")
        b = u0.shape[0]
        if coordinates is None:
            ys = torch.linspace(0, 1, h, device=u0.device, dtype=u0.dtype)
            xs = torch.linspace(0, 1, w, device=u0.device, dtype=u0.dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)
        else:
            coords = torch.as_tensor(coordinates, device=u0.device, dtype=u0.dtype)
            if coords.shape != (n, 2):
                raise ValueError(f"expected coordinates ({n}, 2), got {tuple(coords.shape)}")
        grid = coords.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        v = u0.reshape(b, c, h, w)
        z = self.lift(torch.cat([v, grid], dim=1))
        for spec, local in zip(self.spectral, self.local):
            z = torch.nn.functional.gelu(spec(z) + local(z))
        return self.project(z).squeeze(1).reshape(b, n)
