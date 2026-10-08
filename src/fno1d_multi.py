"""1D FNO for a multi-channel input field, used for the CFD system control.

Why a separate module
---------------------
`FNO1d` lifts a single value channel plus a coordinate channel and requires the input and the
coordinate to have the same length, so it cannot take the three-channel initial state of the
CFD system. `src/fno.py` is one of the files hashed by `source_hash()` in every existing run
identity, so it is not edited; this module is new and leaves those hashes untouched.

Interface mirrors `FNO1d`: `forward(u0, coordinates)` maps a flattened field of shape
`(B, C*N)` to a single output field of shape `(B, N)`.
"""
from __future__ import annotations

import torch
from torch import nn

from .fno import SpectralConv1d


class FNO1dMulti(nn.Module):
    """Maps a C-channel initial 1D state to one component of the later state."""

    def __init__(self, grid_size: int = 256, channels: int = 3, width: int = 32,
                 modes: int = 16, layers: int = 4):
        super().__init__()
        if grid_size <= 0 or channels <= 0:
            raise ValueError(f"bad grid_size {grid_size} or channels {channels}")
        self.grid_size = int(grid_size)
        self.channels = int(channels)
        self.lift = nn.Conv1d(channels + 1, width, 1)   # C value channels plus coordinate
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(),
                                     nn.Conv1d(width, 1, 1))

    def forward(self, u0: torch.Tensor, coordinates: torch.Tensor | None = None) -> torch.Tensor:
        n, c = self.grid_size, self.channels
        if u0.ndim != 2 or u0.shape[-1] != c * n:
            raise ValueError(f"expected (batch, {c * n}), got {tuple(u0.shape)}")
        b = u0.shape[0]
        if coordinates is None:
            x = torch.linspace(0, 1, n, device=u0.device, dtype=u0.dtype)
        else:
            x = torch.as_tensor(coordinates, device=u0.device, dtype=u0.dtype).flatten()
        if x.numel() != n:
            raise ValueError(f"expected coordinate length {n}, got {x.numel()}")
        grid = x.expand(b, 1, -1)
        v = u0.reshape(b, c, n)
        z = self.lift(torch.cat([v, grid], dim=1))
        for spec, local in zip(self.spectral, self.local):
            z = torch.nn.functional.gelu(spec(z) + local(z))
        return self.project(z).squeeze(1)
