"""Multiscale Fourier coordinate encoding (spec §6).

For a coordinate ``x`` along a chromosome (in base pairs), this module returns

    φ(x) = (1/√K) · [ sin(2π x / λ_1), cos(2π x / λ_1),
                       sin(2π x / λ_2), cos(2π x / λ_2),
                       ...,
                       sin(2π x / λ_K), cos(2π x / λ_K) ]

with wavelengths log-spaced between ``lambda_min`` and ``lambda_max``. The
output is interleaved (sin, cos) per wavelength so the head-specific
wavelength weights ``w_h`` can be tiled by `repeat_interleave(2)`.

Properties (verified in tests/test_pos_encoder_v2.py):

* Translation invariance of the dot product: ``φ(x_i)·φ(x_j) = (1/K) Σ_k cos(2π(x_i-x_j)/λ_k)``
  depends only on |x_i − x_j|, not on absolute coordinate (spec §4.1).
* No learnable parameters; the log-spaced wavelengths are stored as a buffer.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierCoordEncoder(nn.Module):
    """Log-spaced multiscale Fourier feature map for genomic coordinates.

    Produces interleaved (sin, cos) pairs at log-spaced wavelengths, scaled
    by 1/sqrt(K) so that the dot product phi(x_i).phi(x_j) lies in [-1, 1]
    and equals 1 when x_i = x_j.

    Args:
        K: Number of wavelengths. Output dim is ``2*K``.
        lambda_min: Finest wavelength in base pairs.
        lambda_max: Coarsest wavelength in base pairs.
    """

    def __init__(self, K: int = 112, lambda_min: float = 10.0, lambda_max: float = 1.0e8) -> None:
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if lambda_min <= 0 or lambda_max <= 0:
            raise ValueError("wavelengths must be positive")
        if K >= 2 and lambda_max <= lambda_min:
            raise ValueError("lambda_max must exceed lambda_min when K >= 2")

        self.K = K
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)

        if K == 1:
            lambdas = torch.tensor([lambda_min], dtype=torch.float32)
        else:
            exponents = torch.arange(K, dtype=torch.float32) / (K - 1)
            ratio = lambda_max / lambda_min
            lambdas = lambda_min * (ratio**exponents)

        # Angular frequencies ω_k = 2π / λ_k. Stored as buffer so .to(device) works.
        omegas = 2.0 * math.pi / lambdas
        self.register_buffer("omegas", omegas)  # [K]
        self.register_buffer("lambdas", lambdas)  # [K], handy for plotting / tests

        self._scale = 1.0 / math.sqrt(K)

    @property
    def d_phi(self) -> int:
        return 2 * self.K

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode genomic coordinates into Fourier features.

        Args:
            coords: Any-shape float tensor of genomic coordinates in bp.

        Returns:
            Tensor ``[..., 2*K]`` -- interleaved (sin, cos) per wavelength,
            scaled by 1/sqrt(K).
        """
        # Promote omegas to the coords device/dtype.
        omegas = self.omegas.to(device=coords.device, dtype=coords.dtype)
        # angles[..., k] = ω_k * x
        angles = coords.unsqueeze(-1) * omegas  # broadcast → [..., K]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        # Interleave (sin_k, cos_k) along the last dim:
        # stack to [..., K, 2] then flatten the last two dims.
        out = torch.stack((sin, cos), dim=-1).flatten(-2)  # [..., 2K]
        return out * self._scale
