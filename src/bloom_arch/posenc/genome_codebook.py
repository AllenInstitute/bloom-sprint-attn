"""Integer-indexed deterministic genome codebook (spec §5-§6).

Pre-generates chromosome baseline vectors ``u_g`` and sign vectors ``r_g``
at init time using a deterministic seed-per-index scheme. All codes are
stored as non-learnable buffers so no gradients flow through the codebook.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GenomeCodebook(nn.Module):
    """Integer-indexed deterministic codebook for genome/chromosome codes.

    Pre-generates unit-norm baseline vectors ``u_g`` and {-1, +1} sign
    vectors ``r_g`` at construction. Forward is a simple buffer index.

    Args:
        max_G: Maximum number of genome/chromosome IDs.
        d_u: Dimension of u_g vectors (unit-norm baselines).
        d_phi: Dimension of r_g sign vectors. Must equal 2*K.
        seed: Base seed for deterministic generation. Each index g uses
            ``seed + g`` so extending max_G preserves existing codes.
    """

    def __init__(self, max_G: int, d_u: int, d_phi: int, seed: int = 42) -> None:
        super().__init__()
        self.max_G = max_G
        self.d_u = d_u
        self.d_phi = d_phi
        self.seed = seed

        u_buf = torch.empty(max_G, d_u)
        r_buf = torch.empty(max_G, d_phi)
        gen = torch.Generator()

        for g in range(max_G):
            gen.manual_seed(seed + g)
            u_vec = torch.randn(d_u, generator=gen)
            u_vec = u_vec / u_vec.norm()
            u_buf[g] = u_vec
            r_buf[g] = torch.bernoulli(torch.full((d_phi,), 0.5), generator=gen) * 2 - 1

        self.register_buffer("u", u_buf)
        self.register_buffer("r", r_buf)

    def forward(self, g: Tensor) -> tuple[Tensor, Tensor]:
        """Look up codes for a batch of genome indices.

        Args:
            g: Long tensor of shape ``[B, L]`` with integer genome/chromosome
                indices in ``[0, max_G)``.

        Returns:
            Tuple of (u, r) where:
                - u: Tensor ``[B, L, d_u]`` -- unit-norm baseline vectors.
                - r: Tensor ``[B, L, d_phi]`` -- sign vectors ({-1, +1}).
        """
        return self.u[g], self.r[g]
