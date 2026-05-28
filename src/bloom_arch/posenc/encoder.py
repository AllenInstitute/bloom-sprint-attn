"""Positional encoder.

Composes :class:`GenomeCodebook` and :class:`FourierCoordEncoder` to produce
the two channels of the positional encoding from integer genome indices and
genomic coordinates:

    forward(x, g) -> {"u": u_g, "r_phi": r_g . phi(x)}

The encoder has zero learnable parameters. Per-head weighting (beta, w_h)
lives in the :class:`PositionHeads` attention module.
"""

from __future__ import annotations

import torch
from torch import nn

from bloom_arch.posenc.fourier import FourierCoordEncoder
from bloom_arch.posenc.genome_codebook import GenomeCodebook


class PosEncoder(nn.Module):
    """GPU-first positional encoder using integer genome indices.

    Composes :class:`GenomeCodebook` and :class:`FourierCoordEncoder`.
    Zero learnable parameters.

    Args:
        max_G: Maximum number of genome/chromosome IDs for the codebook.
        d_u: Chromosome baseline dimension.
        K: Number of Fourier wavelengths (output phi dim = 2K).
        lambda_min: Finest wavelength in base pairs.
        lambda_max: Coarsest wavelength in base pairs.
        seed: Codebook seed for deterministic generation.
    """

    def __init__(
        self,
        max_G: int,
        d_u: int = 32,
        K: int = 112,
        lambda_min: float = 10.0,
        lambda_max: float = 1.0e8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.codebook = GenomeCodebook(max_G=max_G, d_u=d_u, d_phi=2 * K, seed=seed)
        self.fourier = FourierCoordEncoder(K=K, lambda_min=lambda_min, lambda_max=lambda_max)

    @property
    def d_u(self) -> int:
        """Chromosome baseline vector dimension."""
        return self.codebook.d_u

    @property
    def d_phi(self) -> int:
        """Fourier encoding dimension (2K)."""
        return self.fourier.d_phi

    @property
    def K(self) -> int:
        """Number of Fourier wavelengths."""
        return self.fourier.K

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of features using integer genome indices.

        Args:
            x: Float tensor of shape ``[B, L]``. Per-feature genomic
                coordinates in base pairs.
            g: Long tensor of shape ``[B, L]``. Integer genome/chromosome
                indices into the codebook.

        Returns:
            Dict with keys:
                - ``"u"``: Tensor ``[B, L, d_u]`` -- chromosome baseline code.
                - ``"r_phi"``: Tensor ``[B, L, 2K]`` -- sign-gated Fourier
                  encoding (r_g . phi(x)).
        """
        u, r = self.codebook(g)
        phi = self.fourier(x)
        r_phi = r * phi
        return {"u": u, "r_phi": r_phi}
