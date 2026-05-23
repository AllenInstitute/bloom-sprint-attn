"""Chromosome-local positional encoder (spec §3, §10).

Composes :class:`ChromCodeBook` and :class:`FourierCoordEncoder` and returns
the two channels of the positional encoding **without** the β weighting or the
head-specific wavelength weights. Those live in the spatial attention heads
(spec §8 says β should be a learned per-spatial-head scalar; spec §12.2 makes
the Fourier-bank weights per-head), so the encoder is parameter-free and a
single set of positional features is reused across all spatial heads:

    forward(coords, chrom_keys) → {'u': u_g, 'r_phi': r_g ⊙ φ(x)}

with shapes ``u: [B, N, d_u]`` and ``r_phi: [B, N, 2*K]``.

By keeping the encoder free of β and ``w_h``, we (a) avoid materializing one
copy of ``p_i`` per head per layer, (b) keep the encoder genuinely stateless
so tests that check ``no learned chromosome priors`` are easy to write, and
(c) match the spec's "p_i^(h) = [β_h u_g ; r_g ⊙ (w_h ⊙ φ(x_i))]" exactly
when the spatial head composes the parts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from bloom_arch.posenc.chrom_codes import ChromCodeBook, ChromKey
from bloom_arch.posenc.fourier import FourierCoordEncoder


class ChromLocalPosEncoder(nn.Module):
    """Compute (u_g, r_g ⊙ φ(x)) for a batch of features.

    Parameters
    ----------
    d_u : int
        Chromosome-baseline dimension (spec §10.1 default: 32).
    K : int
        Number of Fourier wavelengths (spec §10.1 default: 112; d_phi = 2K).
    lambda_min, lambda_max : float
        Wavelength range in base pairs (spec §10.2 default: 10 bp and 100 Mb).
    salt : str
        ChromCodeBook salt (spec §9).

    Notes
    -----
    Outputs are returned as a dict to keep the API explicit. Spatial attention
    consumes both keys; content attention ignores positional output entirely.
    The encoder has **zero learnable parameters** by design.
    """

    def __init__(
        self,
        d_u: int = 32,
        K: int = 112,
        lambda_min: float = 10.0,
        lambda_max: float = 1.0e8,
        salt: str = "posenc_v1",
    ) -> None:
        super().__init__()
        self.codes = ChromCodeBook(d_u=d_u, d_phi=2 * K, salt=salt)
        self.fourier = FourierCoordEncoder(K=K, lambda_min=lambda_min, lambda_max=lambda_max)

    @property
    def d_u(self) -> int:
        return self.codes.d_u

    @property
    def d_phi(self) -> int:
        return self.fourier.d_phi

    @property
    def K(self) -> int:
        return self.fourier.K

    def forward(
        self,
        coords: torch.Tensor,
        chrom_keys: Sequence[Sequence[ChromKey]],
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of features.

        Parameters
        ----------
        coords : Tensor of shape [B, N]
            Per-feature coordinates in base pairs.
        chrom_keys : nested list, length B × N
            ``chrom_keys[b][n] = (species, assembly, chrom)``.

        Returns
        -------
        dict with keys
            - ``"u"``     : [B, N, d_u]   — u_g, the same-chromosome baseline code
            - ``"r_phi"`` : [B, N, 2*K]   — r_g ⊙ φ(x), sign-gated Fourier encoding
        """
        if coords.dim() != 2:
            raise ValueError(f"coords must be 2-D [B, N], got shape {tuple(coords.shape)}")
        B, N = coords.shape
        if len(chrom_keys) != B or any(len(row) != N for row in chrom_keys):
            raise ValueError("chrom_keys shape must match coords shape")

        u, r = self.codes(chrom_keys, device=coords.device, dtype=coords.dtype)
        phi = self.fourier(coords)  # [B, N, 2K]
        r_phi = r * phi
        return {"u": u, "r_phi": r_phi}
