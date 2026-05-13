"""Typed multi-head attention block (spec §5, §12).

Runs ``H_content`` content heads and ``H_spatial`` spatial heads in parallel,
concatenates their per-head outputs along the head dimension, and projects
back to ``d_s``.

This is the minimum surface needed to exercise the spec's permutation-
equivariance and same-chromosome-preference tests end-to-end. It deliberately
omits FFN, LayerNorm, and residual connections — those would obscure whether
the positional encoding behaves correctly at initialization. A later version
can wrap this block with the usual transformer machinery.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from bloom_arch.attention.content import ContentAttention
from bloom_arch.attention.spatial import SpatialAttention


class TypedMultiHeadAttention(nn.Module):
    """Parallel content + spatial heads with a single output projection.

    Parameters
    ----------
    d_s : int
        Sequence embedding dimension (also the block's input/output width).
    d_v : int
        Per-head value dimension (shared between content and spatial heads).
    H_content, H_spatial : int
        Number of content and spatial heads respectively.
    K : int
        Number of Fourier wavelengths.
    lambdas : Tensor [K]
        Wavelengths from the positional encoder. Passed to ``SpatialAttention``
        for log-Gaussian initialization of ``w_h``.
    target_scales : sequence of float, length H_spatial, optional
        Initial center wavelengths per spatial head.
    """

    def __init__(
        self,
        d_s: int,
        d_v: int,
        H_content: int,
        H_spatial: int,
        K: int,
        lambdas: torch.Tensor,
        target_scales: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if H_content < 0 or H_spatial < 0 or (H_content + H_spatial) == 0:
            raise ValueError("must have at least one head total")
        self.d_s = d_s
        self.d_v = d_v
        self.H_content = H_content
        self.H_spatial = H_spatial

        self.content = ContentAttention(d_s=d_s, d_v=d_v, H=H_content) if H_content > 0 else None
        self.spatial = (
            SpatialAttention(
                d_s=d_s, d_v=d_v, H=H_spatial, K=K, lambdas=lambdas, target_scales=target_scales
            )
            if H_spatial > 0
            else None
        )
        H_total = H_content + H_spatial
        self.W_O = nn.Linear(H_total * d_v, d_s, bias=False)

    def forward(
        self,
        s: torch.Tensor,
        u: torch.Tensor | None = None,
        r_phi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all heads.

        Parameters
        ----------
        s : Tensor [B, N, d_s]
        u, r_phi : positional encoder outputs (required iff H_spatial > 0)

        Returns
        -------
        out : Tensor [B, N, d_s]
        """
        outs = []
        if self.content is not None:
            outs.append(self.content(s))                # [B, N, H_c, d_v]
        if self.spatial is not None:
            if u is None or r_phi is None:
                raise ValueError("spatial heads require u and r_phi")
            outs.append(self.spatial(u, r_phi, s))       # [B, N, H_s, d_v]
        cat = torch.cat(outs, dim=-2)                    # [B, N, H_total, d_v]
        B, N, H_total, d_v = cat.shape
        flat = cat.reshape(B, N, H_total * d_v)
        return self.W_O(flat)
