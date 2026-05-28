"""BloomAttentionBlock — top-level multi-head attention with three head types.

Implements the full attention block from spec §2-§3:
- Sequence heads: Q, K from n_i; V from s_i = [n_i; m_i]
- Position heads: Q = K from positional encoding; V from s_i
- Standard heads: Q, K, V from full feature [n_i; p_i; m_i]

All head types use F.scaled_dot_product_attention for flash/efficient dispatch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from bloom_arch.attention.position_heads import PositionHeads
from bloom_arch.attention.sequence_heads import SequenceHeads
from bloom_arch.attention.standard_heads import StandardHeads
from bloom_arch.posenc.encoder import PosEncoder


class BloomAttentionBlock(nn.Module):
    """Multi-head attention block with sequence, position, and standard heads.

    Args:
        d_seq: Dimension of sequence encoding n.
        d_val: Dimension of value encoding m.
        d_u: Chromosome baseline dimension for positional encoding.
        K: Number of Fourier wavelengths (d_phi = 2K).
        d_v: Per-head value/projection dimension (shared across head types).
        n_sequence_heads: Number of sequence attention heads.
        n_position_heads: Number of position attention heads.
        n_standard_heads: Number of standard attention heads.
        d_out: Output dimension.
        max_G: Maximum number of genome/chromosome IDs for the codebook.
        lambda_min: Finest Fourier wavelength in bp.
        lambda_max: Coarsest Fourier wavelength in bp.
        target_scales: Per-position-head center wavelengths for log-Gaussian
            init. If None, log-spaced across the usable wavelength range.
        rho: Log-Gaussian bandwidth for position head initialization.
        beta_init: Initial beta for position heads.
        tau_init: Initial tau for position heads.
        learn_tau: If True, position head tau is learnable.
        learn_w: If True, position head wavelength weights are learnable.
        pad_v_for_flash: If True, pad V in position heads for flash dispatch.
        seed: Codebook seed for deterministic generation.
    """

    def __init__(
        self,
        d_seq: int,
        d_val: int,
        d_u: int,
        K: int,
        d_v: int,
        n_sequence_heads: int,
        n_position_heads: int,
        n_standard_heads: int,
        d_out: int,
        max_G: int,
        lambda_min: float = 10.0,
        lambda_max: float = 1.0e8,
        target_scales: Sequence[float] | None = None,
        rho: float = math.log(10.0) / 2.0,
        beta_init: float = 1.0,
        tau_init: float = 1.0,
        learn_tau: bool = False,
        learn_w: bool = True,
        pad_v_for_flash: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if n_sequence_heads + n_position_heads + n_standard_heads == 0:
            raise ValueError("Must have at least one head total")

        self.d_seq = d_seq
        self.d_val = d_val
        self.d_u = d_u
        self.K = K
        self.d_v = d_v
        self.n_sequence_heads = n_sequence_heads
        self.n_position_heads = n_position_heads
        self.n_standard_heads = n_standard_heads

        d_s = d_seq + d_val

        self.pos_encoder = PosEncoder(
            max_G=max_G,
            d_u=d_u,
            K=K,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            seed=seed,
        )

        self.seq_heads = (
            SequenceHeads(d_seq=d_seq, d_s=d_s, d_v=d_v, H=n_sequence_heads)
            if n_sequence_heads > 0
            else None
        )

        self.pos_heads = (
            PositionHeads(
                d_s=d_s,
                d_u=d_u,
                K=K,
                d_v=d_v,
                H=n_position_heads,
                lambdas=self.pos_encoder.fourier.lambdas,
                target_scales=target_scales,
                rho=rho,
                beta_init=beta_init,
                tau_init=tau_init,
                learn_tau=learn_tau,
                learn_w=learn_w,
                pad_v_for_flash=pad_v_for_flash,
            )
            if n_position_heads > 0
            else None
        )

        d_in_std = d_seq + d_u + 2 * K + d_val
        self.std_heads = (
            StandardHeads(d_in=d_in_std, d_v=d_v, H=n_standard_heads)
            if n_standard_heads > 0
            else None
        )

        H_total = n_sequence_heads + n_position_heads + n_standard_heads
        self.W_O = nn.Linear(H_total * d_v, d_out, bias=False)

    def forward(
        self,
        n: torch.Tensor,
        x: torch.Tensor,
        g: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full attention block.

        Args:
            n: Tensor ``[B, L, d_seq]`` -- sequence encoding (from DNA LM).
            x: Tensor ``[B, L]`` -- genomic coordinates in base pairs (float).
            g: Tensor ``[B, L]`` -- integer genome/chromosome indices (long).
            m: Tensor ``[B, L, d_val]`` -- measured value encoding.

        Returns:
            Tensor ``[B, L, d_out]`` -- block output.
        """
        s = torch.cat([n, m], dim=-1)  # [B, L, d_s]

        pos = self.pos_encoder(x, g)
        u = pos["u"].to(dtype=n.dtype)  # [B, L, d_u]
        r_phi = pos["r_phi"].to(dtype=n.dtype)  # [B, L, 2K]

        outs = []

        if self.seq_heads is not None:
            outs.append(self.seq_heads(n, s))

        if self.pos_heads is not None:
            outs.append(self.pos_heads(u, r_phi, s))

        if self.std_heads is not None:
            p = torch.cat([u, r_phi], dim=-1)  # [B, L, d_u + 2K]
            f = torch.cat([n, p, m], dim=-1)  # [B, L, d_seq + d_u + 2K + d_val]
            outs.append(self.std_heads(f))

        cat = torch.cat(outs, dim=-1)  # [B, L, H_total * d_v]
        return self.W_O(cat)
