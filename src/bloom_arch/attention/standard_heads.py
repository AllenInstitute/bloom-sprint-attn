"""Standard attention heads.

Q, K, and V are all projected from the full feature encoding
f_i = [n_i; p_i; m_i] where p_i = [beta * u_g; r_phi_i].
Uses F.scaled_dot_product_attention for GPU-optimized dispatch.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StandardHeads(nn.Module):
    """Multi-head self-attention on full feature encodings.

    Args:
        d_in: Dimension of full feature vector [n; p; m].
            Typically d_seq + d_u + 2*K + d_val.
        d_v: Per-head dimension for Q, K, V projections.
        H: Number of standard heads.
    """

    def __init__(self, d_in: int, d_v: int, H: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_v = d_v
        self.H = H
        self.W_Q = nn.Linear(d_in, H * d_v, bias=False)
        self.W_K = nn.Linear(d_in, H * d_v, bias=False)
        self.W_V = nn.Linear(d_in, H * d_v, bias=False)

    def forward(
        self,
        f: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run standard multi-head self-attention.

        Args:
            f: Tensor ``[B, L, d_in]`` -- full feature encoding [n_i; p_i; m_i].
            attn_mask: Optional ``[B, 1, 1, L]`` boolean mask. True = attend.

        Returns:
            Tensor ``[B, L, H * d_v]`` -- concatenated per-head outputs.
        """
        B, L, _ = f.shape
        H, d_v = self.H, self.d_v

        q = self.W_Q(f).view(B, L, H, d_v).transpose(1, 2)
        k = self.W_K(f).view(B, L, H, d_v).transpose(1, 2)
        v = self.W_V(f).view(B, L, H, d_v).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return out.transpose(1, 2).reshape(B, L, H * d_v)
