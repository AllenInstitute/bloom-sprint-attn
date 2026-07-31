"""Sequence attention heads.

Q and K are projected from the sequence encoding n_i.
V is projected from the content encoding s_i = [n_i; m_i].
Scaling: 1/sqrt(d_v) where d_v is the per-head query/key dimension
(standard scaled-dot-product attention).
Uses F.scaled_dot_product_attention for GPU-optimized dispatch.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SequenceHeads(nn.Module):
    """Multi-head attention on sequence encodings.

    Args:
        d_seq: Dimension of sequence encoding n (input for Q, K).
        d_s: Dimension of content encoding s = [n; m] (input for V).
        d_v: Per-head dimension for Q, K, and V projections.
        H: Number of sequence heads.
    """

    def __init__(self, d_seq: int, d_s: int, d_v: int, H: int) -> None:
        super().__init__()
        self.d_seq = d_seq
        self.d_s = d_s
        self.d_v = d_v
        self.H = H
        self._scale = 1.0 / math.sqrt(d_v)
        self.W_Q = nn.Linear(d_seq, H * d_v, bias=False)
        self.W_K = nn.Linear(d_seq, H * d_v, bias=False)
        self.W_V = nn.Linear(d_s, H * d_v, bias=False)

    def forward(
        self,
        n: torch.Tensor,
        s: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run sequence attention.

        Args:
            n: Tensor ``[B, L, d_seq]`` -- sequence encoding.
            s: Tensor ``[B, L, d_s]`` -- content encoding (typically [n; m]).
            attn_mask: Optional ``[B, 1, 1, L]`` boolean mask. True = attend.

        Returns:
            Tensor ``[B, L, H * d_v]`` -- concatenated per-head outputs.
        """
        B, L, _ = n.shape
        H, d_v = self.H, self.d_v

        q = self.W_Q(n).view(B, L, H, d_v).transpose(1, 2)
        k = self.W_K(n).view(B, L, H, d_v).transpose(1, 2)
        v = self.W_V(s).view(B, L, H, d_v).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self._scale)
        return out.transpose(1, 2).reshape(B, L, H * d_v)
