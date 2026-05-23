"""Content attention head (spec §5.1).

Standard multi-head self-attention on sequence embeddings — included for
parity with spatial attention and so the :class:`TypedMultiHeadAttention`
block can run both head families in one layer.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ContentAttention(nn.Module):
    """Multi-head self-attention on the sequence channel.

    Parameters
    ----------
    d_s : int
        Sequence embedding dimension (input dim).
    d_v : int
        Per-head value/projection dimension. Also used as head dim for Q and K.
    H : int
        Number of content heads.
    """

    def __init__(self, d_s: int, d_v: int, H: int) -> None:
        super().__init__()
        if H <= 0:
            raise ValueError(f"H must be positive, got {H}")
        self.d_s = d_s
        self.d_v = d_v
        self.H = H
        # Fused QKV projection: [d_s] → [3 * H * d_v]
        self.W_Q = nn.Linear(d_s, H * d_v, bias=False)
        self.W_K = nn.Linear(d_s, H * d_v, bias=False)
        self.W_V = nn.Linear(d_s, H * d_v, bias=False)
        self._scale = 1.0 / math.sqrt(d_v)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        s : Tensor [B, N, d_s]

        Returns
        -------
        out : Tensor [B, N, H, d_v]
        """
        B, N, _ = s.shape
        q = self.W_Q(s).view(B, N, self.H, self.d_v).transpose(1, 2)  # [B, H, N, d_v]
        k = self.W_K(s).view(B, N, self.H, self.d_v).transpose(1, 2)
        v = self.W_V(s).view(B, N, self.H, self.d_v).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self._scale  # [B, H, N, N]
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, N, d_v]
        return out.transpose(1, 2).contiguous()  # [B, N, H, d_v]
