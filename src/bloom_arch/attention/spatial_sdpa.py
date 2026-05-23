"""Flash-Attention-compatible spatial attention using ``F.scaled_dot_product_attention``.

Same parameters and mathematical behavior as :class:`SpatialAttention`, but
expresses the per-head logit as :math:`Q \\cdot K^T` so PyTorch's fused SDPA
kernel runs the softmax-matmul step. On Ampere/Ada/Hopper GPUs with
fp16/bf16 inputs, SDPA dispatches to a Flash Attention kernel automatically.

How the kernel maps to Q.K^T
----------------------------
The spec section 12.2 logit for spatial head ``h`` between features ``i``
and ``j`` is

    l_{ij}^{(h)} = tau_h * (beta_h^2 * u_i.u_j
                            + sum_d w_tiled[h,d]^2 * r_phi_i[d] * r_phi_j[d])

Construct per-head Q^{(h)}, K^{(h)} of dimension ``d_u + 2K`` such that

    Q^{(h)}_i = sqrt(tau_h) * [beta_h * u_i ;  w_tiled[h] (.) r_phi_i]
    K^{(h)}_j = sqrt(tau_h) * [beta_h * u_j ;  w_tiled[h] (.) r_phi_j]

Then Q^{(h)}_i . K^{(h)}_j = l_{ij}^{(h)} exactly. We pass ``scale=1.0`` to
SDPA so it does NOT apply its default ``1/sqrt(d_k)`` scaling -- ``tau_h``
is the only scaling we want.

Shape contract for SDPA: ``[B, H, N, head_dim]``.

Backend dispatch and the ``pad_v_for_flash`` switch
---------------------------------------------------
PyTorch SDPA chooses between FLASH_ATTENTION, EFFICIENT_ATTENTION,
CUDNN_ATTENTION, and MATH backends. Flash kicks in when:

  - dtype is float16 or bfloat16,
  - device is CUDA with compute capability >= 8.0,
  - per-head dim <= 256 (PyTorch >= 2.2; 128 in older versions),
  - **head_dim(Q) == head_dim(K) == head_dim(V)** -- flash requires the
    three to match.

With spec defaults (``d_u=32``, ``K=112``), our QK head_dim is
``32 + 2*112 = 256`` but our V head_dim is ``d_v = 64``, so the asymmetry
disqualifies flash. The xFormers-lineage EFFICIENT_ATTENTION backend has
no such restriction and dispatches in bf16/fp16.

If you want a true flash dispatch, instantiate with
``pad_v_for_flash=True``. The forward then right-pads ``V`` with zeros
along the head dimension up to the QK head dim before SDPA, and slices
the output back to ``d_v`` afterward. Padding with zeros is
mathematically inert: SDPA's output is a softmax-weighted sum over the
sequence axis, so the padded V columns produce padded output columns,
which we discard. Cost: V tile is ``head_dim_qk / d_v`` times larger
(~4x with spec defaults); attention scores are unchanged.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from bloom_arch.attention.spatial import SpatialAttention


class SpatialAttentionSDPA(SpatialAttention):
    """Drop-in SDPA implementation of :class:`SpatialAttention`.

    Same constructor signature as :class:`SpatialAttention`, with one
    extra keyword:

    Parameters
    ----------
    pad_v_for_flash : bool, default False
        If True, pad ``V`` with zeros along the head dimension to match
        the QK head dim before calling SDPA, then slice the output back
        to ``d_v``. Required to satisfy PyTorch's flash-attention
        backend, which insists on equal Q/K/V head dims. Has no effect
        on the math; only on which SDPA backend dispatches.

    Notes
    -----
    State dict round-trips with :class:`SpatialAttention` (no new
    parameters are added by either ``__init__`` extension or the
    padding switch).
    """

    def __init__(
        self,
        *args,
        pad_v_for_flash: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pad_v_for_flash = pad_v_for_flash

    def forward(  # type: ignore[override]
        self,
        u: torch.Tensor,  # [B, N, d_u]
        r_phi: torch.Tensor,  # [B, N, 2K]
        s: torch.Tensor,  # [B, N, d_s]
    ) -> torch.Tensor:
        B, N, _ = r_phi.shape
        if u.shape[:2] != (B, N) or s.shape[:2] != (B, N):
            raise ValueError("u, r_phi, s must agree on [B, N]")
        if r_phi.shape[-1] != 2 * self.K:
            raise ValueError(f"r_phi last dim must be 2*K = {2 * self.K}, got {r_phi.shape[-1]}")

        H, d_v = self.H, self.d_v
        w_tiled = self.w_tiled()  # [H, 2K]
        # beta_h and sqrt(tau_h) broadcasted over heads.
        beta = self.beta.view(1, H, 1, 1)  # [1, H, 1, 1]
        sqrt_tau = self.tau.sqrt().view(1, H, 1, 1)  # [1, H, 1, 1]

        # u-part for QK: beta_h * u, broadcast across heads -> [B, H, N, d_u].
        u_part = u.unsqueeze(1) * beta
        # r-part for QK: w_tiled[h] (.) r_phi, per head -> [B, H, N, 2K].
        r_part = r_phi.unsqueeze(1) * w_tiled.view(1, H, 1, -1)

        # Concatenate and absorb sqrt(tau_h) so SDPA's scale=1.0 gives tau_h * Q.K^T.
        qk = torch.cat([u_part, r_part], dim=-1) * sqrt_tau  # [B, H, N, head_dim_qk]
        head_dim_qk = qk.shape[-1]

        # V: project sequence into per-head values.
        v = self.W_V(s).view(B, N, H, d_v).transpose(1, 2).contiguous()  # [B, H, N, d_v]

        if self.pad_v_for_flash and head_dim_qk > d_v:
            # Pad V with zeros along the head dim to match QK. Zeros are
            # inert under softmax-weighted summation, so the padded output
            # columns can be sliced off after SDPA without changing the
            # result.
            v_padded = F.pad(v, (0, head_dim_qk - d_v))  # [B, H, N, head_dim_qk]
            out_padded = F.scaled_dot_product_attention(qk, qk, v_padded, scale=1.0)
            out = out_padded[..., :d_v].contiguous()  # [B, H, N, d_v]
        else:
            # Positional self-attention: Q == K. SDPA expects the same
            # dtype for Q, K, V; caller controls precision via .to(dtype).
            out = F.scaled_dot_product_attention(qk, qk, v, scale=1.0)  # [B, H, N, d_v]
        return out.transpose(1, 2).contiguous()  # [B, N, H, d_v]

    def forward_explicit(
        self,
        u: torch.Tensor,
        r_phi: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Reference math via the parent class's explicit logit computation.

        Useful as an oracle in tests and for differential debugging when SDPA
        outputs look suspect.
        """
        return SpatialAttention.forward(self, u, r_phi, s)
