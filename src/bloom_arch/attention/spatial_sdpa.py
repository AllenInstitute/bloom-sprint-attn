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
SDPA so it does NOT apply its default ``1/sqrt(d_k)`` scaling — ``tau_h`` is
the only scaling we want.

Shape contract for SDPA: ``[B, H, N, head_dim]``.

Backend dispatch
----------------
PyTorch SDPA chooses between FLASH_ATTENTION, EFFICIENT_ATTENTION,
CUDNN_ATTENTION, and MATH backends. Flash kicks in when:

  - dtype is float16 or bfloat16,
  - device is CUDA with compute capability >= 8.0,
  - per-head dim <= 256 (PyTorch >= 2.2; 128 in older versions).

With spec defaults (``d_u=32``, ``K=112``), our head_dim is
``32 + 2*112 = 256`` — exactly at the flash upper limit.
``scripts/check_sdpa_backends.py`` reports which kernel actually dispatches
on the target device.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from bloom_arch.attention.spatial import SpatialAttention


class SpatialAttentionSDPA(SpatialAttention):
    """Drop-in SDPA implementation of :class:`SpatialAttention`.

    Same constructor signature, same parameter shapes, same initialization,
    so ``state_dict`` round-trips with the parent class. The only change is
    the forward path: instead of an explicit ``einsum`` + ``softmax``, we
    build per-head Q,K,V tensors and call
    ``torch.nn.functional.scaled_dot_product_attention``.
    """

    def forward(  # type: ignore[override]
        self,
        u: torch.Tensor,        # [B, N, d_u]
        r_phi: torch.Tensor,    # [B, N, 2K]
        s: torch.Tensor,        # [B, N, d_s]
    ) -> torch.Tensor:
        B, N, _ = r_phi.shape
        if u.shape[:2] != (B, N) or s.shape[:2] != (B, N):
            raise ValueError("u, r_phi, s must agree on [B, N]")
        if r_phi.shape[-1] != 2 * self.K:
            raise ValueError(
                f"r_phi last dim must be 2*K = {2 * self.K}, got {r_phi.shape[-1]}"
            )

        H, d_v = self.H, self.d_v
        w_tiled = self.w_tiled()                                          # [H, 2K]
        # beta_h and sqrt(tau_h) broadcasted over heads.
        beta = self.beta.view(1, H, 1, 1)                                  # [1, H, 1, 1]
        sqrt_tau = self.tau.sqrt().view(1, H, 1, 1)                        # [1, H, 1, 1]

        # u-part for QK: beta_h * u, broadcast across heads -> [B, H, N, d_u].
        u_part = u.unsqueeze(1) * beta
        # r-part for QK: w_tiled[h] (.) r_phi, per head -> [B, H, N, 2K].
        r_part = r_phi.unsqueeze(1) * w_tiled.view(1, H, 1, -1)

        # Concatenate and absorb sqrt(tau_h) so SDPA's scale=1.0 gives tau_h * Q.K^T.
        qk = torch.cat([u_part, r_part], dim=-1) * sqrt_tau                # [B, H, N, d_u+2K]

        # V: project sequence into per-head values.
        v = self.W_V(s).view(B, N, H, d_v).transpose(1, 2).contiguous()    # [B, H, N, d_v]

        # Positional self-attention: Q == K. SDPA expects the same dtype for Q, K, V.
        # Caller controls precision via .to(dtype) on the module + inputs.
        out = F.scaled_dot_product_attention(qk, qk, v, scale=1.0)         # [B, H, N, d_v]
        return out.transpose(1, 2).contiguous()                             # [B, N, H, d_v]

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
