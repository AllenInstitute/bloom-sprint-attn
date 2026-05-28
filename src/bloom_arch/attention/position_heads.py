"""Position attention heads.

Q = K = sqrt(tau_h) * [beta_h * u_g; w_tiled_h * r_phi_i]
V from projected content encoding s.
Uses F.scaled_dot_product_attention with scale=1.0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _log_gaussian_init(
    K: int,
    lambdas: torch.Tensor,
    target_scales: Sequence[float],
    rho: float,
) -> torch.Tensor:
    """Log-Gaussian window initialization for wavelength weights.

    Returns log_w of shape [H, K] such that w = exp(log_w) is a
    log-Gaussian centered at each target scale, normalized so ||w||^2 = K.
    """
    log_lambdas = torch.log(lambdas)  # [K]
    log_L = torch.log(torch.tensor(list(target_scales), dtype=lambdas.dtype))  # [H]
    diff = log_lambdas.unsqueeze(0) - log_L.unsqueeze(1)  # [H, K]
    log_w_unnorm = -(diff**2) / (2.0 * rho**2)
    # Exponentiate to get unnormalized w, normalize to ||w||^2 = K, take log
    w_unnorm = log_w_unnorm.exp()
    w_norm = w_unnorm * (K**0.5) / w_unnorm.norm(dim=-1, keepdim=True)
    return w_norm.log()


class PositionHeads(nn.Module):
    """Multi-head position attention with per-head wavelength weighting.

    Constructs Q = K = sqrt(tau) * [beta * u_g; w_tiled * r_phi] so that
    Q.K^T = tau * (beta^2 * u_i.u_j + (1/K) sum_k w_k^2 cos(2pi*Delta/lam_k)).
    Uses SDPA with scale=1.0 (tau baked into Q=K).

    Args:
        d_s: Content encoding dimension (input for V projection).
        d_u: Chromosome baseline dimension (from codebook).
        K: Number of Fourier wavelengths (d_phi = 2K).
        d_v: Per-head value dimension.
        H: Number of position heads.
        lambdas: Tensor ``[K]`` -- wavelength grid from the Fourier encoder.
        target_scales: Per-head center wavelengths for log-Gaussian init.
            If None, log-spaced between lambdas.min()*10 and lambdas.max()/10.
        rho: Log-Gaussian bandwidth in natural log units.
        beta_init: Initial value for per-head beta.
        tau_init: Initial value for per-head tau.
        learn_tau: If True, tau is a learnable parameter. Otherwise a buffer.
        learn_w: Reserved. Must be False (learnable w requires
            reparameterization for ||w||^2=K and non-negativity; not yet
            implemented).
        pad_v_for_flash: If True, zero-pad V to match QK head_dim for flash
            attention dispatch. Mathematically inert.
    """

    def __init__(
        self,
        d_s: int,
        d_u: int,
        K: int,
        d_v: int,
        H: int,
        lambdas: torch.Tensor,
        target_scales: Sequence[float] | None = None,
        rho: float = math.log(10.0) / 2.0,
        beta_init: float = 0.0,
        tau_init: float = 1.0,
        learn_tau: bool = False,
        learn_w: bool = False,
        pad_v_for_flash: bool = False,
    ) -> None:
        super().__init__()
        if learn_w:
            raise NotImplementedError(
                "learn_w=True requires reparameterization to maintain "
                "||w_h||^2 = K and non-negativity; not yet implemented"
            )

        self.d_s = d_s
        self.d_u = d_u
        self.K = K
        self.d_v = d_v
        self.H = H
        self.pad_v_for_flash = pad_v_for_flash

        # Derive target scales if not provided
        if target_scales is None:
            lo = float(lambdas.min()) * 10.0
            hi = float(lambdas.max()) / 10.0
            if H == 1:
                target_scales = [math.sqrt(lo * hi)]
            else:
                log_lo, log_hi = math.log(lo), math.log(hi)
                target_scales = [
                    math.exp(log_lo + (log_hi - log_lo) * h / (H - 1)) for h in range(H)
                ]

        self.register_buffer("lambdas", lambdas.detach().clone())

        # Per-head wavelength weights in log space (fixed at init)
        log_w_init = _log_gaussian_init(K, lambdas, target_scales, rho)
        self.register_buffer("log_w", log_w_init)

        # Per-head beta
        self.beta = nn.Parameter(torch.full((H,), float(beta_init)))

        # Per-head tau in log space
        log_tau_init = torch.full((H,), math.log(tau_init))
        if learn_tau:
            self.log_tau = nn.Parameter(log_tau_init)
        else:
            self.register_buffer("log_tau", log_tau_init)

        # Value projection
        self.W_V = nn.Linear(d_s, H * d_v, bias=False)

    @property
    def head_dim_qk(self) -> int:
        return self.d_u + 2 * self.K

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def w_tiled(self) -> torch.Tensor:
        """[H, 2K] — w values repeated across sin/cos pairs."""
        return self.log_w.exp().repeat_interleave(2, dim=-1)

    def forward(
        self,
        u: torch.Tensor,
        r_phi: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Run position attention.

        Args:
            u: Tensor ``[B, L, d_u]`` -- chromosome baseline codes.
            r_phi: Tensor ``[B, L, 2K]`` -- sign-gated Fourier encoding.
            s: Tensor ``[B, L, d_s]`` -- content encoding for V projection.

        Returns:
            Tensor ``[B, L, H * d_v]`` -- concatenated per-head outputs.
        """
        B, L, _ = r_phi.shape
        H, d_v = self.H, self.d_v

        w_tiled = self.w_tiled()  # [H, 2K]
        sqrt_tau = self.tau.sqrt().view(1, H, 1, 1)
        beta = self.beta.view(1, H, 1, 1)

        # Build QK vector: [B, H, L, d_u + 2K]
        u_part = u.unsqueeze(1).expand(-1, H, -1, -1) * beta  # [B, H, L, d_u]
        r_part = r_phi.unsqueeze(1) * w_tiled.view(1, H, 1, -1)  # [B, H, L, 2K]
        qk = torch.cat([u_part, r_part], dim=-1) * sqrt_tau  # [B, H, L, head_dim_qk]

        # V: project content
        v = self.W_V(s).view(B, L, H, d_v).transpose(1, 2).contiguous()  # [B, H, L, d_v]

        # Ensure QK and V share dtype (codebook buffers may be fp32 while model is bf16)
        qk = qk.to(dtype=v.dtype)

        head_dim_qk = qk.shape[-1]

        if self.pad_v_for_flash and head_dim_qk > d_v:
            v_padded = F.pad(v, (0, head_dim_qk - d_v))
            out_padded = F.scaled_dot_product_attention(qk, qk, v_padded, scale=1.0)
            out = out_padded[..., :d_v]
        else:
            out = F.scaled_dot_product_attention(qk, qk, v, scale=1.0)

        return out.transpose(1, 2).reshape(B, L, H * d_v)
