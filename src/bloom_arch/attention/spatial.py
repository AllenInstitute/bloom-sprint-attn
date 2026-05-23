"""Spatial attention head (spec §5.2, §12).

Each spatial head uses the positional encoding for queries and keys, and a
learned linear projection of the sequence embedding for values:

    q_i^(h) = p_i^(h)
    k_j^(h) = p_j^(h)
    v_j^(h) = W_V^(h) s_j

With ``use_baseline=True`` (default):

    p_i^(h) = [β_h u_g(i) ; r_g(i) ⊙ (w_h ⊙ φ(x_i))]

    ℓ_ij^(h) = τ_h · (β_h² · u_i·u_j + Σ_d w_h_tied[d]² · r_phi_i[d] · r_phi_j[d])

With ``use_baseline=False``, the β·u term is dropped:

    p_i^(h) = r_g(i) ⊙ (w_h ⊙ φ(x_i))

    ℓ_ij^(h) = τ_h · Σ_d w_h_tied[d]² · r_phi_i[d] · r_phi_j[d]

``w_h_tied`` repeats each scalar weight across the (sin, cos) pair of its
wavelength so the head specializes coherently per wavelength rather than per
individual basis dimension.

Parameters per head (learnable by default, individually freezable):
    - ``log_w_h``  : ℝ^K, parameterized in log-space (``learn_w=False`` to freeze)
    - ``β_h``      : scalar, present only when ``use_baseline=True``
    - ``log_tau_h``: scalar temperature in log-space (``learn_tau=False`` to freeze)
    - ``W_V^(h)``  : Linear(d_s, d_v) — always learnable
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def _default_log_gaussian_init(
    K: int,
    lambdas: torch.Tensor,
    target_scales: Sequence[float],
    rho: float,
) -> torch.Tensor:
    """log-Gaussian window over log-wavelengths (spec §12.3).

    Returns ``log_w`` of shape ``[H, K]`` with
        log_w[h, k] = -((log λ_k - log L_h)^2) / (2 ρ²)

    so ``w[h, k] = exp(log_w[h, k])`` is a log-Gaussian centered at ``L_h``.
    """
    log_lambdas = torch.log(lambdas)  # [K]
    log_L = torch.log(
        torch.tensor(list(target_scales), dtype=lambdas.dtype, device=lambdas.device)
    )  # [H]
    diff = log_lambdas.unsqueeze(0) - log_L.unsqueeze(1)  # [H, K]
    return -(diff**2) / (2.0 * rho**2)


class SpatialAttention(nn.Module):
    """Multi-head spatial attention with per-head log-Gaussian wavelength windows.

    Parameters
    ----------
    d_s : int
        Sequence embedding dimension (input dim for values).
    d_v : int
        Per-head value dimension.
    H : int
        Number of spatial heads.
    K : int
        Number of Fourier wavelengths (``2*K = d_phi``).
    lambdas : Tensor of shape [K]
        The wavelengths from the Fourier encoder. Used to initialize ``log_w``;
        registered as a non-learnable buffer for plotting/inspection.
    target_scales : sequence of float, length H, optional
        Initial center wavelengths per head (bp). If omitted, defaults to
        ``H`` log-spaced scales between ``lambdas.min() * 10`` and ``lambdas.max() / 10``.
    rho : float
        Broadness of the log-Gaussian window in natural log units. Default
        ``log(10)/2`` ≈ 1.15 (FWHM ≈ 1.2 decades).
    beta_init : float
        Initial value for β_h. Only used when ``use_baseline=True``.
    tau_init : float
        Initial value for τ_h.
    use_baseline : bool
        If True (default), include the β_h² · (u_i · u_j) same-chromosome
        baseline term in the logit and require ``u`` in forward(). If False,
        the logit consists solely of the Fourier r_phi term, ``beta`` is not
        created as a parameter, and ``u`` may be None.
    learn_tau : bool
        If True (default), τ_h is a learnable parameter. If False, τ_h is
        held fixed at ``tau_init`` (registered as a buffer).
    learn_w : bool
        If True (default), log_w is a learnable parameter. If False, the
        wavelength weights are held fixed at their log-Gaussian
        initialization (registered as a buffer).
    """

    def __init__(
        self,
        d_s: int,
        d_v: int,
        H: int,
        K: int,
        lambdas: torch.Tensor,
        target_scales: Sequence[float] | None = None,
        rho: float = math.log(10.0) / 2.0,
        beta_init: float = 1.0,
        tau_init: float = 1.0,
        use_baseline: bool = True,
        learn_tau: bool = True,
        learn_w: bool = True,
    ) -> None:
        super().__init__()
        if lambdas.shape != (K,):
            raise ValueError(f"lambdas must have shape [K]=[{K}], got {tuple(lambdas.shape)}")
        if H <= 0:
            raise ValueError(f"H must be positive, got {H}")
        self.d_s = d_s
        self.d_v = d_v
        self.H = H
        self.K = K
        self.use_baseline = use_baseline
        self.learn_tau = learn_tau
        self.learn_w = learn_w

        if target_scales is None:
            lo = float(lambdas.min()) * 10.0
            hi = float(lambdas.max()) / 10.0
            if H == 1:
                scales = [math.sqrt(lo * hi)]
            else:
                # Geometric spacing across the usable range.
                log_lo, log_hi = math.log(lo), math.log(hi)
                scales = [math.exp(log_lo + (log_hi - log_lo) * h / (H - 1)) for h in range(H)]
            target_scales = scales

        if len(target_scales) != H:
            raise ValueError(f"target_scales must have length H={H}, got {len(target_scales)}")

        self.register_buffer("lambdas", lambdas.detach().clone())
        self.target_scales = tuple(float(s) for s in target_scales)
        self.rho = rho

        # Per-head wavelength weights (log-space).
        log_w_init = _default_log_gaussian_init(K, self.lambdas, target_scales, rho)  # [H, K]
        if learn_w:
            self.log_w = nn.Parameter(log_w_init)
        else:
            self.register_buffer("log_w", log_w_init)

        # Per-head baseline (optional).
        self.beta = nn.Parameter(torch.full((H,), float(beta_init))) if use_baseline else None

        # Per-head temperature (log-space).
        log_tau_init = torch.full((H,), math.log(tau_init))
        if learn_tau:
            self.log_tau = nn.Parameter(log_tau_init)
        else:
            self.register_buffer("log_tau", log_tau_init)

        # Sequence → values.
        self.W_V = nn.Linear(d_s, H * d_v, bias=False)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def w(self) -> torch.Tensor:
        """Return ``w_h`` (nonnegative) of shape [H, K]."""
        return self.log_w.exp()

    def w_tiled(self) -> torch.Tensor:
        """``w_h`` repeated across the (sin, cos) pair for each wavelength → [H, 2K]."""
        return self.w().repeat_interleave(2, dim=-1)

    def kernel_same_chrom(self, delta: torch.Tensor) -> torch.Tensor:
        """Compute the same-chromosome kernel K_h(Δ) for visualization.

        For two same-chromosome features with coordinate difference Δ, the
        spec §12.2 logit (before τ scaling) is

            K_h(Δ) = β_h² + (1/K) Σ_k w_{h,k}² cos(2π Δ / λ_k)

        When ``use_baseline=False``, the β² term is omitted.

        Parameters
        ----------
        delta : Tensor of shape [D]
            Coordinate differences in bp.

        Returns
        -------
        kernel : Tensor of shape [H, D]
            Per-head same-chromosome kernel values.
        """
        omegas = (2.0 * math.pi / self.lambdas).to(delta.device, delta.dtype)  # [K]
        cosines = torch.cos(delta.unsqueeze(-1) * omegas)  # [D, K]
        w = self.w()  # [H, K]
        phi_term = (cosines.unsqueeze(0) * (w**2).unsqueeze(1)).sum(-1) / self.K  # [H, D]
        if self.beta is not None:
            return self.beta.unsqueeze(-1) ** 2 + phi_term
        return phi_term

    def forward(
        self,
        u: torch.Tensor | None,  # [B, N, d_u] or None when use_baseline=False
        r_phi: torch.Tensor,  # [B, N, 2K]
        s: torch.Tensor,  # [B, N, d_s]
    ) -> torch.Tensor:
        """Run multi-head spatial attention.

        Parameters
        ----------
        u : Tensor [B, N, d_u] or None
            Per-feature u_g code. Required when ``use_baseline=True``, ignored
            (may be None) otherwise.
        r_phi : Tensor [B, N, 2K]
            Per-feature r_g ⊙ φ(x).
        s : Tensor [B, N, d_s]
            Per-feature sequence embedding (only used to produce values).

        Returns
        -------
        out : Tensor [B, N, H, d_v]
            Per-head attention output.
        """
        B, N, _ = r_phi.shape
        if s.shape[:2] != (B, N):
            raise ValueError("r_phi and s must agree on [B, N]")
        if r_phi.shape[-1] != 2 * self.K:
            raise ValueError(f"r_phi last dim must be 2*K = {2 * self.K}, got {r_phi.shape[-1]}")

        # r_phi term: weight by w_tiled per head, then inner product.
        w_tiled = self.w_tiled()  # [H, 2K]
        weighted_r = r_phi.unsqueeze(2) * w_tiled.view(1, 1, self.H, -1)  # [B, N, H, 2K]
        phi_logit = torch.einsum("bihd,bjhd->bhij", weighted_r, weighted_r)  # [B, H, N, N]

        # Baseline u term: β_h² · (u_i · u_j).
        if self.beta is not None:
            if u is None:
                raise ValueError("u is required when use_baseline=True")
            if u.shape[:2] != (B, N):
                raise ValueError("u, r_phi, s must agree on [B, N]")
            u_dot = torch.matmul(u, u.transpose(-1, -2))  # [B, N, N]
            beta_sq = (self.beta**2).view(1, self.H, 1, 1)  # [1, H, 1, 1]
            logit = phi_logit + beta_sq * u_dot.unsqueeze(1)
        else:
            logit = phi_logit

        # Scale by τ_h and apply softmax over keys.
        tau = self.tau.view(1, self.H, 1, 1)  # [1, H, 1, 1]
        attn = torch.softmax(tau * logit, dim=-1)  # [B, H, N, N]

        # Values: project sequence and reshape to [B, H, N, d_v].
        v = self.W_V(s).view(B, N, self.H, self.d_v).transpose(1, 2)  # [B, H, N, d_v]
        out = torch.matmul(attn, v)  # [B, H, N, d_v]
        return out.transpose(1, 2).contiguous()  # [B, N, H, d_v]
