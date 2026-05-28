"""Tests for PositionHeads (spec §3.2, §4.2)."""

from __future__ import annotations

import math

import torch

from bloom_arch.attention.position_heads import PositionHeads


def _make_lambdas(K: int = 64, lam_min: float = 10.0, lam_max: float = 1e8) -> torch.Tensor:
    exponents = torch.arange(K, dtype=torch.float32) / (K - 1)
    return lam_min * ((lam_max / lam_min) ** exponents)


def _make_layer(d_s=32, d_u=16, K=64, d_v=8, H=4, **kwargs):
    lambdas = _make_lambdas(K)
    return PositionHeads(d_s=d_s, d_u=d_u, K=K, d_v=d_v, H=H, lambdas=lambdas, **kwargs)


def test_output_shape():
    layer = _make_layer()
    B, L = 2, 10
    u = torch.randn(B, L, 16)
    r_phi = torch.randn(B, L, 128)  # 2*K=128
    s = torch.randn(B, L, 32)
    out = layer(u, r_phi, s)
    assert out.shape == (B, L, 32)  # H*d_v = 4*8


def test_logit_matches_formula():
    """Verify Q.K^T matches tau * (beta^2 * u.u^T + w^2 * r_phi.r_phi^T)."""
    torch.manual_seed(42)
    K, d_u, H = 32, 8, 2
    layer = _make_layer(d_s=16, d_u=d_u, K=K, d_v=8, H=H)

    B, L = 1, 5
    u = torch.randn(B, L, d_u)
    r_phi = torch.randn(B, L, 2 * K)

    # Compute QK explicitly
    w_tiled = layer.w_tiled()  # [H, 2K]
    sqrt_tau = layer.tau.sqrt().view(1, H, 1, 1)
    beta = layer.beta.view(1, H, 1, 1)

    u_part = u.unsqueeze(1).expand(-1, H, -1, -1) * beta
    r_part = r_phi.unsqueeze(1) * w_tiled.view(1, H, 1, -1)
    qk = torch.cat([u_part, r_part], dim=-1) * sqrt_tau

    # Q.K^T
    logits_from_qk = torch.matmul(qk, qk.transpose(-1, -2))  # [B, H, L, L]

    # Expected formula: tau * (beta^2 * u.u^T + (w*r_phi) @ (w*r_phi)^T)
    tau = layer.tau.view(1, H, 1, 1)
    beta_sq = (layer.beta**2).view(1, H, 1, 1)
    u_dot = torch.matmul(u, u.transpose(-1, -2))  # [B, L, L]
    u_dot = u_dot.unsqueeze(1)  # [B, 1, L, L]

    weighted_r = r_phi.unsqueeze(1) * w_tiled.view(1, H, 1, -1)  # [B, H, L, 2K]
    phi_dot = torch.matmul(weighted_r, weighted_r.transpose(-1, -2))  # [B, H, L, L]

    logits_expected = tau * (beta_sq * u_dot + phi_dot)

    diff = (logits_from_qk - logits_expected).abs().max().item()
    assert diff < 1e-5, f"Logit mismatch: {diff:.2e}"


def test_pad_v_parity():
    """Padded vs unpadded give same result in fp32."""
    torch.manual_seed(7)
    K, d_u = 32, 8
    layer_nopad = _make_layer(d_s=16, d_u=d_u, K=K, d_v=8, H=2, pad_v_for_flash=False)
    layer_pad = _make_layer(d_s=16, d_u=d_u, K=K, d_v=8, H=2, pad_v_for_flash=True)
    layer_pad.load_state_dict(layer_nopad.state_dict())

    B, L = 2, 6
    u = torch.randn(B, L, d_u)
    r_phi = torch.randn(B, L, 2 * K)
    s = torch.randn(B, L, 16)

    out_nopad = layer_nopad(u, r_phi, s)
    out_pad = layer_pad(u, r_phi, s)

    diff = (out_nopad - out_pad).abs().max().item()
    assert diff < 1e-5, f"Pad vs no-pad differ by {diff:.2e}"


def test_w_tiled_sin_cos_sharing():
    """Adjacent pairs in w_tiled are equal (sin/cos share weight)."""
    layer = _make_layer(K=32, H=3)
    w_tiled = layer.w_tiled()  # [3, 64]
    even = w_tiled[:, 0::2]
    odd = w_tiled[:, 1::2]
    assert torch.allclose(even, odd)


def test_head_scale_ordering():
    """Heads with larger target_scale have wider kernels."""
    K = 64
    lambdas = _make_lambdas(K)
    scales = [1e2, 1e4, 1e6]
    layer = PositionHeads(
        d_s=16,
        d_u=8,
        K=K,
        d_v=8,
        H=3,
        lambdas=lambdas,
        target_scales=scales,
        beta_init=0.0,
    )
    # Evaluate kernel at increasing distances
    deltas = torch.logspace(0, 8, steps=200)
    omegas = 2.0 * math.pi / lambdas
    w = layer.log_w.exp()  # [H, K]

    # Same-chrom kernel: (1/K) sum_k w_k^2 * cos(2pi*delta/lambda_k)
    cosines = torch.cos(deltas.unsqueeze(-1) * omegas)  # [D, K]
    kernels = (cosines.unsqueeze(0) * (w**2).unsqueeze(1)).sum(-1) / K  # [H, D]

    # Find half-decay distance for each head
    half_dists = []
    for h in range(3):
        peak = kernels[h, 0].item()
        below = (kernels[h] < 0.5 * peak).nonzero(as_tuple=False)
        if below.numel() == 0:
            half_dists.append(float(deltas[-1]))
        else:
            half_dists.append(float(deltas[below[0].item()]))

    # Should be monotonically increasing
    assert half_dists[1] > half_dists[0], f"{half_dists}"
    assert half_dists[2] > half_dists[1], f"{half_dists}"


def test_learn_tau_creates_parameter():
    layer = _make_layer(learn_tau=True)
    param_names = {n for n, _ in layer.named_parameters()}
    assert "log_tau" in param_names


def test_learn_tau_false_creates_buffer():
    layer = _make_layer(learn_tau=False)
    param_names = {n for n, _ in layer.named_parameters()}
    assert "log_tau" not in param_names
    buf_names = {n for n, _ in layer.named_buffers()}
    assert "log_tau" in buf_names


def test_learn_w_raises():
    """learn_w=True raises NotImplementedError (not yet implemented)."""
    import pytest

    with pytest.raises(NotImplementedError, match="reparameterization"):
        _make_layer(learn_w=True)


def test_learn_w_false_creates_buffer():
    layer = _make_layer(learn_w=False)
    param_names = {n for n, _ in layer.named_parameters()}
    assert "log_w" not in param_names


def test_gradients_flow():
    """All learnable params receive gradients."""
    torch.manual_seed(3)
    layer = _make_layer(d_s=16, d_u=8, K=32, d_v=8, H=2, learn_tau=True, beta_init=1.0)

    B, L = 2, 6
    u = torch.randn(B, L, 8)
    r_phi = torch.randn(B, L, 64)
    s = torch.randn(B, L, 16)

    out = layer(u, r_phi, s)
    loss = (out**2).mean()
    loss.backward()

    assert layer.beta.grad is not None and layer.beta.grad.abs().sum() > 0
    assert layer.log_tau.grad is not None and layer.log_tau.grad.abs().sum() > 0
    assert layer.W_V.weight.grad is not None and layer.W_V.weight.grad.abs().sum() > 0
