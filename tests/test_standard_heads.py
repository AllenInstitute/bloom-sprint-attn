"""Tests for StandardHeads (spec §3, head type 3)."""

from __future__ import annotations

import torch

from bloom_arch.attention.standard_heads import StandardHeads


def test_output_shape():
    """Output shape is [B, L, H * d_v]."""
    layer = StandardHeads(d_in=256, d_v=16, H=4)
    f = torch.randn(2, 10, 256)
    out = layer(f)
    assert out.shape == (2, 10, 64)  # H*d_v = 4*16


def test_permutation_equivariance():
    """Permuting the L axis of input permutes output identically."""
    torch.manual_seed(0)
    layer = StandardHeads(d_in=128, d_v=8, H=4)
    B, L = 1, 8
    f = torch.randn(B, L, 128)

    out_orig = layer(f)

    perm = torch.randperm(L)
    out_perm = layer(f[:, perm])

    diff = (out_orig[:, perm] - out_perm).abs().max().item()
    assert diff < 1e-5, f"Equivariance violated: max diff = {diff:.2e}"


def test_gradients_flow():
    """All projection matrices receive gradients."""
    torch.manual_seed(1)
    layer = StandardHeads(d_in=128, d_v=8, H=2)
    f = torch.randn(2, 6, 128)

    out = layer(f)
    loss = (out**2).mean()
    loss.backward()

    assert layer.W_Q.weight.grad is not None and layer.W_Q.weight.grad.abs().sum() > 0
    assert layer.W_K.weight.grad is not None and layer.W_K.weight.grad.abs().sum() > 0
    assert layer.W_V.weight.grad is not None and layer.W_V.weight.grad.abs().sum() > 0


def test_single_token():
    """Works with L=1 (degenerate but valid)."""
    layer = StandardHeads(d_in=64, d_v=8, H=2)
    f = torch.randn(2, 1, 64)
    out = layer(f)
    assert out.shape == (2, 1, 16)
    assert torch.isfinite(out).all()


def test_large_d_in():
    """Works with large input dim (typical: d_seq=512 + d_u=32 + 2K=224 + d_val=256 = 1024)."""
    layer = StandardHeads(d_in=1024, d_v=64, H=8)
    f = torch.randn(1, 4, 1024)
    out = layer(f)
    assert out.shape == (1, 4, 512)  # H*d_v = 8*64
