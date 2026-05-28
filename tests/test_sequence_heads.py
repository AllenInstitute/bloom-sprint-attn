"""Tests for SequenceHeads (spec §3.1)."""

from __future__ import annotations

import torch

from bloom_arch.attention.sequence_heads import SequenceHeads


def test_output_shape():
    """Output shape is [B, L, H * d_v]."""
    layer = SequenceHeads(d_seq=64, d_s=96, d_v=16, H=4)
    n = torch.randn(2, 10, 64)
    s = torch.randn(2, 10, 96)
    out = layer(n, s)
    assert out.shape == (2, 10, 64)  # H*d_v = 4*16 = 64


def test_permutation_equivariance():
    """Permuting the L axis of inputs permutes output identically."""
    torch.manual_seed(0)
    layer = SequenceHeads(d_seq=32, d_s=48, d_v=8, H=4)
    B, L = 1, 8
    n = torch.randn(B, L, 32)
    s = torch.randn(B, L, 48)

    out_orig = layer(n, s)

    perm = torch.randperm(L)
    out_perm = layer(n[:, perm], s[:, perm])

    diff = (out_orig[:, perm] - out_perm).abs().max().item()
    assert diff < 1e-5, f"Equivariance violated: max diff = {diff:.2e}"


def test_gradients_flow():
    """All projection matrices receive gradients."""
    torch.manual_seed(1)
    layer = SequenceHeads(d_seq=32, d_s=48, d_v=8, H=2)
    n = torch.randn(2, 6, 32)
    s = torch.randn(2, 6, 48)

    out = layer(n, s)
    loss = (out**2).mean()
    loss.backward()

    assert layer.W_Q.weight.grad is not None and layer.W_Q.weight.grad.abs().sum() > 0
    assert layer.W_K.weight.grad is not None and layer.W_K.weight.grad.abs().sum() > 0
    assert layer.W_V.weight.grad is not None and layer.W_V.weight.grad.abs().sum() > 0


def test_different_n_and_s_dims():
    """Works when d_seq != d_s (the typical case: d_s = d_seq + d_val)."""
    layer = SequenceHeads(d_seq=64, d_s=128, d_v=16, H=3)
    n = torch.randn(1, 5, 64)
    s = torch.randn(1, 5, 128)
    out = layer(n, s)
    assert out.shape == (1, 5, 48)  # H*d_v = 3*16


def test_single_token():
    """Works with L=1 (degenerate case)."""
    layer = SequenceHeads(d_seq=32, d_s=48, d_v=8, H=2)
    n = torch.randn(2, 1, 32)
    s = torch.randn(2, 1, 48)
    out = layer(n, s)
    assert out.shape == (2, 1, 16)
    assert torch.isfinite(out).all()
