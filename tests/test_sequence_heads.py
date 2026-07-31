"""Tests for SequenceHeads (spec §3.1)."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

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


def test_scale_is_per_head_dim():
    """Scale is 1/sqrt(d_v), not 1/sqrt(d_seq) (spec §3.1).

    The QK dot product runs over the per-head projected dimension d_v, so the
    softmax temperature must be 1/sqrt(d_v) — matching StandardHeads and SDPA's
    default. Uses d_seq != d_v so the wrong (input-dim) scaling would fail.
    """
    layer = SequenceHeads(d_seq=64, d_s=96, d_v=8, H=4)
    assert layer._scale == pytest.approx(1.0 / math.sqrt(8))
    assert layer._scale != pytest.approx(1.0 / math.sqrt(64))


def test_forward_matches_reference_scaling():
    """End-to-end: forward equals softmax(QK^T / sqrt(d_v)) @ V.

    Reconstructs the reference from the layer's own projections, pinning the
    effective scale (not just the stored constant) to the per-head dimension.
    """
    torch.manual_seed(3)
    d_seq, d_s, d_v, H = 64, 96, 8, 4
    B, L = 2, 7
    layer = SequenceHeads(d_seq=d_seq, d_s=d_s, d_v=d_v, H=H)
    n = torch.randn(B, L, d_seq)
    s = torch.randn(B, L, d_s)

    out = layer(n, s)

    # Hand-rolled reference using the layer's own projected q/k/v.
    q = layer.W_Q(n).view(B, L, H, d_v).transpose(1, 2)
    k = layer.W_K(n).view(B, L, H, d_v).transpose(1, 2)
    v = layer.W_V(s).view(B, L, H, d_v).transpose(1, 2)
    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_v)
    ref = torch.matmul(F.softmax(logits, dim=-1), v)
    ref = ref.transpose(1, 2).reshape(B, L, H * d_v)

    diff = (out - ref).abs().max().item()
    assert diff < 1e-5, f"Forward does not match 1/sqrt(d_v) reference: {diff:.2e}"
