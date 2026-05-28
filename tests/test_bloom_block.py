"""Integration tests for BloomAttentionBlock."""

from __future__ import annotations

import pytest
import torch

from bloom_arch.attention.block import BloomAttentionBlock


def _make_block(**kwargs):
    defaults = dict(
        d_seq=64,
        d_val=32,
        d_u=16,
        K=32,
        d_v=8,
        n_sequence_heads=2,
        n_position_heads=3,
        n_standard_heads=2,
        d_out=64,
        max_G=30,
        seed=42,
    )
    defaults.update(kwargs)
    return BloomAttentionBlock(**defaults)


def _make_inputs(B=2, L=8, d_seq=64, d_val=32, max_G=30):
    n = torch.randn(B, L, d_seq)
    x = torch.rand(B, L) * 1e7
    g = torch.randint(0, max_G, (B, L))
    m = torch.randn(B, L, d_val)
    return n, x, g, m


def test_output_shape():
    block = _make_block()
    n, x, g, m = _make_inputs()
    out = block(n, x, g, m)
    assert out.shape == (2, 8, 64)


def test_output_shape_custom_d_out():
    block = _make_block(d_out=128)
    n, x, g, m = _make_inputs()
    out = block(n, x, g, m)
    assert out.shape == (2, 8, 128)


def test_all_head_types_run():
    """No errors with all three head counts > 0."""
    torch.manual_seed(0)
    block = _make_block(n_sequence_heads=3, n_position_heads=4, n_standard_heads=2)
    n, x, g, m = _make_inputs()
    out = block(n, x, g, m)
    assert torch.isfinite(out).all()


def test_zero_sequence_heads():
    block = _make_block(n_sequence_heads=0)
    n, x, g, m = _make_inputs()
    out = block(n, x, g, m)
    assert out.shape == (2, 8, 64)


def test_zero_position_heads():
    block = _make_block(n_position_heads=0)
    n, x, g, m = _make_inputs()
    out = block(n, x, g, m)
    assert out.shape == (2, 8, 64)


def test_zero_standard_heads():
    block = _make_block(n_standard_heads=0)
    n, x, g, m = _make_inputs()
    out = block(n, x, g, m)
    assert out.shape == (2, 8, 64)


def test_permutation_equivariance():
    """Permuting L for all inputs permutes output identically."""
    torch.manual_seed(1)
    block = _make_block()
    B, L = 1, 10
    n, x, g, m = _make_inputs(B=B, L=L)

    out_orig = block(n, x, g, m)

    perm = torch.randperm(L)
    out_perm = block(n[:, perm], x[:, perm], g[:, perm], m[:, perm])

    diff = (out_orig[:, perm] - out_perm).abs().max().item()
    assert diff < 1e-4, f"Equivariance violated: max diff = {diff:.2e}"


def test_same_chrom_nearby_preference():
    """Position heads place more attention mass on nearby same-chrom features."""
    torch.manual_seed(0)
    block = BloomAttentionBlock(
        d_seq=32,
        d_val=16,
        d_u=16,
        K=64,
        d_v=8,
        n_sequence_heads=0,
        n_position_heads=1,
        n_standard_heads=0,
        d_out=8,
        max_G=10,
        tau_init=4.0,
        target_scales=[1e5],
    )
    # Feature 0: query on chr 0 @ 50 Mb
    # Features 1-4: same chr, nearby (within 5 kb)
    # Features 5-8: different chr (chr 1), same coordinates
    query_x = 50_000_000.0
    xs = [
        query_x,
        query_x - 5000,
        query_x - 1000,
        query_x + 1000,
        query_x + 5000,
        query_x - 5000,
        query_x - 1000,
        query_x + 1000,
        query_x + 5000,
    ]
    x = torch.tensor([xs])
    g = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1]])

    # To inspect attention, compute the position head logits directly
    pos = block.pos_encoder(x, g)
    u, r_phi = pos["u"], pos["r_phi"]
    ph = block.pos_heads

    w_tiled = ph.w_tiled()
    sqrt_tau = ph.tau.sqrt().view(1, ph.H, 1, 1)
    beta = ph.beta.view(1, ph.H, 1, 1)
    u_part = u.unsqueeze(1) * beta
    r_part = r_phi.unsqueeze(1) * w_tiled.view(1, ph.H, 1, -1)
    qk = torch.cat([u_part, r_part], dim=-1) * sqrt_tau

    logits = torch.matmul(qk, qk.transpose(-1, -2))  # [B, H, L, L]
    attn = torch.softmax(logits, dim=-1)[0, 0]  # [L, L] (single head)
    row = attn[0]  # attention from query feature

    mass_near = row[1:5].sum().item()
    mass_cross = row[5:9].sum().item()
    assert mass_near > mass_cross, (
        f"Nearby same-chrom mass {mass_near:.3f} not > cross-chrom {mass_cross:.3f}"
    )


def test_all_params_receive_gradients():
    """Full backward pass reaches all learnable parameters."""
    torch.manual_seed(2)
    block = _make_block(learn_tau=True, learn_w=True)
    n, x, g, m = _make_inputs()

    out = block(n, x, g, m)
    loss = (out**2).mean()
    loss.backward()

    for name, param in block.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, (
            f"Parameter '{name}' has no gradient"
        )


def test_integer_g_contract():
    """g must be a LongTensor (integer indices)."""
    block = _make_block()
    B, L = 1, 4
    n = torch.randn(B, L, 64)
    x = torch.rand(B, L) * 1e6
    m = torch.randn(B, L, 32)
    g = torch.randint(0, 30, (B, L))
    # Should work with long
    out = block(n, x, g, m)
    assert out.shape == (B, L, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_cuda_smoke():
    """No NaN/Inf in bf16 on CUDA."""
    block = _make_block().cuda().to(torch.bfloat16)
    B, L = 2, 16
    n = torch.randn(B, L, 64, device="cuda", dtype=torch.bfloat16)
    x = (torch.rand(B, L, device="cuda") * 1e7).float()
    g = torch.randint(0, 30, (B, L), device="cuda")
    m = torch.randn(B, L, 32, device="cuda", dtype=torch.bfloat16)
    out = block(n, x, g, m)
    assert out.shape == (B, L, 64)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
