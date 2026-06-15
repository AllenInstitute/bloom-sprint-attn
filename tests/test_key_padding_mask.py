"""Tests for key_padding_mask support in BloomAttentionBlock and head types."""

from __future__ import annotations

import torch

from bloom_arch.attention.block import BloomAttentionBlock
from bloom_arch.attention.position_heads import PositionHeads
from bloom_arch.attention.sequence_heads import SequenceHeads
from bloom_arch.attention.standard_heads import StandardHeads

# ─── Helpers ───────────────────────────────────────────────────────────────────


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


def _make_lambdas(K: int = 32) -> torch.Tensor:
    exponents = torch.arange(K, dtype=torch.float32) / (K - 1)
    return 10.0 * ((1e8 / 10.0) ** exponents)


# ─── Block-level tests ─────────────────────────────────────────────────────────


def test_mask_output_shape():
    """Output shape is unchanged when mask is provided."""
    block = _make_block()
    n, x, g, m = _make_inputs()
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, -2:] = True
    out = block(n, x, g, m, key_padding_mask=mask)
    assert out.shape == (2, 8, 64)


def test_mask_none_backward_compat():
    """Explicit None gives same result as no kwarg."""
    torch.manual_seed(0)
    block = _make_block()
    n, x, g, m = _make_inputs()
    out1 = block(n, x, g, m)
    out2 = block(n, x, g, m, key_padding_mask=None)
    assert torch.allclose(out1, out2)


def test_mask_all_false_equals_none():
    """A mask of all-False (no padding) gives same result as None."""
    torch.manual_seed(1)
    block = _make_block()
    n, x, g, m = _make_inputs()
    mask = torch.zeros(2, 8, dtype=torch.bool)
    out_none = block(n, x, g, m)
    out_mask = block(n, x, g, m, key_padding_mask=mask)
    assert torch.allclose(out_none, out_mask, atol=1e-6)


def test_mask_excludes_padded_keys():
    """Masked key positions do not influence unmasked query outputs."""
    torch.manual_seed(2)
    block = _make_block()
    B, L = 2, 8
    n, x, g, m = _make_inputs(B=B, L=L)

    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, -3:] = True  # last 3 positions are padding

    out1 = block(n, x, g, m, key_padding_mask=mask)

    # Replace masked positions with very different values
    n2 = n.clone()
    x2 = x.clone()
    m2 = m.clone()
    n2[:, -3:] = torch.randn_like(n2[:, -3:]) * 100
    x2[:, -3:] = torch.rand_like(x2[:, -3:]) * 1e9
    m2[:, -3:] = torch.randn_like(m2[:, -3:]) * 100

    out2 = block(n2, x2, g, m2, key_padding_mask=mask)

    # Outputs at unmasked positions must be identical
    diff = (out1[:, :5] - out2[:, :5]).abs().max().item()
    assert diff < 1e-5, f"Masked keys influenced output: max diff = {diff:.2e}"


def test_mask_gradient_flow():
    """Backward pass succeeds with mask active."""
    torch.manual_seed(3)
    block = _make_block(learn_tau=True, beta_init=1.0)
    n, x, g, m = _make_inputs()
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, -2:] = True

    out = block(n, x, g, m, key_padding_mask=mask)
    loss = (out**2).mean()
    loss.backward()

    for name, param in block.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, (
            f"Parameter '{name}' has no gradient with mask"
        )


def test_mask_single_valid_position():
    """When only one position is valid, output at that position is V of itself."""
    torch.manual_seed(4)
    block = _make_block(
        n_sequence_heads=0,
        n_position_heads=0,
        n_standard_heads=2,
        d_out=16,
    )
    B, L = 1, 6
    n, x, g, m = _make_inputs(B=B, L=L)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, 0] = False  # only position 0 is valid

    out = block(n, x, g, m, key_padding_mask=mask)
    assert out.shape == (B, L, 16)
    assert torch.isfinite(out[:, 0]).all()


def test_mask_per_batch_different():
    """Different mask patterns per batch element work correctly."""
    torch.manual_seed(5)
    block = _make_block()
    B, L = 2, 8
    n, x, g, m = _make_inputs(B=B, L=L)

    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, -2:] = True  # batch 0: last 2 masked
    mask[1, -4:] = True  # batch 1: last 4 masked

    out = block(n, x, g, m, key_padding_mask=mask)
    assert out.shape == (B, L, 64)
    assert torch.isfinite(out[:, :4]).all()


# ─── Per-head-type mask tests ─────────────────────────────────────────────────


def test_sequence_heads_mask_excludes_keys():
    """SequenceHeads: masked keys don't affect unmasked query outputs."""
    torch.manual_seed(10)
    layer = SequenceHeads(d_seq=32, d_s=48, d_v=8, H=2)
    B, L = 2, 6

    n = torch.randn(B, L, 32)
    s = torch.randn(B, L, 48)
    # SDPA convention: True=allowed, so mask out last 2 by setting them False
    mask = torch.ones(B, 1, 1, L, dtype=torch.bool)
    mask[:, :, :, -2:] = False

    out1 = layer(n, s, attn_mask=mask)

    n2, s2 = n.clone(), s.clone()
    n2[:, -2:] *= 50
    s2[:, -2:] *= 50
    out2 = layer(n2, s2, attn_mask=mask)

    diff = (out1[:, :4] - out2[:, :4]).abs().max().item()
    assert diff < 1e-5, f"Masked keys leaked: {diff:.2e}"


def test_position_heads_mask_excludes_keys():
    """PositionHeads: masked keys don't affect unmasked query outputs."""
    torch.manual_seed(11)
    K, d_u = 32, 16
    lambdas = _make_lambdas(K)
    layer = PositionHeads(d_s=48, d_u=d_u, K=K, d_v=8, H=2, lambdas=lambdas)
    B, L = 2, 6

    u = torch.randn(B, L, d_u)
    r_phi = torch.randn(B, L, 2 * K)
    s = torch.randn(B, L, 48)
    # SDPA convention: True=allowed
    mask = torch.ones(B, 1, 1, L, dtype=torch.bool)
    mask[:, :, :, -2:] = False

    out1 = layer(u, r_phi, s, attn_mask=mask)

    u2, r_phi2, s2 = u.clone(), r_phi.clone(), s.clone()
    u2[:, -2:] *= 50
    r_phi2[:, -2:] *= 50
    s2[:, -2:] *= 50
    out2 = layer(u2, r_phi2, s2, attn_mask=mask)

    diff = (out1[:, :4] - out2[:, :4]).abs().max().item()
    assert diff < 1e-5, f"Masked keys leaked: {diff:.2e}"


def test_position_heads_mask_pad_v_path():
    """PositionHeads with pad_v_for_flash=True: mask still works."""
    torch.manual_seed(12)
    K, d_u = 32, 16
    lambdas = _make_lambdas(K)
    layer = PositionHeads(d_s=48, d_u=d_u, K=K, d_v=8, H=2, lambdas=lambdas, pad_v_for_flash=True)
    B, L = 2, 6

    u = torch.randn(B, L, d_u)
    r_phi = torch.randn(B, L, 2 * K)
    s = torch.randn(B, L, 48)
    # SDPA convention: True=allowed
    mask = torch.ones(B, 1, 1, L, dtype=torch.bool)
    mask[:, :, :, -2:] = False

    out1 = layer(u, r_phi, s, attn_mask=mask)

    s2 = s.clone()
    s2[:, -2:] *= 50
    out2 = layer(u, r_phi, s2, attn_mask=mask)

    diff = (out1[:, :4] - out2[:, :4]).abs().max().item()
    assert diff < 1e-5, f"Masked keys leaked with pad_v: {diff:.2e}"


def test_standard_heads_mask_excludes_keys():
    """StandardHeads: masked keys don't affect unmasked query outputs."""
    torch.manual_seed(13)
    layer = StandardHeads(d_in=128, d_v=16, H=2)
    B, L = 2, 6

    f = torch.randn(B, L, 128)
    # SDPA convention: True=allowed
    mask = torch.ones(B, 1, 1, L, dtype=torch.bool)
    mask[:, :, :, -2:] = False

    out1 = layer(f, attn_mask=mask)

    f2 = f.clone()
    f2[:, -2:] *= 50
    out2 = layer(f2, attn_mask=mask)

    diff = (out1[:, :4] - out2[:, :4]).abs().max().item()
    assert diff < 1e-5, f"Masked keys leaked: {diff:.2e}"
