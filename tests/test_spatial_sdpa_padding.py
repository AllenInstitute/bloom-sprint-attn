"""Tests for the pad_v_for_flash option on SpatialAttentionSDPA."""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from bloom_arch.attention import SpatialAttentionSDPA
from bloom_arch.posenc import ChromLocalPosEncoder

SPECIES = "musmus"
ASSEMBLY = "mm10"


def _make(layer_kwargs: dict | None = None, seed: int = 0):
    torch.manual_seed(seed)
    encoder = ChromLocalPosEncoder(d_u=32, K=64, lambda_min=10.0, lambda_max=1.0e8)
    layer = SpatialAttentionSDPA(
        d_s=32,
        d_v=8,
        H=4,
        K=64,
        lambdas=encoder.fourier.lambdas,
        target_scales=[1e2, 1e3, 1e4, 1e5],
        **(layer_kwargs or {}),
    )
    return encoder, layer


# ---------------------------------------------------------------------------
# Parity: padding does not change the output.
# ---------------------------------------------------------------------------
def test_v_padding_does_not_change_output_fp32():
    encoder, layer_plain = _make(seed=11)
    _, layer_padded = _make({"pad_v_for_flash": True}, seed=11)
    # Same seed reproduces the same parameter init.
    layer_padded.load_state_dict(layer_plain.state_dict())

    B, N = 2, 9
    coords = torch.rand(B, N) * 1.0e7
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * N for _ in range(B)]
    s = torch.randn(B, N, layer_plain.d_s)
    pos = encoder(coords, keys)

    out_plain = layer_plain(pos["u"], pos["r_phi"], s)
    out_pad = layer_padded(pos["u"], pos["r_phi"], s)
    diff = (out_plain - out_pad).abs().max().item()
    assert diff < 1e-6, f"padding changed output by {diff:.2e}"


def test_v_padding_state_dict_compatible_with_unpadded():
    """state_dict should round-trip regardless of the pad flag."""
    _, layer_plain = _make(seed=2)
    _, layer_padded = _make({"pad_v_for_flash": True}, seed=99)  # different seed
    layer_padded.load_state_dict(layer_plain.state_dict())
    # All learnables now equal.
    for name, p_plain in layer_plain.named_parameters():
        p_pad = dict(layer_padded.named_parameters())[name]
        assert torch.equal(p_plain, p_pad), f"mismatch on {name}"


# ---------------------------------------------------------------------------
# Flash backend now dispatches on CUDA + bf16.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_padded_flash_dispatch_bf16_cuda():
    """With pad_v_for_flash=True, SDPA must accept the FLASH backend."""
    encoder, layer = _make({"pad_v_for_flash": True}, seed=0)
    encoder = encoder.cuda()
    layer = layer.cuda().to(torch.bfloat16)

    B, N = 2, 16
    coords = (torch.rand(B, N) * 1.0e7).cuda()
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * N for _ in range(B)]
    pos = encoder(coords, keys)
    u_ = pos["u"].to(torch.bfloat16)
    r_ = pos["r_phi"].to(torch.bfloat16)
    s = torch.randn(B, N, layer.d_s, device="cuda", dtype=torch.bfloat16)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = layer(u_, r_, s)
    assert out.shape == (B, N, layer.H, layer.d_v)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_padded_flash_matches_efficient_bf16_cuda():
    """The flash and efficient backends should agree numerically (bf16 tol)."""
    encoder, layer = _make({"pad_v_for_flash": True}, seed=3)
    encoder = encoder.cuda()
    layer = layer.cuda().to(torch.bfloat16)

    B, N = 2, 12
    coords = (torch.rand(B, N) * 1.0e6).cuda()
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * N for _ in range(B)]
    pos = encoder(coords, keys)
    u_ = pos["u"].to(torch.bfloat16)
    r_ = pos["r_phi"].to(torch.bfloat16)
    s = torch.randn(B, N, layer.d_s, device="cuda", dtype=torch.bfloat16)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out_flash = layer(u_, r_, s)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out_eff = layer(u_, r_, s)

    diff = (out_flash.float() - out_eff.float()).abs().max().item()
    assert diff < 5e-2, f"flash vs efficient diverge by {diff:.2e}"
