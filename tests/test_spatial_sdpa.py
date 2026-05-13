"""Tests for SpatialAttentionSDPA - parity with math version, bf16 smoke, autograd."""

from __future__ import annotations

import pytest
import torch

from bloom_arch.attention import SpatialAttentionSDPA
from bloom_arch.posenc import ChromLocalPosEncoder

SPECIES = "musmus"
ASSEMBLY = "mm10"


def _make_layer(d_s: int = 32, d_v: int = 8, H: int = 4, K: int = 64, seed: int = 0):
    torch.manual_seed(seed)
    encoder = ChromLocalPosEncoder(d_u=32, K=K, lambda_min=10.0, lambda_max=1.0e8)
    layer = SpatialAttentionSDPA(
        d_s=d_s, d_v=d_v, H=H, K=K, lambdas=encoder.fourier.lambdas,
        target_scales=[1e2, 1e3, 1e4, 1e5][:H],
    )
    return encoder, layer


# ---------------------------------------------------------------------------
# Parity: SDPA forward must match the explicit math forward bitwise-close.
# ---------------------------------------------------------------------------
def test_sdpa_matches_math_version_fp32_same_chrom():
    encoder, layer = _make_layer(seed=42)
    B, N, d_s = 2, 6, 32
    coords = torch.rand(B, N) * 1.0e7
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * N for _ in range(B)]
    s = torch.randn(B, N, d_s)
    pos = encoder(coords, keys)

    out_sdpa = layer(pos["u"], pos["r_phi"], s)
    out_math = layer.forward_explicit(pos["u"], pos["r_phi"], s)

    assert out_sdpa.shape == out_math.shape == (B, N, layer.H, layer.d_v)
    diff = (out_sdpa - out_math).abs().max().item()
    assert diff < 1e-5, f"SDPA vs explicit math diverge by {diff:.2e}"


def test_sdpa_matches_math_version_fp32_mixed_chroms():
    encoder, layer = _make_layer(seed=7)
    B, N, d_s = 1, 8, 32
    coords = torch.rand(B, N) * 1.0e7
    keys = [
        [(SPECIES, ASSEMBLY, "chr1" if n % 2 == 0 else "chr2") for n in range(N)]
    ]
    s = torch.randn(B, N, d_s)
    pos = encoder(coords, keys)
    out_sdpa = layer(pos["u"], pos["r_phi"], s)
    out_math = layer.forward_explicit(pos["u"], pos["r_phi"], s)
    diff = (out_sdpa - out_math).abs().max().item()
    assert diff < 1e-5, f"SDPA vs explicit math diverge by {diff:.2e}"


# ---------------------------------------------------------------------------
# Autograd: gradients reach every learnable parameter through the SDPA path.
# ---------------------------------------------------------------------------
def test_sdpa_autograd_reaches_all_parameters():
    encoder, layer = _make_layer(seed=1)
    B, N = 2, 5
    coords = torch.rand(B, N) * 1.0e7
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * N for _ in range(B)]
    s = torch.randn(B, N, 32, requires_grad=True)
    pos = encoder(coords, keys)
    out = layer(pos["u"], pos["r_phi"], s)
    loss = (out ** 2).mean()
    loss.backward()
    assert layer.log_w.grad is not None and layer.log_w.grad.abs().sum() > 0
    assert layer.beta.grad is not None and layer.beta.grad.abs().sum() > 0
    assert layer.log_tau.grad is not None and layer.log_tau.grad.abs().sum() > 0
    assert layer.W_V.weight.grad is not None and layer.W_V.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# bf16 smoke: shapes, dtype, no NaNs (CUDA only - bf16 SDPA needs the device).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sdpa_bf16_smoke_cuda():
    encoder, layer = _make_layer(seed=0)
    layer = layer.cuda().to(torch.bfloat16)
    B, N = 2, 16
    coords = (torch.rand(B, N) * 1.0e7).cuda()
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * N for _ in range(B)]
    encoder = encoder.cuda()
    pos = encoder(coords, keys)
    u_bf = pos["u"].to(torch.bfloat16)
    r_bf = pos["r_phi"].to(torch.bfloat16)
    s_bf = torch.randn(B, N, layer.d_s, device="cuda", dtype=torch.bfloat16)
    out = layer(u_bf, r_bf, s_bf)
    assert out.shape == (B, N, layer.H, layer.d_v)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sdpa_bf16_close_to_fp32_cuda():
    """bf16 output should approximate fp32 SDPA output within bf16 precision.

    bf16 carries ~3 decimal digits of mantissa; absolute tolerance of 5e-2 on
    small attention outputs (mean(s^2) ~ 1) is consistent with typical bf16
    rounding through one matmul + softmax + matmul.
    """
    encoder, layer_fp32 = _make_layer(seed=2)
    layer_bf16 = SpatialAttentionSDPA(
        d_s=layer_fp32.d_s, d_v=layer_fp32.d_v, H=layer_fp32.H, K=layer_fp32.K,
        lambdas=layer_fp32.lambdas,
    )
    layer_bf16.load_state_dict(layer_fp32.state_dict())
    layer_fp32 = layer_fp32.cuda()
    layer_bf16 = layer_bf16.cuda().to(torch.bfloat16)
    encoder = encoder.cuda()

    B, N = 2, 12
    coords = (torch.rand(B, N) * 1.0e6).cuda()
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * N for _ in range(B)]
    s = torch.randn(B, N, layer_fp32.d_s, device="cuda")
    pos = encoder(coords, keys)
    out_fp32 = layer_fp32(pos["u"], pos["r_phi"], s)
    out_bf16 = layer_bf16(
        pos["u"].to(torch.bfloat16),
        pos["r_phi"].to(torch.bfloat16),
        s.to(torch.bfloat16),
    )
    diff = (out_fp32 - out_bf16.float()).abs().max().item()
    assert diff < 5e-2, f"bf16 vs fp32 diverge by {diff:.2e}"
