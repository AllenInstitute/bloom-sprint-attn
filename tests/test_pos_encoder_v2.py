"""Tests for PosEncoder (integer-indexed, GPU-first)."""

import torch

from bloom_arch.posenc.encoder import PosEncoder


def test_encoder_output_shapes():
    enc = PosEncoder(max_G=30, d_u=32, K=64, lambda_min=10.0, lambda_max=1e8, seed=42)
    x = torch.rand(2, 8) * 1e7
    g = torch.randint(0, 30, (2, 8))
    pos = enc(x, g)
    assert pos["u"].shape == (2, 8, 32)
    assert pos["r_phi"].shape == (2, 8, 128)  # 2*K=128


def test_encoder_no_parameters():
    enc = PosEncoder(max_G=10, d_u=16, K=32, seed=0)
    assert list(enc.parameters()) == []


def test_encoder_same_chrom_sign_cancellation():
    """r_phi_i . r_phi_j == phi_i . phi_j when same g (sign cancels)."""
    enc = PosEncoder(max_G=10, d_u=16, K=64, lambda_min=10.0, lambda_max=1e8, seed=7)
    # Two features on the same chromosome
    x = torch.tensor([[1000.0, 2000.0]])  # [1, 2]
    g = torch.tensor([[5, 5]])  # same g
    pos = enc(x, g)
    r_phi = pos["r_phi"]  # [1, 2, 128]
    dot_r_phi = (r_phi[0, 0] * r_phi[0, 1]).sum()
    # Compute raw phi dot product for comparison
    phi = enc.fourier(x)  # [1, 2, 128]
    dot_phi = (phi[0, 0] * phi[0, 1]).sum()
    assert torch.allclose(dot_r_phi, dot_phi, atol=1e-5)


def test_encoder_cross_chrom_decorrelation():
    """Mean cross-chrom r_phi dot product is near zero."""
    enc = PosEncoder(max_G=100, d_u=32, K=112, lambda_min=10.0, lambda_max=1e8, seed=42)
    # Many features on different chromosomes at similar coordinates
    B, L = 1, 50
    x = torch.rand(B, L) * 1e6
    g = torch.arange(L).unsqueeze(0)  # all different
    pos = enc(x, g)
    r_phi = pos["r_phi"][0]  # [L, 2K]
    # Compute pairwise dots
    dots = r_phi @ r_phi.T  # [L, L]
    mask = ~torch.eye(L, dtype=torch.bool)
    off_diag = dots[mask]
    # Expected: mean ~ 0, std ~ 1/sqrt(K)
    assert off_diag.abs().mean() < 0.15


def test_encoder_deterministic_across_instances():
    enc1 = PosEncoder(max_G=10, d_u=16, K=32, seed=7)
    enc2 = PosEncoder(max_G=10, d_u=16, K=32, seed=7)
    x = torch.rand(2, 4) * 1e6
    g = torch.randint(0, 10, (2, 4))
    pos1 = enc1(x, g)
    pos2 = enc2(x, g)
    assert torch.equal(pos1["u"], pos2["u"])
    assert torch.equal(pos1["r_phi"], pos2["r_phi"])
