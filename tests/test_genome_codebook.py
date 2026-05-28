"""Tests for GenomeCodebook and PosEncoder (integer-indexed)."""

import torch

from bloom_arch.posenc.genome_codebook import GenomeCodebook


def test_codebook_deterministic():
    """Same max_G, d_u, d_phi, seed produces identical buffers."""
    cb1 = GenomeCodebook(max_G=100, d_u=32, d_phi=224, seed=42)
    cb2 = GenomeCodebook(max_G=100, d_u=32, d_phi=224, seed=42)
    assert torch.equal(cb1.u, cb2.u)
    assert torch.equal(cb1.r, cb2.r)


def test_codebook_r_is_pm_one():
    """All entries of r are exactly +1 or -1."""
    cb = GenomeCodebook(max_G=50, d_u=16, d_phi=128, seed=7)
    assert ((cb.r == 1.0) | (cb.r == -1.0)).all()


def test_codebook_u_is_unit_norm():
    """Each row of u has L2 norm 1.0."""
    cb = GenomeCodebook(max_G=50, d_u=32, d_phi=128, seed=7)
    norms = cb.u.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(50), atol=1e-6)


def test_codebook_different_seeds_differ():
    cb1 = GenomeCodebook(max_G=10, d_u=16, d_phi=64, seed=1)
    cb2 = GenomeCodebook(max_G=10, d_u=16, d_phi=64, seed=2)
    assert not torch.equal(cb1.u, cb2.u)
    assert not torch.equal(cb1.r, cb2.r)


def test_codebook_forward_indexing():
    """forward(g) returns correct rows from buffers."""
    cb = GenomeCodebook(max_G=20, d_u=8, d_phi=16, seed=0)
    g = torch.tensor([[0, 5, 19], [3, 3, 10]])  # [2, 3]
    u, r = cb(g)
    assert u.shape == (2, 3, 8)
    assert r.shape == (2, 3, 16)
    # Check specific entries
    assert torch.equal(u[0, 0], cb.u[0])
    assert torch.equal(u[0, 1], cb.u[5])
    assert torch.equal(u[1, 2], cb.u[10])
    assert torch.equal(r[0, 2], cb.r[19])


def test_codebook_no_parameters():
    cb = GenomeCodebook(max_G=10, d_u=8, d_phi=16, seed=0)
    assert list(cb.parameters()) == []


def test_codebook_approximate_orthogonality():
    """For large d_u, u_i . u_j for i != j is close to 0."""
    cb = GenomeCodebook(max_G=50, d_u=64, d_phi=128, seed=42)
    # Compute all pairwise dot products
    dots = cb.u @ cb.u.T  # [50, 50]
    # Diagonal should be 1
    assert torch.allclose(dots.diag(), torch.ones(50), atol=1e-5)
    # Off-diagonal should be small (expected std ~ 1/sqrt(d_u) = 1/8)
    mask = ~torch.eye(50, dtype=torch.bool)
    off_diag = dots[mask]
    assert off_diag.abs().mean() < 0.2  # generous bound
