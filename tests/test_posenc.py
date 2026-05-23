"""Tests for the positional encoding stack (spec §11.1–§11.3, §11.6, plus guards)."""

from __future__ import annotations

import math

import pytest
import torch

from bloom_arch.posenc import ChromCodeBook, ChromLocalPosEncoder, FourierCoordEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SPECIES = "musmus"
ASSEMBLY = "mm10"


def _make_encoder(d_u: int = 32, K: int = 112) -> ChromLocalPosEncoder:
    return ChromLocalPosEncoder(d_u=d_u, K=K, lambda_min=10.0, lambda_max=1.0e8)


# ---------------------------------------------------------------------------
# T1 — determinism (spec §11.1)
# ---------------------------------------------------------------------------
def test_chrom_codes_deterministic_within_book():
    book = ChromCodeBook(d_u=32, d_phi=224)
    key = (SPECIES, ASSEMBLY, "chr1")
    u1, r1 = book._generate(key)
    u2, r2 = book._generate(key)
    assert torch.equal(u1, u2)
    assert torch.equal(r1, r2)


def test_chrom_codes_deterministic_across_books():
    book_a = ChromCodeBook(d_u=32, d_phi=224, salt="posenc_v1")
    book_b = ChromCodeBook(d_u=32, d_phi=224, salt="posenc_v1")
    key = (SPECIES, ASSEMBLY, "chr17")
    u_a, r_a = book_a._generate(key)
    u_b, r_b = book_b._generate(key)
    assert torch.equal(u_a, u_b)
    assert torch.equal(r_a, r_b)


def test_chrom_codes_salt_changes_output():
    book_a = ChromCodeBook(d_u=32, d_phi=224, salt="posenc_v1")
    book_b = ChromCodeBook(d_u=32, d_phi=224, salt="posenc_v2")
    key = (SPECIES, ASSEMBLY, "chr1")
    u_a, _ = book_a._generate(key)
    u_b, _ = book_b._generate(key)
    assert not torch.equal(u_a, u_b)


def test_chrom_codes_r_is_pm_one_and_u_is_unit_norm():
    book = ChromCodeBook(d_u=32, d_phi=224)
    u, r = book._generate((SPECIES, ASSEMBLY, "chrX"))
    assert r.shape == (224,)
    assert torch.all((r == 1.0) | (r == -1.0))
    assert u.shape == (32,)
    assert math.isclose(u.norm().item(), 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# T2 — Fourier translation invariance (spec §6.1)
# ---------------------------------------------------------------------------
def test_fourier_dot_depends_only_on_distance():
    """φ(x_i)·φ(x_j) is a function of x_i − x_j only (spec §6.1).

    Run in float64 — at K=112 and λ_min=10 bp, the highest-frequency argument
    is 2π·x/10, which is ~1.3×10⁸ radians for x ~ 2×10⁸. float32 sin/cos lose
    >3 decimal digits at that magnitude; float64 has the precision to express
    the actual mathematical claim within ~1e-7. Float32 behavior is exercised
    by the more lenient downstream tests (sign cancellation, decorrelation).
    """
    K = 112
    fcoder = FourierCoordEncoder(K=K, lambda_min=10.0, lambda_max=1.0e8)

    pairs = [
        (1_000_000.0, 2_000_000.0),
        (50_000_000.0, 51_000_000.0),
        (123.0, 1_000_123.0),
        (200_000_000.0, 199_000_000.0),
    ]
    dots = []
    for x_i, x_j in pairs:
        x = torch.tensor([[x_i, x_j]], dtype=torch.float64)
        phi = fcoder(x)[0]  # [2, 2K]
        dots.append((phi[0] * phi[1]).sum().item())
    spread = max(dots) - min(dots)
    assert spread < 1e-6, f"dot products varied by {spread:.2e} across translations: {dots}"


def test_fourier_dot_depends_only_on_distance_float32_short_range():
    """At biologically interesting short ranges (kb-scale), float32 is enough.

    Same property as above but at coordinates and Δ values that don't push
    against float32's resolution limit at the 10 bp wavelength.
    """
    K = 64
    fcoder = FourierCoordEncoder(K=K, lambda_min=100.0, lambda_max=1.0e6)
    delta = 5_000.0
    bases = [10_000.0, 100_000.0, 500_000.0, 1_000_000.0]
    dots = []
    for b in bases:
        x = torch.tensor([[b, b + delta]], dtype=torch.float32)
        phi = fcoder(x)[0]
        dots.append((phi[0] * phi[1]).sum().item())
    spread = max(dots) - min(dots)
    assert spread < 5e-4, f"dot products varied by {spread:.2e}: {dots}"


def test_fourier_dot_is_one_at_zero_distance():
    K = 64
    fcoder = FourierCoordEncoder(K=K, lambda_min=10.0, lambda_max=1.0e8)
    x = torch.tensor([[12345.0, 12345.0]])
    phi = fcoder(x)[0]
    dot = (phi[0] * phi[1]).sum().item()
    assert math.isclose(dot, 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# T3 — same-chromosome sign cancellation (spec §11.2)
# ---------------------------------------------------------------------------
def test_same_chrom_sign_cancellation():
    encoder = _make_encoder()
    fcoder = encoder.fourier

    coords = torch.tensor([[1_000_000.0, 1_010_000.0]])
    chrom_keys = [[(SPECIES, ASSEMBLY, "chr3"), (SPECIES, ASSEMBLY, "chr3")]]

    out = encoder(coords, chrom_keys)
    r_phi = out["r_phi"][0]  # [2, 2K]
    phi = fcoder(coords)[0]  # [2, 2K]

    # (r ⊙ φ_i) · (r ⊙ φ_j) == φ_i · φ_j, because r is ±1 and shared across i, j on same chrom.
    lhs = (r_phi[0] * r_phi[1]).sum().item()
    rhs = (phi[0] * phi[1]).sum().item()
    assert math.isclose(lhs, rhs, abs_tol=1e-5), f"lhs={lhs} rhs={rhs}"


# ---------------------------------------------------------------------------
# T4 — cross-chromosome decorrelation (spec §11.3)
# ---------------------------------------------------------------------------
def test_cross_chrom_decorrelation_mean_near_zero():
    encoder = _make_encoder(d_u=32, K=112)
    # 200 random pairs, each with one feature on chr_a and one on chr_b.
    torch.manual_seed(0)
    N_PAIRS = 200
    coords_a = torch.rand(N_PAIRS) * 1.0e8
    coords_b = torch.rand(N_PAIRS) * 1.0e8
    coords = torch.stack([coords_a, coords_b], dim=-1)  # [N_PAIRS, 2]
    chrom_keys = [
        [(SPECIES, ASSEMBLY, "chrA"), (SPECIES, ASSEMBLY, "chrB")] for _ in range(N_PAIRS)
    ]

    out = encoder(coords, chrom_keys)
    r_phi = out["r_phi"]  # [N_PAIRS, 2, 2K]
    dots = (r_phi[:, 0] * r_phi[:, 1]).sum(-1)  # [N_PAIRS]
    mean = dots.mean().item()
    std = dots.std().item()

    # Variance per pair ≈ 2/K (each of 2K independent products with var ≈ (1/K)² each summed)
    # so std should be small for K=112.
    assert abs(mean) < 0.05, f"cross-chrom mean dot product = {mean:.4f}, expected near 0"
    assert std < 0.25, f"cross-chrom std = {std:.4f}, expected small for K=112"


def test_cross_chrom_decorrelation_variance_decreases_with_K():
    # The cross-chrom dot product variance should shrink as K grows.
    torch.manual_seed(1)

    def cross_chrom_std(K: int) -> float:
        enc = ChromLocalPosEncoder(d_u=32, K=K, lambda_min=10.0, lambda_max=1.0e8)
        N = 400
        coords = torch.rand(N, 2) * 1.0e8
        chrom_keys = [[(SPECIES, ASSEMBLY, "chrA"), (SPECIES, ASSEMBLY, "chrB")] for _ in range(N)]
        out = enc(coords, chrom_keys)
        r_phi = out["r_phi"]
        dots = (r_phi[:, 0] * r_phi[:, 1]).sum(-1)
        return dots.std().item()

    s_small = cross_chrom_std(16)
    s_large = cross_chrom_std(256)
    assert s_large < s_small, f"std(K=256)={s_large} not < std(K=16)={s_small}"


# ---------------------------------------------------------------------------
# T7 — new chromosome / species (spec §11.6)
# ---------------------------------------------------------------------------
def test_new_chromosome_works_without_state():
    encoder = _make_encoder()

    # "Seen" set
    seen_keys = [(SPECIES, ASSEMBLY, c) for c in ("chr1", "chr2", "chr3")]
    coords_seen = torch.tensor([[100.0, 200.0, 300.0]])
    encoder(coords_seen, [seen_keys])

    # A novel species/assembly/chromosome triple.
    novel_key = ("homsap", "hg38", "chr22")
    novel_pair = [novel_key, novel_key]
    coords_novel = torch.tensor([[1_000_000.0, 1_010_000.0]])
    out = encoder(coords_novel, [novel_pair])
    assert out["u"].shape == (1, 2, encoder.d_u)
    assert out["r_phi"].shape == (1, 2, encoder.d_phi)

    # Same-chrom property still holds for the novel key.
    phi = encoder.fourier(coords_novel)[0]
    r_phi = out["r_phi"][0]
    same_dot = (r_phi[0] * r_phi[1]).sum().item()
    expected = (phi[0] * phi[1]).sum().item()
    assert math.isclose(same_dot, expected, abs_tol=1e-5)

    # Decorrelated from one of the seen chromosomes.
    seen_pair = [(SPECIES, ASSEMBLY, "chr1"), novel_key]
    coords_pair = torch.tensor([[1_000_000.0, 1_000_000.0]])
    out_cross = encoder(coords_pair, [seen_pair])
    cross_dot = (out_cross["r_phi"][0, 0] * out_cross["r_phi"][0, 1]).sum().item()
    assert abs(cross_dot) < 0.5  # much smaller than the same-chrom value (≈1.0 at Δ=0)


# ---------------------------------------------------------------------------
# T8 — encoder has no learned state (spec §2.3)
# ---------------------------------------------------------------------------
def test_encoder_has_no_parameters():
    encoder = _make_encoder()
    params = list(encoder.parameters())
    assert params == [], f"expected no parameters, found {len(params)}: {[p.shape for p in params]}"

    # Buffers are allowed (the wavelength schedule is a buffer); they must not be learnable.
    for name, buf in encoder.named_buffers():
        assert not buf.requires_grad, f"buffer {name} should not require grad"


def test_codebook_has_no_parameters():
    book = ChromCodeBook(d_u=32, d_phi=224)
    assert list(book.parameters()) == []
    assert list(book.buffers()) == []


# ---------------------------------------------------------------------------
# Device-move smoke test (GPU only)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_encoder_moves_to_cuda():
    encoder = _make_encoder().cuda()
    coords = torch.tensor([[100.0, 200.0]], device="cuda")
    keys = [[(SPECIES, ASSEMBLY, "chr1"), (SPECIES, ASSEMBLY, "chr1")]]
    out = encoder(coords, keys)
    assert out["u"].device.type == "cuda"
    assert out["r_phi"].device.type == "cuda"
