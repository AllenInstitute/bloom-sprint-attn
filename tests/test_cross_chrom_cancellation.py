"""Numerical companion to plot_cross_chrom_cancellation.py.

The figures (`docs/figures/cross_chrom_kernel_3x6.png`) make the
cancellation visible. This test makes the load-bearing claims assertable.
Each test targets a *specific* property of the kernel structure -- not a
global mean-ratio summary, which would dilute the signal because most
pair-Δ values in a log-spaced grid place the kernel near zero anyway.

The four claims:

  1. Full design: same-coord pairs across different chromosomes have
     near-zero kernel value (sign-cancellation works).

  2. Without sign gating, those same pairs hit the head's peak kernel
     value -- chr1@x and chr2@x become indistinguishable from
     chr1@x and chr1@x for the wider heads.

  3. β controls the on-chromosome floor only. Removing β leaves the
     cross-chromosome block unchanged.

  4. Increasing K reduces the cross-chromosome noise floor (CLT,
     variance ~ 1/K).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make scripts/ importable so we can reuse build_pair, build_inputs, kernels.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_cross_chrom_cancellation import (  # noqa: E402
    N_PER_CHROM,
    build_inputs,
    build_pair,
    kernels,
)


def _full_kernel(K: int = 112, beta_init: float = 0.5):
    """Build kernel for the full design at the requested K and beta_init."""
    torch.manual_seed(0)
    enc, sp = build_pair(K=K, beta_init=beta_init)
    u, r_phi, phi = build_inputs(enc)
    return kernels(sp, u, r_phi, phi, "full"), sp


def _same_coord_cross_chrom_values(K_hij: torch.Tensor) -> torch.Tensor:
    """Return K_h at (i, i+N_PER_CHROM) for every head, every i."""
    n = N_PER_CHROM
    return K_hij[:, torch.arange(n), torch.arange(n) + n]  # [H, n]


def _cross_chrom_block(K_hij: torch.Tensor) -> torch.Tensor:
    """Return the upper-right cross-chromosome block for every head."""
    n = N_PER_CHROM
    return K_hij[:, :n, n:]  # [H, n, n]


def test_full_design_same_coord_cross_chrom_is_near_zero():
    """Spec section 7: with sign gating, K(chr1@x, chr2@x) ≈ 0 for every head."""
    K_hij, _ = _full_kernel()
    twins = _same_coord_cross_chrom_values(K_hij)  # [H, n]
    # With K=112, the noise floor is O(1/√K) ≈ 0.09 per dim; the per-head
    # log-Gaussian weighting suppresses this further. Bound at 0.2.
    for h in range(twins.shape[0]):
        assert twins[h].abs().max().item() < 0.2, (
            f"head {h}: max |K(chr1@x, chr2@x)| = {twins[h].abs().max().item():.4f} "
            "should be near zero"
        )


def test_no_sign_gating_leaks_coordinate_into_cross_chrom_block():
    """Within the cross-chromosome block, same-coordinate pairs should be
    statistically indistinguishable from far-coordinate pairs under the full
    design (cross-chrom kernel is pure noise around β² · u_chr1·u_chr2).
    Without sign gating, same-coordinate cross-chrom pairs spike above
    far-coordinate cross-chrom pairs -- that's exactly the "diagonal stripe"
    visible in the middle row of cross_chrom_kernel_3x6.png for the wider heads.

    Test on the wider heads (L >= 100kb), which have non-negligible w_h on
    enough wavelengths that the same-coord peak rises clearly above the
    far-coord baseline. Narrow heads cover only a few wavelengths so the
    same-coord-vs-far-coord differential is small in absolute terms.
    """
    torch.manual_seed(0)
    enc, sp = build_pair()
    u, r_phi, phi = build_inputs(enc)
    K_full = kernels(sp, u, r_phi, phi, "full")
    K_nosign = kernels(sp, u, r_phi, phi, "no_sign")
    n = N_PER_CHROM
    eye_off = ~torch.eye(n, dtype=torch.bool)
    diag = torch.eye(n, dtype=torch.bool)

    for h in range(3, 6):  # L = 100kb, 1Mb, 10Mb -- see docstring.
        cross_full = K_full[h, :n, n:]
        cross_nosign = K_nosign[h, :n, n:]

        full_diff = (cross_full[diag].mean() - cross_full[eye_off].mean()).item()
        nosign_diff = (cross_nosign[diag].mean() - cross_nosign[eye_off].mean()).item()

        # Full: same-coord twin mean is statistically equal to far-coord mean.
        # (Both are samples from the same N(β²·u_chr1·u_chr2, σ²) distribution.)
        assert abs(full_diff) < 0.05, (
            f"head {h} full: same-coord vs far-coord cross-chrom differ by {full_diff:.4f}; "
            "should be ~0 (cross-chrom should not 'see' coordinate)"
        )
        # No-sign: same-coord twin spikes well above far-coord baseline.
        assert nosign_diff > 0.1, (
            f"head {h} no_sign: same-coord vs far-coord cross-chrom differ by {nosign_diff:.4f}; "
            "should be » 0 (sign-gating is the only thing hiding this leak)"
        )


def test_beta_shifts_cross_chrom_by_a_constant_not_a_pattern():
    """The chromosome baseline β² · u_g adds a *constant* to the cross-chrom block
    (because u_chr1 · u_chr2 is a single realized inner product), shifting its
    mean by exactly β² · (u_chr1 · u_chr2). Crucially, β does not change the
    *texture* of the cross-chrom block -- the variance over the block is
    invariant in β. The figure ('cross_chrom_beta_sweep.png') shows this:
    cross-chrom blocks look visually identical across β ∈ {0, 0.5, 1.0, 1.5}.
    """
    torch.manual_seed(0)
    enc, sp = build_pair()
    u, r_phi, phi = build_inputs(enc)
    K_full = kernels(sp, u, r_phi, phi, "full")
    K_nobeta = kernels(sp, u, r_phi, phi, "no_beta")
    n = N_PER_CHROM

    for h in range(K_full.shape[0]):
        cross_full = K_full[h, :n, n:]
        cross_nobeta = K_nobeta[h, :n, n:]

        # Texture invariance: cross-chrom variance is unchanged by β.
        var_full = cross_full.var(unbiased=False).item()
        var_nobeta = cross_nobeta.var(unbiased=False).item()
        # var should match to machine precision (β·u is added uniformly).
        assert abs(var_full - var_nobeta) < 1e-6, (
            f"head {h}: cross-chrom var(full) = {var_full:.6f} vs "
            f"var(no_beta) = {var_nobeta:.6f} -- β should add only a constant"
        )

        # The constant added IS β² · u_chr1·u_chr2, the same number for every
        # entry of the cross-chrom block. Check it's the same shift everywhere.
        delta = cross_full - cross_nobeta  # [n, n]
        shift_max = delta.max().item()
        shift_min = delta.min().item()
        assert (shift_max - shift_min) < 1e-6, (
            f"head {h}: β shift varies across cross-chrom block "
            f"(range {shift_max - shift_min:.6f}); should be a single constant"
        )

    # Also: the on-chrom block sees a shift of β² · (u_g · u_g) = β² = 0.25,
    # which is much larger than the cross-chrom shift β² · u_chr1·u_chr2 in
    # absolute value (|u_chr1·u_chr2| < 1 by Cauchy-Schwarz).
    on_shift = (K_full[0, :n, :n] - K_nobeta[0, :n, :n]).mean().item()
    cross_shift = (K_full[0, :n, n:] - K_nobeta[0, :n, n:]).mean().item()
    assert on_shift > abs(cross_shift), (
        f"on-chrom shift {on_shift:.4f} should exceed |cross-chrom shift| "
        f"{abs(cross_shift):.4f} (β²·1 > β²·|u_chr1·u_chr2|)"
    )


def test_K_reduces_cross_chrom_noise_floor():
    """Cross-chrom noise variance should decrease with K (CLT, variance ~ 1/K)."""
    # Measure cross-chrom std for each head at multiple K values.
    K_values = [16, 32, 112, 256]
    stds = []  # [K_idx][H]
    for K_val in K_values:
        K_hij, _ = _full_kernel(K=K_val)
        cross = _cross_chrom_block(K_hij)  # [H, n, n]
        # Remove the (near-zero) mean: std of cross-chrom block per head.
        stds.append(cross.std(dim=(1, 2)))  # [H]

    # For every head, std at K=256 should be smaller than std at K=16.
    H = stds[0].shape[0]
    for h in range(H):
        s_lo = stds[0][h].item()  # K=16
        s_hi = stds[-1][h].item()  # K=256
        assert s_hi < s_lo, f"head {h}: std(K=256) = {s_hi:.4f} should be < std(K=16) = {s_lo:.4f}"
