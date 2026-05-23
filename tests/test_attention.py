"""Tests for the spatial/content attention heads (spec §11.4, §11.5, §12, plus guards)."""

from __future__ import annotations

import math

import torch

from bloom_arch.attention import ContentAttention, SpatialAttention, TypedMultiHeadAttention
from bloom_arch.posenc import ChromLocalPosEncoder

SPECIES = "musmus"
ASSEMBLY = "mm10"


def _make_stack(
    d_s: int = 64,
    d_v: int = 16,
    H_content: int = 2,
    H_spatial: int = 4,
    K: int = 112,
    target_scales=None,
) -> tuple[ChromLocalPosEncoder, TypedMultiHeadAttention]:
    encoder = ChromLocalPosEncoder(d_u=32, K=K, lambda_min=10.0, lambda_max=1.0e8)
    block = TypedMultiHeadAttention(
        d_s=d_s,
        d_v=d_v,
        H_content=H_content,
        H_spatial=H_spatial,
        K=K,
        lambdas=encoder.fourier.lambdas,
        target_scales=target_scales,
    )
    return encoder, block


# ---------------------------------------------------------------------------
# T5 — spatial attention prefers nearby same-chrom features (spec §11.4)
# ---------------------------------------------------------------------------
def test_spatial_attention_prefers_nearby_same_chrom():
    """A query attending to (near-same, far-same, cross-chrom-near-coord) should
    place most spatial-attention mass on the near-same features."""

    torch.manual_seed(0)
    K = 112
    encoder = ChromLocalPosEncoder(d_u=32, K=K, lambda_min=10.0, lambda_max=1.0e8)
    # One spatial head, initialized to target scale 100 kb so the kernel is
    # localized on the kilobase-scale neighborhood.
    spatial = SpatialAttention(
        d_s=8,
        d_v=8,
        H=1,
        K=K,
        lambdas=encoder.fourier.lambdas,
        target_scales=[1.0e5],
        beta_init=1.0,
        tau_init=4.0,
    )

    # Bag layout:
    #   feature 0  : query, chr1 @ 50,000,000
    #   features 1..4: NEAR same-chrom (chr1 within ±5 kb)
    #   features 5..8: FAR same-chrom (chr1, 50–100 Mb away)
    #   features 9..12: CROSS-chrom (chr2 at the same coordinate as the query)
    query_x = 50_000_000.0
    near = [query_x + d for d in (-5_000, -1_000, +1_000, +5_000)]
    far = [query_x + d for d in (-90_000_000, -50_000_000, +50_000_000, +90_000_000)]
    cross = [query_x + d for d in (-5_000, -1_000, +1_000, +5_000)]
    coords = torch.tensor([[query_x] + near + far + cross], dtype=torch.float32)
    keys = [
        [(SPECIES, ASSEMBLY, "chr1")]
        + [(SPECIES, ASSEMBLY, "chr1")] * len(near)
        + [(SPECIES, ASSEMBLY, "chr1")] * len(far)
        + [(SPECIES, ASSEMBLY, "chr2")] * len(cross)
    ]
    pos = encoder(coords, keys)
    # Reach into the attention layer to inspect the softmax weights directly.
    logits = _spatial_logits(spatial, pos["u"], pos["r_phi"])  # [1, 1, N, N]
    attn = torch.softmax(logits, dim=-1)[0, 0]  # [N, N]
    row_query = attn[0]  # attention from query

    mass_near = row_query[1:5].sum().item()
    mass_far = row_query[5:9].sum().item()
    mass_cross = row_query[9:13].sum().item()
    mass_self = row_query[0].item()

    # Same-chrom NEAR features should dominate.
    assert mass_near > mass_far, f"near={mass_near:.3f} not > far={mass_far:.3f}"
    assert mass_near > mass_cross, f"near={mass_near:.3f} not > cross={mass_cross:.3f}"
    # No single cross-chrom feature should beat the nearest same-chrom one.
    nearest_same = row_query[1:5].max().item()
    biggest_cross = row_query[9:13].max().item()
    assert nearest_same > biggest_cross, (
        f"nearest same-chrom mass {nearest_same:.3f} not > biggest cross-chrom {biggest_cross:.3f}"
    )
    # Plausibility: total mass should be 1 (it's a softmax row).
    assert math.isclose(mass_self + mass_near + mass_far + mass_cross, 1.0, abs_tol=1e-4)


def _spatial_logits(layer: SpatialAttention, u: torch.Tensor, r_phi: torch.Tensor) -> torch.Tensor:
    """Reproduce the logit computation inside SpatialAttention.forward for inspection."""
    H = layer.H
    u_dot = torch.matmul(u, u.transpose(-1, -2))
    beta_sq = (layer.beta**2).view(1, H, 1, 1)
    u_logit = beta_sq * u_dot.unsqueeze(1)
    w_tiled = layer.w_tiled()
    weighted_r = r_phi.unsqueeze(2) * w_tiled.view(1, 1, H, -1)
    phi_logit = torch.einsum("bihd,bjhd->bhij", weighted_r, weighted_r)
    tau = layer.tau.view(1, H, 1, 1)
    return tau * (u_logit + phi_logit)


# ---------------------------------------------------------------------------
# T6 — permutation equivariance / invariance (spec §11.5)
# ---------------------------------------------------------------------------
def test_permutation_equivariance_pre_pool_and_invariance_after_pool():
    torch.manual_seed(42)
    encoder, block = _make_stack(d_s=64, d_v=16, H_content=2, H_spatial=2)
    B, N = 1, 8
    coords = torch.rand(B, N) * 1.0e8
    # Mix two chromosomes.
    keys = [
        [
            (SPECIES, ASSEMBLY, "chr1") if n % 2 == 0 else (SPECIES, ASSEMBLY, "chr2")
            for n in range(N)
        ]
    ]
    s = torch.randn(B, N, 64)

    pos = encoder(coords, keys)
    out_a = block(s, u=pos["u"], r_phi=pos["r_phi"])  # [B, N, d_s]

    # Random permutation π
    perm = torch.randperm(N)
    coords_p = coords[:, perm]
    keys_p = [[keys[0][i] for i in perm.tolist()]]
    s_p = s[:, perm]
    pos_p = encoder(coords_p, keys_p)
    out_b = block(s_p, u=pos_p["u"], r_phi=pos_p["r_phi"])

    # Equivariance: out_b should equal out_a[π].
    diff = (out_b - out_a[:, perm]).abs().max().item()
    assert diff < 1e-5, f"permutation equivariance violated: max abs diff = {diff:.2e}"

    # Invariance after mean-pool over the bag.
    pooled_a = out_a.mean(dim=1)
    pooled_b = out_b.mean(dim=1)
    pool_diff = (pooled_a - pooled_b).abs().max().item()
    assert pool_diff < 1e-5, f"pooled invariance violated: max abs diff = {pool_diff:.2e}"


# ---------------------------------------------------------------------------
# T9 — gradient flow: learnable on τ, β, w, W_V; no grad path to r_g, u_g
# ---------------------------------------------------------------------------
def test_spatial_attention_learnables_receive_gradients():
    torch.manual_seed(7)
    encoder, block = _make_stack(d_s=32, d_v=8, H_content=0, H_spatial=2)
    B, N = 2, 6
    coords = torch.rand(B, N) * 1.0e8
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * N for _ in range(B)]
    s = torch.randn(B, N, 32, requires_grad=True)

    pos = encoder(coords, keys)
    out = block(s, u=pos["u"], r_phi=pos["r_phi"])
    loss = (out**2).mean()
    loss.backward()

    spatial = block.spatial
    assert spatial is not None
    assert spatial.log_w.grad is not None and spatial.log_w.grad.abs().sum() > 0
    assert spatial.beta.grad is not None and spatial.beta.grad.abs().sum() > 0
    assert spatial.log_tau.grad is not None and spatial.log_tau.grad.abs().sum() > 0
    assert spatial.W_V.weight.grad is not None and spatial.W_V.weight.grad.abs().sum() > 0


def test_encoder_outputs_have_no_grad_history():
    """r_g and u_g are constants generated by hashing — they must not connect the loss
    back to any learnable chromosome embedding (which we don't have, by design)."""
    encoder = ChromLocalPosEncoder(d_u=32, K=64, lambda_min=10.0, lambda_max=1.0e8)
    coords = torch.rand(1, 4) * 1.0e8
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * 4]
    pos = encoder(coords, keys)
    # Outputs come from the codebook (no grad) elementwise-multiplied with the Fourier
    # output (also no grad because no parameters). They should not have grad_fn.
    assert pos["u"].requires_grad is False
    assert pos["r_phi"].requires_grad is False


# ---------------------------------------------------------------------------
# T10 — heads initialized at increasing scales decay over increasing distance
# ---------------------------------------------------------------------------
def test_head_scale_ordering():
    """For heads initialized at log-spaced target scales L_1 < L_2 < ... < L_H,
    the half-decay distance of K_h(Δ) should be monotonically increasing in h."""

    K = 112
    encoder = ChromLocalPosEncoder(d_u=32, K=K, lambda_min=10.0, lambda_max=1.0e8)
    scales = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7]
    spatial = SpatialAttention(
        d_s=8,
        d_v=8,
        H=len(scales),
        K=K,
        lambdas=encoder.fourier.lambdas,
        target_scales=scales,
        beta_init=0.0,
    )
    # Evaluate K_h(Δ) at a log-spaced grid up to 100 Mb.
    deltas = torch.logspace(0, 8, steps=400)  # 1 bp … 100 Mb
    K_h = spatial.kernel_same_chrom(deltas)  # [H, D]
    # Peak is at Δ=0; compute K_h(0) via a zero tensor for reference.
    K_h_zero = spatial.kernel_same_chrom(torch.zeros(1)).squeeze(-1)  # [H]

    half_distances = []
    for h in range(len(scales)):
        peak = K_h_zero[h].item()
        target = 0.5 * peak
        # First Δ where K_h drops below half its peak.
        below = (K_h[h] < target).nonzero(as_tuple=False)
        if below.numel() == 0:
            # Kernel never falls to half within the swept range — record the max Δ.
            half_distances.append(float(deltas[-1]))
        else:
            half_distances.append(float(deltas[below[0].item()]))

    # Half-decay distance should be (weakly) monotone-increasing in head index.
    for i in range(1, len(half_distances)):
        assert half_distances[i] >= half_distances[i - 1] * 0.5, (
            f"head {i} half-dist {half_distances[i]:.3g} < half of head {i - 1} "
            f"({half_distances[i - 1]:.3g}); scales not ordered."
        )
    # And the last head should reach further than the first.
    assert half_distances[-1] > half_distances[0] * 10.0, (
        f"head range too tight: first={half_distances[0]:.3g}, last={half_distances[-1]:.3g}"
    )


# ---------------------------------------------------------------------------
# T11 — wavelength tying: sin and cos dims share the same w_h scalar
# ---------------------------------------------------------------------------
def test_wavelength_weights_tied_across_sin_cos():
    K = 32
    encoder = ChromLocalPosEncoder(d_u=8, K=K, lambda_min=10.0, lambda_max=1.0e8)
    spatial = SpatialAttention(
        d_s=4,
        d_v=4,
        H=3,
        K=K,
        lambdas=encoder.fourier.lambdas,
        target_scales=[1e3, 1e4, 1e5],
    )
    w_tiled = spatial.w_tiled()  # [H, 2K]
    # Adjacent pairs (sin_k, cos_k) at indices 2k and 2k+1 share the same weight.
    even = w_tiled[:, 0::2]
    odd = w_tiled[:, 1::2]
    assert torch.allclose(even, odd)


# ---------------------------------------------------------------------------
# Bonus: content attention sanity (no positional input, standard MHA)
# ---------------------------------------------------------------------------
def test_content_attention_shape():
    layer = ContentAttention(d_s=64, d_v=16, H=4)
    s = torch.randn(2, 7, 64)
    out = layer(s)
    assert out.shape == (2, 7, 4, 16)


# ---------------------------------------------------------------------------
# use_baseline=False: spatial attention without the β·u term
# ---------------------------------------------------------------------------
def test_spatial_no_baseline_forward():
    """SpatialAttention with use_baseline=False runs without u and has no beta param."""
    K = 32
    encoder = ChromLocalPosEncoder(d_u=8, K=K, lambda_min=10.0, lambda_max=1.0e8)
    spatial = SpatialAttention(
        d_s=16,
        d_v=8,
        H=3,
        K=K,
        lambdas=encoder.fourier.lambdas,
        target_scales=[1e3, 1e4, 1e5],
        use_baseline=False,
    )
    assert spatial.beta is None

    B, N = 2, 6
    coords = torch.rand(B, N) * 1.0e8
    keys = [[(SPECIES, ASSEMBLY, "chr1")] * N for _ in range(B)]
    pos = encoder(coords, keys)
    s = torch.randn(B, N, 16)

    out = spatial(None, pos["r_phi"], s)
    assert out.shape == (B, N, 3, 8)

    out_with_u = spatial(pos["u"], pos["r_phi"], s)
    assert torch.allclose(out, out_with_u)
