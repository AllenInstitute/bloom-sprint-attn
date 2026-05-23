"""Cross-chromosome cancellation diagnostic.

Visualizes the structural cancellation of cross-chromosome dot products in
the chromosome-local positional encoding, with ablations that isolate the
contributions of the sign-gating term (r_g) and the chromosome baseline
(beta * u_g).

Setup:
  - 64 features, half on chr1 and half on chr2.
  - Coordinates log-spaced over [10 bp, 100 Mb] on each chromosome.
  - Feature i on chr1 has a "coord-twin" at index i + 32 on chr2 — same
    coordinate, different chromosome. This is the case the design has to
    handle: cross-chrom dot products should vanish *even when coordinates
    match exactly*.

For each spatial head, computes the pre-softmax kernel
    K_h(i, j) = beta_h^2 * (u_i . u_j)
              + sum_d w_tiled[h, d]^2 * r_phi_i[d] * r_phi_j[d]
under three variants:
  - 'full'     : the shipping design
  - 'no_sign'  : replace r_g (.) phi with bare phi (drop the sign gating)
  - 'no_beta'  : set beta = 0 (drop the chromosome baseline)

Then renders heatmaps for each (variant, head) cell. Expected block
structure for 'full':

         chr1↔chr1 | chr1↔chr2
        -----------+-----------
         chr2↔chr1 | chr2↔chr2

with diagonal blocks showing the scale-selective kernel on a beta^2 floor
and off-diagonal blocks at ~0 in absolute value.

Outputs (docs/figures/):
  - cross_chrom_kernel_3x6.png   : main ablation grid
  - cross_chrom_attention.png    : softmax attention for the full design
  - cross_chrom_K_sweep.png      : effect of K on the cross-chrom noise
  - cross_chrom_beta_sweep.png   : effect of beta on the on-chrom floor
  - cross_chrom_tau_sweep.png    : effect of tau on softmax sharpness
  - cross_chrom_summary.json     : numerical mean(|cross|) / mean(|on|)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from bloom_arch.attention.spatial import SpatialAttention
from bloom_arch.posenc import ChromLocalPosEncoder

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------
SPECIES, ASSEMBLY = "musmus", "mm10"
TARGET_SCALES = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
DEFAULT_K = 112
N_PER_CHROM = 32
COORDS_PER = torch.logspace(1.0, 8.0, N_PER_CHROM, dtype=torch.float64)  # 10 bp .. 100 Mb


def format_bp(x: float) -> str:
    if x >= 1.0e6:
        return f"{x / 1.0e6:.0f} Mb"
    if x >= 1.0e3:
        return f"{x / 1.0e3:.0f} kb"
    return f"{x:.0f} bp"


def build_pair(
    K: int = DEFAULT_K,
    beta_init: float = 0.5,
    tau_init: float = 4.0,
    rho: float = math.log(10.0) / 2.0,
):
    """Build encoder + spatial layer in float64 (precision matters at λ_min=10)."""
    enc = ChromLocalPosEncoder(d_u=32, K=K, lambda_min=10.0, lambda_max=1.0e8).double()
    sp = SpatialAttention(
        d_s=32,
        d_v=8,
        H=6,
        K=K,
        lambdas=enc.fourier.lambdas,
        target_scales=TARGET_SCALES,
        beta_init=beta_init,
        tau_init=tau_init,
        rho=rho,
    ).double()
    return enc, sp


def build_inputs(encoder):
    coords = torch.cat([COORDS_PER, COORDS_PER]).unsqueeze(0)  # [1, 2 * N_PER_CHROM]
    keys = [
        [(SPECIES, ASSEMBLY, "chr1")] * N_PER_CHROM + [(SPECIES, ASSEMBLY, "chr2")] * N_PER_CHROM
    ]
    pos = encoder(coords, keys)
    phi = encoder.fourier(coords)  # unsigned Fourier features — needed for 'no_sign'.
    return pos["u"][0], pos["r_phi"][0], phi[0]


# ---------------------------------------------------------------------------
# Kernel computation under each ablation
# ---------------------------------------------------------------------------
def kernels(spatial, u, r_phi, phi, mode: str) -> torch.Tensor:
    """Return per-head pre-softmax kernel of shape [H, N, N] under one mode."""
    H = spatial.H
    w_tiled = spatial.w_tiled().detach().to(u.dtype)  # [H, 2K]

    u_dot = u @ u.T  # [N, N]
    beta_sq = (
        torch.zeros(H, dtype=u.dtype)
        if mode == "no_beta"
        else (spatial.beta**2).detach().to(u.dtype)  # [H]
    )

    if mode == "no_sign":
        base = phi.unsqueeze(0).expand(H, -1, -1)  # [H, N, 2K]
    else:
        base = r_phi.unsqueeze(0).expand(H, -1, -1)  # [H, N, 2K]

    weighted = base * w_tiled.unsqueeze(1)  # [H, N, 2K]
    phi_logit = torch.einsum("hid,hjd->hij", weighted, weighted)  # [H, N, N]
    return beta_sq.view(H, 1, 1) * u_dot.unsqueeze(0) + phi_logit


# ---------------------------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------------------------
def block_ratio(K_hij: torch.Tensor, n_per: int = N_PER_CHROM):
    """Per-head mean(|cross-chrom|) / mean(|on-chrom, off-diagonal|)."""
    H, N, _ = K_hij.shape
    chrom = torch.cat([torch.zeros(n_per), torch.ones(n_per)]).long()
    same = chrom.unsqueeze(0) == chrom.unsqueeze(1)
    eye = torch.eye(N, dtype=torch.bool)
    on_mask = same & ~eye
    cross_mask = ~same

    out = []
    for h in range(H):
        on = K_hij[h][on_mask].abs().mean().item()
        cross = K_hij[h][cross_mask].abs().mean().item()
        out.append({"on": on, "cross": cross, "ratio": cross / on if on > 0 else float("nan")})
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _block_lines(ax, color="k", lw=0.6):
    ax.axhline(N_PER_CHROM - 0.5, color=color, lw=lw)
    ax.axvline(N_PER_CHROM - 0.5, color=color, lw=lw)


def plot_kernel_grid(K_by_mode: dict, outpath: Path) -> None:
    modes = ["full", "no_sign", "no_beta"]
    labels = {
        "full": "Full design\n(β·u_g + r_g⊙φ)",
        "no_sign": "Ablation: drop r_g\n(no sign gating)",
        "no_beta": "Ablation: β = 0\n(no chrom baseline)",
    }
    H = K_by_mode["full"].shape[0]
    fig, axes = plt.subplots(
        len(modes), H, figsize=(2.4 * H, 2.7 * len(modes)), constrained_layout=True
    )
    vmax = max(K_by_mode[m].abs().max().item() for m in modes)
    vmax = min(vmax, 1.3)
    vmin = -vmax
    im = None
    for r, mode in enumerate(modes):
        for h in range(H):
            ax = axes[r, h]
            im = ax.imshow(
                K_by_mode[mode][h].numpy(),
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                origin="upper",
            )
            _block_lines(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"L = {format_bp(TARGET_SCALES[h])}", fontsize=10)
            if h == 0:
                ax.set_ylabel(labels[mode], fontsize=10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.012)
    cbar.set_label("kernel  K_h(i, j)")
    fig.suptitle(
        "Cross-chromosome cancellation: pre-softmax kernel  "
        "(features 0–31 = chr1, 32–63 = chr2, coords log-spaced 10 bp .. 100 Mb)",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_softmax_attention(K_full: torch.Tensor, tau: float, outpath: Path) -> None:
    H, N, _ = K_full.shape
    A = torch.softmax(tau * K_full, dim=-1)  # softmax over j
    fig, axes = plt.subplots(1, H, figsize=(2.5 * H, 2.9), constrained_layout=True)
    vmax = float(A.max())
    im = None
    for h in range(H):
        ax = axes[h]
        im = ax.imshow(A[h].numpy(), cmap="magma", vmin=0, vmax=vmax, aspect="equal")
        _block_lines(ax, color="white")
        ax.set_title(f"L = {format_bp(TARGET_SCALES[h])}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.012)
    cbar.set_label("attention weight")
    fig.suptitle(
        f"Softmax attention under the full design  (τ = {tau:.1f})  "
        "— rows are queries, columns are keys",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_K_sweep(values, head_idx: int, outpath: Path) -> None:
    fig, axes = plt.subplots(
        1, len(values), figsize=(2.6 * len(values), 2.9), constrained_layout=True
    )
    vmax = None
    im = None
    L_label = None
    for ax, K_val in zip(axes, values, strict=True):
        enc, sp = build_pair(K=K_val)
        u, r_phi, phi = build_inputs(enc)
        K_hij = kernels(sp, u, r_phi, phi, "full")
        if vmax is None:
            vmax = K_hij[head_idx].abs().max().item()
            L_label = format_bp(TARGET_SCALES[head_idx])
        im = ax.imshow(
            K_hij[head_idx].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal"
        )
        _block_lines(ax)
        ax.set_title(f"K = {K_val}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.012)
    cbar.set_label("kernel value")
    fig.suptitle(
        f"Effect of K on cross-chrom noise floor  (head L = {L_label}; CLT predicts ~1/K variance)",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_beta_sweep(values, head_idx: int, outpath: Path) -> None:
    fig, axes = plt.subplots(
        1, len(values), figsize=(2.6 * len(values), 2.9), constrained_layout=True
    )
    # Determine vmax from the largest beta to keep a shared scale legible.
    enc_max, sp_max = build_pair(beta_init=max(values))
    u_m, r_m, phi_m = build_inputs(enc_max)
    vmax = kernels(sp_max, u_m, r_m, phi_m, "full")[head_idx].abs().max().item()
    L_label = format_bp(TARGET_SCALES[head_idx])
    im = None
    for ax, beta in zip(axes, values, strict=True):
        enc, sp = build_pair(beta_init=beta)
        u, r_phi, phi = build_inputs(enc)
        K_hij = kernels(sp, u, r_phi, phi, "full")
        im = ax.imshow(
            K_hij[head_idx].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal"
        )
        _block_lines(ax)
        ax.set_title(f"β = {beta:.1f}   (floor β² = {beta**2:.2f})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.012)
    cbar.set_label("kernel value")
    fig.suptitle(
        f"Effect of β on the on-chrom floor (head L = {L_label}); "
        "cross-chrom blocks are unaffected by β.",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_tau_sweep(values, head_idx: int, outpath: Path) -> None:
    enc, sp = build_pair()
    u, r_phi, phi = build_inputs(enc)
    K_hij = kernels(sp, u, r_phi, phi, "full")
    fig, axes = plt.subplots(
        1, len(values), figsize=(2.6 * len(values), 2.9), constrained_layout=True
    )
    L_label = format_bp(TARGET_SCALES[head_idx])
    im = None
    for ax, tau in zip(axes, values, strict=True):
        A = torch.softmax(tau * K_hij[head_idx], dim=-1)
        im = ax.imshow(A.numpy(), cmap="magma", vmin=0, vmax=float(A.max()), aspect="equal")
        _block_lines(ax, color="white")
        ax.set_title(f"τ = {tau:.1f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.012)
    cbar.set_label("attention weight")
    fig.suptitle(
        f"Effect of τ on softmax sharpness (head L = {L_label}); "
        "kernel is fixed; τ alone controls concentration.",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(0)
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    enc, sp = build_pair()
    u, r_phi, phi = build_inputs(enc)
    K_by_mode = {m: kernels(sp, u, r_phi, phi, m) for m in ("full", "no_sign", "no_beta")}

    plot_kernel_grid(K_by_mode, out_dir / "cross_chrom_kernel_3x6.png")
    plot_softmax_attention(
        K_by_mode["full"], tau=4.0, outpath=out_dir / "cross_chrom_attention.png"
    )
    plot_K_sweep([16, 32, 112, 256], head_idx=2, outpath=out_dir / "cross_chrom_K_sweep.png")
    plot_beta_sweep(
        [0.0, 0.5, 1.0, 1.5], head_idx=2, outpath=out_dir / "cross_chrom_beta_sweep.png"
    )
    plot_tau_sweep([1.0, 2.0, 4.0, 8.0], head_idx=2, outpath=out_dir / "cross_chrom_tau_sweep.png")

    # Numerical summary: mean(|cross|) / mean(|on|) per head per mode.
    summary = {m: block_ratio(K_by_mode[m]) for m in ("full", "no_sign", "no_beta")}
    (out_dir / "cross_chrom_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- mean(|cross|) / mean(|on|)  (lower is better; target < 0.1 for 'full') ---")
    print(f"{'head':<5}{'L':<10}{'full':<10}{'no_sign':<10}{'no_beta':<10}")
    for h in range(6):
        L = format_bp(TARGET_SCALES[h])
        print(
            f"{h:<5}{L:<10}"
            f"{summary['full'][h]['ratio']:<10.4f}"
            f"{summary['no_sign'][h]['ratio']:<10.4f}"
            f"{summary['no_beta'][h]['ratio']:<10.4f}"
        )


if __name__ == "__main__":
    main()
