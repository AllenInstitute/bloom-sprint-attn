"""Visualize the per-head spatial kernel K_h(Δ) at initialization.

Materializes a :class:`SpatialAttention` with default 6 heads at target scales
{100 bp, 1 kb, 10 kb, 100 kb, 1 Mb, 10 Mb} and sweeps the same-chromosome
kernel as a function of genomic distance Δ. The resulting figure is the
qualitative sanity check requested in the spec discussion: each head should
peak at Δ=0 and decay around its target scale, with the family of curves
ordered from short-range to long-range.

Also overlays the unweighted Fourier kernel (1/K) Σ_k cos(2π Δ / λ_k) and the
β² floor so we can see exactly what scale-weighting buys us over the raw
multi-scale bank.

Usage
-----
$ python scripts/plot_kernel_scales.py --out docs/figures/kernel_sweep.png

Defaults match docs/spec.md §10.1 and §12.3.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bloom_arch.attention import SpatialAttention
from bloom_arch.posenc import FourierCoordEncoder


def kernel_curves(
    K: int,
    lambda_min: float,
    lambda_max: float,
    target_scales: list[float],
    rho: float,
    deltas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (per-head K_h, baseline cos kernel, β² floor) at initialization."""
    fcoder = FourierCoordEncoder(K=K, lambda_min=lambda_min, lambda_max=lambda_max)
    spatial = SpatialAttention(
        d_s=8, d_v=8, H=len(target_scales), K=K,
        lambdas=fcoder.lambdas,
        target_scales=target_scales,
        rho=rho,
        beta_init=0.3,           # small β so the floor is visible but doesn't dominate
        tau_init=1.0,
    )
    with torch.no_grad():
        K_h = spatial.kernel_same_chrom(deltas)        # [H, D]
        floor = (spatial.beta ** 2).clone()             # [H]
        # Unweighted baseline: (1/K) Σ_k cos(2π Δ / λ_k)
        omegas = 2.0 * math.pi / fcoder.lambdas
        baseline = (
            torch.cos(deltas.unsqueeze(-1) * omegas).sum(-1) / K
        )                                               # [D]
    return K_h, baseline, floor


def format_bp(bp: float) -> str:
    """Pretty-print a genomic distance."""
    if bp >= 1e6:
        return f"{bp / 1e6:.0f} Mb"
    if bp >= 1e3:
        return f"{bp / 1e3:.0f} kb"
    return f"{bp:.0f} bp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("docs/figures/kernel_sweep.png"),
        help="Output PNG path (relative to repo root).",
    )
    parser.add_argument("--K", type=int, default=112, help="Number of Fourier wavelengths.")
    parser.add_argument("--lambda-min", type=float, default=10.0, help="Min wavelength (bp).")
    parser.add_argument("--lambda-max", type=float, default=1.0e8, help="Max wavelength (bp).")
    parser.add_argument("--rho", type=float, default=math.log(10.0) / 2.0,
                        help="Broadness of log-Gaussian window (in nat log units).")
    args = parser.parse_args()

    target_scales = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7]  # 100 bp .. 10 Mb
    deltas = torch.logspace(0, 8, steps=600)  # 1 bp .. 100 Mb

    K_h, baseline, floor = kernel_curves(
        K=args.K, lambda_min=args.lambda_min, lambda_max=args.lambda_max,
        target_scales=target_scales, rho=args.rho, deltas=deltas,
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)

    cmap = plt.get_cmap("viridis")
    H = len(target_scales)
    for h, L in enumerate(target_scales):
        color = cmap(h / max(H - 1, 1))
        ax.plot(
            deltas.numpy(),
            K_h[h].numpy(),
            color=color,
            linewidth=2.0,
            label=f"head {h}: L={format_bp(L)}",
        )
        # β² floor for each head — small horizontal line at the left edge.
        ax.axhline(floor[h].item(), color=color, alpha=0.15, linewidth=0.8, linestyle=":")

    ax.plot(
        deltas.numpy(), baseline.numpy(),
        color="black", linestyle="--", linewidth=1.4, alpha=0.7,
        label=r"unweighted $\phi(x_i)\cdot\phi(x_j)$",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"genomic distance $\Delta$ (bp)")
    ax.set_ylabel(r"same-chromosome kernel $K_h(\Delta)$")
    ax.set_title(
        "Per-head spatial kernel at initialization\n"
        rf"K={args.K} wavelengths, $\lambda\in[{format_bp(args.lambda_min)},\,{format_bp(args.lambda_max)}]$,"
        rf" log-Gaussian windows ($\rho$={args.rho:.2f})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Vertical lines at each target scale, faintly.
    for h, L in enumerate(target_scales):
        ax.axvline(L, color=cmap(h / max(H - 1, 1)), alpha=0.15, linewidth=0.8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    # Also print the half-decay distances so the log captures a textual summary.
    print("\nHalf-decay distances (Δ where K_h drops to half its peak):")
    peaks = K_h[:, 0]
    for h, L in enumerate(target_scales):
        peak = peaks[h].item()
        target = 0.5 * peak
        below = (K_h[h] < target).nonzero(as_tuple=False)
        if below.numel() == 0:
            half_d = float(deltas[-1])
            note = " (never crossed; report max Δ)"
        else:
            half_d = float(deltas[below[0].item()])
            note = ""
        print(
            f"  head {h}: target L={format_bp(L):>7}  "
            f"peak={peak:.3f}  half-Δ={format_bp(half_d):>7}{note}"
        )


if __name__ == "__main__":
    main()
