"""Plot per-head spatial kernel K_h(Delta) at initialization.

Reproduces the kernel_sweep figure showing how each position head's
log-Gaussian wavelength weighting produces a distinct length-scale
sensitivity profile.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from bloom_arch.posenc import FourierCoordEncoder


def compute_kernel(
    K: int,
    lambda_min: float,
    lambda_max: float,
    target_scales: list[float],
    rho: float,
    beta: float,
    deltas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-head kernel and unweighted kernel over distance array.

    Returns:
        kernels: [H, len(deltas)] -- per-head K_h(Delta)
        unweighted: [len(deltas)] -- (1/K) sum_k cos(2pi*Delta/lam_k)
    """
    encoder = FourierCoordEncoder(K=K, lambda_min=lambda_min, lambda_max=lambda_max)
    lambdas = encoder.lambdas.numpy()

    # Log-Gaussian weights per head (matching _log_gaussian_init)
    log_lambdas = np.log(lambdas)
    H = len(target_scales)
    kernels = np.zeros((H, len(deltas)))

    for h, L_h in enumerate(target_scales):
        log_L = math.log(L_h)
        log_w_unnorm = -((log_lambdas - log_L) ** 2) / (2 * rho**2)
        w_unnorm = np.exp(log_w_unnorm)
        w = w_unnorm * math.sqrt(K) / np.linalg.norm(w_unnorm)

        for i, d in enumerate(deltas):
            cos_terms = np.cos(2 * np.pi * d / lambdas)
            kernels[h, i] = beta**2 + np.sum(w**2 * cos_terms) / K

    # Unweighted kernel (all w_k = 1, so ||w||^2 = K automatically)
    unweighted = np.zeros(len(deltas))
    for i, d in enumerate(deltas):
        unweighted[i] = np.mean(np.cos(2 * np.pi * d / lambdas))

    return kernels, unweighted


def main():
    K = 112
    lambda_min = 10.0
    lambda_max = 1.0e8
    rho = math.log(10.0) / 2.0
    beta = 1.0

    # 6 heads log-spaced
    lo = lambda_min * 10
    hi = lambda_max / 10
    H = 6
    target_scales = [
        math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * h / (H - 1)) for h in range(H)
    ]

    deltas = np.logspace(0, 8.5, 500)  # 1 bp to ~300 Mb

    kernels, unweighted = compute_kernel(
        K, lambda_min, lambda_max, target_scales, rho, beta, deltas
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, H))
    for h in range(H):
        scale_str = _format_bp(target_scales[h])
        ax.plot(
            deltas, kernels[h], color=colors[h], linewidth=1.5, label=f"head {h}: L={scale_str}"
        )

    ax.plot(deltas, unweighted, "k--", linewidth=1, alpha=0.6, label="unweighted φ(x)·φ(x')")

    ax.set_xscale("log")
    ax.set_xlabel("genomic distance Δ (bp)")
    ax.set_ylabel("same-chrom kernel K_h(Δ)")
    lmin_s, lmax_s = _format_bp(lambda_min), _format_bp(lambda_max)
    ax.set_title(
        f"Per-head spatial kernel at initialization\n"
        f"K={K} wavelengths, λ∈[{lmin_s}, {lmax_s}], log-Gaussian windows (ρ={rho:.2f})"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.2, 2.2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "docs/figures/kernel_sweep_v2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def _format_bp(bp: float) -> str:
    if bp >= 1e6:
        return f"{bp / 1e6:.0f} Mb"
    elif bp >= 1e3:
        return f"{bp / 1e3:.0f} kb"
    else:
        return f"{bp:.0f} bp"


if __name__ == "__main__":
    main()
