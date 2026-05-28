"""Plot softmax attention matrices under the full position-head design.

Reproduces the cross_chrom_attention figure showing per-head attention
patterns for features on two chromosomes, with coordinates sorted so that
within-chrom locality appears as diagonal concentration.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bloom_arch.posenc import PosEncoder


def build_attention_matrices(
    K: int = 112,
    d_u: int = 32,
    lambda_min: float = 10.0,
    lambda_max: float = 1.0e8,
    rho: float = math.log(10.0) / 2.0,
    beta: float = 1.0,
    tau: float = 4.0,
    n_features_per_chrom: int = 48,
    max_coord: float = 5.0e6,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[str]]:
    """Build per-head attention matrices for 2 chromosomes.

    Returns:
        attn_list: list of [L, L] attention weight arrays, one per head
        scale_labels: human-readable scale for each head
    """
    L_per = n_features_per_chrom

    # Sorted coordinates on two chromosomes. Use log-uniform spacing so
    # there are pairs at every length scale from ~10bp to ~max_coord.
    torch.manual_seed(seed)
    log_min, log_max = math.log10(100.0), math.log10(max_coord)
    coords_chr1 = torch.sort(10 ** (torch.rand(L_per) * (log_max - log_min) + log_min))[0]
    coords_chr2 = torch.sort(10 ** (torch.rand(L_per) * (log_max - log_min) + log_min))[0]
    x = torch.cat([coords_chr1, coords_chr2]).unsqueeze(0)  # [1, L]
    g = torch.cat(
        [torch.zeros(L_per, dtype=torch.long), torch.ones(L_per, dtype=torch.long)]
    ).unsqueeze(0)  # [1, L]

    # Encode positions
    encoder = PosEncoder(
        max_G=30, d_u=d_u, K=K, lambda_min=lambda_min, lambda_max=lambda_max, seed=seed
    )
    pos = encoder(x, g)
    u = pos["u"]  # [1, L, d_u]
    r_phi = pos["r_phi"]  # [1, L, 2K]

    # Derive target scales (same logic as PositionHeads default)
    lambdas = encoder.fourier.lambdas
    lo = float(lambdas.min()) * 10.0
    hi = float(lambdas.max()) / 10.0
    H = 6
    target_scales = [
        math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * h / (H - 1)) for h in range(H)
    ]

    # Compute per-head log-Gaussian weights
    log_lambdas = torch.log(lambdas)
    attn_list = []
    scale_labels = []

    for _h, L_h in enumerate(target_scales):
        log_L = math.log(L_h)
        diff = log_lambdas - log_L
        log_w_unnorm = -(diff**2) / (2 * rho**2)
        w_unnorm = log_w_unnorm.exp()
        w = w_unnorm * math.sqrt(K) / w_unnorm.norm()
        w_tiled = w.repeat_interleave(2)  # [2K]

        # Build QK: sqrt(tau) * [beta * u; w_tiled * r_phi]
        sqrt_tau = math.sqrt(tau)
        u_part = beta * u  # [1, L, d_u]
        r_part = r_phi * w_tiled.unsqueeze(0).unsqueeze(0)  # [1, L, 2K]
        qk = torch.cat([u_part, r_part], dim=-1) * sqrt_tau  # [1, L, d_u + 2K]

        # Attention logits: Q.K^T with scale=1.0
        logits = torch.bmm(qk, qk.transpose(1, 2))  # [1, L, L]
        attn = F.softmax(logits, dim=-1).squeeze(0).detach().numpy()
        attn_list.append(attn)
        scale_labels.append(_format_bp(L_h))

    return attn_list, scale_labels


def main():
    attn_list, scale_labels = build_attention_matrices(tau=4.0)

    H = len(attn_list)
    fig, axes = plt.subplots(1, H, figsize=(3.2 * H, 3.0))

    # Shared color scale across all panels
    uniform_attn = 1.0 / attn_list[0].shape[0]
    vmax = uniform_attn * 3  # clip at 3x uniform for visibility

    for h, (attn, label) in enumerate(zip(attn_list, scale_labels, strict=False)):
        ax = axes[h]
        im = ax.imshow(attn, aspect="equal", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"L = {label}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Softmax attention under the full design  (τ = 4.0)  — rows are queries, columns are keys",
        fontsize=11,
    )
    fig.colorbar(im, ax=axes, shrink=0.8, label="attention weight")
    plt.tight_layout(rect=[0, 0, 0.93, 0.93])

    out_path = "docs/figures/cross_chrom_attention_v2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def _format_bp(bp: float) -> str:
    if bp >= 1e6:
        return f"{bp / 1e6:.0f} Mb"
    elif bp >= 1e3:
        return f"{bp / 1e3:.1f} kb"
    else:
        return f"{bp:.0f} bp"


if __name__ == "__main__":
    main()
