# bloom-arch

Chromosome-local positional encoding and typed multi-head attention for
transformer models that operate on bags of genome-derived features per cell.

The user gives the model a bag of features. Each feature carries a content
embedding `s_i` (from a DNA encoder, an existing token embedding, etc.) and
a genomic locus `(species, assembly, chromosome, coordinate)`. Attention is
split into two head types:

- **Content heads** — standard self-attention over `s`.
- **Spatial heads** — attention weights come from the locus; values come
  from `s`. Same-chromosome features attract; same-coordinate features on
  *different* chromosomes do not. No learned chromosome lookup tables, so
  the encoder generalizes to assemblies and species not seen at training.

Each spatial head has a learnable wavelength window centered at a chosen
length scale (100 bp ... 10 Mb), so a single block can resolve sequence-,
enhancer-, and TAD-level locality at the same time.

See `docs/spec.md` for the design rationale and `docs/implementation.md`
for the mathematical details, parameter ranges, and tuning notes.

## Quick start

```python
import torch
from bloom_arch.posenc import ChromLocalPosEncoder
from bloom_arch.attention import TypedMultiHeadAttention

# Dimensions.
B, N    = 4, 32          # batch, features per cell
d_s     = 512            # sequence/content width (the model's residual width)
d_v     = 64             # per-head value dim
H_c     = 4              # content heads
H_s     = 6              # spatial heads, one per length scale
K       = 112            # number of Fourier wavelengths

# 1. Build the positional encoder. Zero learnable parameters.
encoder = ChromLocalPosEncoder(d_u=32, K=K, lambda_min=10.0, lambda_max=1.0e8)

# 2. Build a typed attention block with content + spatial heads.
attn = TypedMultiHeadAttention(
    d_s=d_s, d_v=d_v,
    H_content=H_c, H_spatial=H_s,
    K=K, lambdas=encoder.fourier.lambdas,
    target_scales=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7],  # bp; one per spatial head
)

# 3. Inputs: content embeddings, coordinates, and (species, assembly, chrom) keys.
s      = torch.randn(B, N, d_s)
coords = torch.rand(B, N) * 2.0e8                          # bp on the chromosome
keys   = [[("mus_musculus", "mm10", "chr1")] * N for _ in range(B)]

# 4. Encode positions, then run attention.
pos = encoder(coords, keys)                                # {"u": ..., "r_phi": ...}
out = attn(s, u=pos["u"], r_phi=pos["r_phi"])              # [B, N, d_s]
```

A few notes on the interface:

- `coords` are raw base-pair integers cast to a float tensor. Use
  `dtype=torch.long`-then-`.float()` to avoid precision issues at scales
  beyond ~1e7 bp. (Float32 has ~7 decimal digits.)
- `keys[b][n] = (species, assembly, chrom)` are arbitrary strings; the
  codebook hashes them deterministically. Novel keys at inference time are
  fine — the codebook will generate stable codes for them on the fly.
- `target_scales` controls each spatial head's initial center wavelength.
  Defaults to a geometric spacing across the encoder's `[λ_min, λ_max]`
  range when omitted.

## Controlling what is learned

Each spatial head has three component families that can individually be
made learnable or fixed:

| Component | Flag | Default | Effect when disabled |
|---|---|---|---|
| β·u baseline | `use_baseline=False` | on | Removes the β_h² · (u_i·u_j) floor entirely; `u` may be None |
| Temperature τ | `learn_tau=False` | learns | τ_h stays fixed at `tau_init` (buffer, no gradient) |
| Wavelength weights w | `learn_w=False` | learns | w_h stays at its log-Gaussian initialization (buffer, no gradient) |

```python
# Example: fixed temperature, learned weights, no chromosome baseline.
spatial = SpatialAttention(
    d_s=512, d_v=64, H=6, K=112,
    lambdas=encoder.fourier.lambdas,
    target_scales=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
    use_baseline=False,
    learn_tau=False,
    learn_w=True,
)
# u is not needed when baseline is off:
out = spatial(None, pos["r_phi"], s)
```

`W_V` (the value projection) is always learnable. These flags apply
identically to `SpatialAttentionSDPA`.

## Flash Attention / SDPA

For high-throughput training, swap in `SpatialAttentionSDPA`, which
expresses the same logit as a single `Q·K^T` and runs the
softmax-matmul step through
`torch.nn.functional.scaled_dot_product_attention`. The state dict is
identical to `SpatialAttention`, so you can train one and serve with the
other.

```python
from bloom_arch.attention import SpatialAttentionSDPA

# pad_v_for_flash=True zero-pads V to match QK head dim so PyTorch's
# flash-attention backend will accept the call. Mathematically inert.
fast = SpatialAttentionSDPA(
    d_s=d_s, d_v=d_v, H=H_s, K=K, lambdas=encoder.fourier.lambdas,
    target_scales=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
    pad_v_for_flash=True,
).cuda().to(torch.bfloat16)
```

On the L4 (sm_89, PyTorch 2.6+cu124), `pad_v_for_flash=True` + bf16/fp16
dispatches to FLASH_ATTENTION. Without padding, the call falls back to
EFFICIENT_ATTENTION (still fast; just not the flash kernel). Reproduce
with `python scripts/check_sdpa_backends.py`.

## Multi-scale kernel sweep

`scripts/plot_kernel_scales.py` plots the per-head same-chromosome
kernel `K_h(Δ) = β_h² + (1/K) Σ_k w_{h,k}² cos(2π Δ / λ_k)` across
`Δ ∈ [1 bp, 100 Mb]`. The saved figure
(`docs/figures/kernel_sweep.png`) shows clean log-Gaussian decay around
each head's target scale, with half-decay distances ordered from 23 bp
(L=100 bp) to 2 Mb (L=10 Mb).

## Layout

```
src/bloom_arch/
├── posenc/
│   ├── chrom_codes.py    # ChromCodeBook (deterministic BLAKE2b -> r_g, u_g)
│   ├── fourier.py        # FourierCoordEncoder (multiscale sin/cos bank)
│   └── encoder.py        # ChromLocalPosEncoder (composes the two)
└── attention/
    ├── content.py        # ContentAttention (standard MHA over s)
    ├── spatial.py        # SpatialAttention (kernel from positions)
    ├── spatial_sdpa.py   # SpatialAttentionSDPA (SDPA / Flash variant)
    └── typed_block.py    # TypedMultiHeadAttention (content + spatial)
tests/                    # 32 tests — spec section 11 + section 12 + SDPA
scripts/
├── plot_kernel_scales.py
└── check_sdpa_backends.py
docs/
├── spec.md
├── implementation.md     # math, parameter ranges, tuning notes
└── figures/kernel_sweep.png
```

## Install & test

```bash
# Inside the bloom-arch conda env
pip install -e ".[dev]"
pytest -v tests
```

32 tests cover spec section 11 (determinism, sign cancellation,
cross-chrom decorrelation, head ordering, permutation equivariance),
section 12 (wavelength tying, per-head scale ordering), the SDPA path
(fp32 parity, bf16 CUDA smoke, flash dispatch under padding), and
cross-device construction.
