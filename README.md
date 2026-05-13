# bloom-arch

Transformer model components for genomic sequence-to-function modeling
(ATAC-seq focus).

## Install

```bash
# create env (one-time)
conda create -n bloom-arch python=3.12 -y
$HOME/miniconda3/envs/bloom-arch/bin/pip install -e ".[dev]"
```

## Layout

```
src/bloom_arch/   library code (importable as `bloom_arch`)
tests/            pytest suite
scripts/          one-off / experiment driver scripts
docs/             design notes
```

## Components

`bloom_arch` currently implements the **chromosome-local positional encoding**
and **typed multi-head attention** components from `docs/spec.md`.

### Positional encoding (`bloom_arch.posenc`)

`ChromLocalPosEncoder(d_u=32, K=112, lambda_min=10, lambda_max=1e8)` is a
parameter-free module that returns two channels per feature:

| key       | shape       | description                                   |
| --------- | ----------- | --------------------------------------------- |
| `u`       | [B, N, d_u] | u_g — same-chromosome baseline code (spec §8) |
| `r_phi`   | [B, N, 2K]  | r_g ⊙ φ(x) — sign-gated Fourier coordinate    |

Chromosome codes `r_g, u_g` are derived deterministically from
`(species, assembly, chrom)` via BLAKE2b; the codebook generalizes to unseen
species at test time without any state transfer.

### Attention (`bloom_arch.attention`)

- `ContentAttention` — standard MHA on the sequence embedding (spec §5.1).
- `SpatialAttention` — uses positional Q,K and sequence V (spec §5.2),
  with **per-head log-Gaussian wavelength weights** initialized at target
  scales (spec §12.3). Each head learns `β_h`, `τ_h`, and a non-negative
  per-wavelength weight vector `w_h`.
- `TypedMultiHeadAttention` — runs the two head families in parallel.

### Validation

All 21 tests in `tests/` pass — see `pytest.log`. Tests cover every checkpoint
in spec §11.1–§11.6 plus the §12 multi-scale guarantees and parameter-free
encoder guards:

- determinism of `r_g`, `u_g` (§11.1)
- Fourier translation invariance (§6.1) at float64 precision and short-range float32
- same-chrom sign cancellation (§11.2)
- cross-chrom decorrelation: mean ≈ 0, variance shrinks with K (§11.3)
- spatial heads prefer near-same > far-same > cross-chrom-near-coord (§11.4)
- permutation equivariance pre-pool, invariance post-pool (§11.5)
- novel `(species, assembly, chrom)` works (§11.6)
- encoder has no learnable parameters; gradients reach `τ`, `β`, `w`, `W_V` only

### Multi-scale kernel sweep

`scripts/plot_kernel_scales.py` initializes a 6-head `SpatialAttention` at
target scales `{100 bp, 1 kb, 10 kb, 100 kb, 1 Mb, 10 Mb}` and sweeps the
same-chromosome kernel `K_h(Δ)` across Δ ∈ [1 bp, 100 Mb]. The resulting
`docs/figures/kernel_sweep.png` shows clean log-Gaussian decay around each
head's target scale; half-decay distances (printed by the script) are:

```
head 0: target L=100 bp  half-Δ= 23 bp
head 1: target L=  1 kb  half-Δ=224 bp
head 2: target L= 10 kb  half-Δ=  2 kb
head 3: target L=100 kb  half-Δ= 23 kb
head 4: target L=  1 Mb  half-Δ=227 kb
head 5: target L= 10 Mb  half-Δ=  2 Mb
```

### Tree

```
src/bloom_arch/
├── posenc/
│   ├── chrom_codes.py    # ChromCodeBook (BLAKE2b → r_g, u_g; cached)
│   ├── fourier.py        # FourierCoordEncoder (log-spaced 1/√K bank)
│   └── encoder.py        # ChromLocalPosEncoder (composes the two)
└── attention/
    ├── spatial.py        # SpatialAttention (per-head log-w windows, β_h, τ_h)
    ├── content.py        # ContentAttention (standard MHA)
    └── typed_block.py    # TypedMultiHeadAttention (content + spatial in parallel)
```

## SDPA / Flash Attention path

`SpatialAttentionSDPA` is a sibling of `SpatialAttention` that maps the
section-12 logit onto `Q·K^T` and runs the softmax-matmul step through
`torch.nn.functional.scaled_dot_product_attention`. State dict round-trips
with the math version, so you can train one and infer with the other.

**Backend dispatch on NVIDIA L4 (sm_89, PyTorch 2.6+cu124):**

| backend | bf16 | fp16 | fp32 |
| --- | --- | --- | --- |
| FLASH_ATTENTION | refused | refused | refused (by design) |
| EFFICIENT_ATTENTION (xFormers-lineage) | **dispatches** | **dispatches** | dispatches |
| CUDNN_ATTENTION | refused | refused | refused |
| MATH | works | works | works |

Reproduce with `python scripts/check_sdpa_backends.py`.

PyTorch's flash kernel rejects our shapes for two reasons with spec defaults
(`d_u=32`, `K=112`, `d_v=64`):

1. Flash requires `head_dim(Q) == head_dim(K) == head_dim(V)`. Ours are
   `(256, 256, 64)`.
2. Some flash paths cap head dim at 128. Our QK head dim is 256.

In practice EFFICIENT_ATTENTION is the path that dispatches and is the
"flash-class" kernel for asymmetric Q/V head dims — it provides the same
IO-aware tiling and is what our bf16 tests transparently exercise.

To force a true flash dispatch you would need to either (a) pad V to match
QK head dim, or (b) shrink K so QK head dim ≤ 128. Neither is needed today;
`SpatialAttentionSDPA` is bf16/fp16-ready as-is.
