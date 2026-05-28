# Implementation notes

This document covers the architecture of the `BloomAttentionBlock` module,
the mathematical structure of its positional encoding, the reasoning behind
each design choice, sensible defaults, and ranges to constrain learnable
parameters during training.

Cross-references in parentheses (e.g. `spec §3`) point to `docs/spec.md`.

---

## 1. Architecture overview

`BloomAttentionBlock` is a single `nn.Module` that takes four tensors —
sequence encoding `n`, genomic coordinates `x`, integer genome index `g`,
and value encoding `m` — and produces an output of dimension `d_out`.

```
Inputs: n [B, L, d_seq], x [B, L], g [B, L], m [B, L, d_val]

                 ┌──────────────┐
      x, g  ───►│  PosEncoder  │───► u [B, L, d_u], r_phi [B, L, 2K]
                 └──────────────┘
                        │
     s = [n; m]         │         f = [n; u; r_phi; m]
         │              │              │
         ▼              ▼              ▼
  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
  │ SequenceHeads│ │ PositionHeads│ │ StandardHeads│
  └──────┬──────┘ └──────┬───────┘ └──────┬───────┘
         │               │                │
         └───────────────┼────────────────┘
                         │ concat along dim=-1
                         ▼
                    ┌──────────┐
                    │   W_O    │ Linear(H_total * d_v, d_out)
                    └────┬─────┘
                         ▼
                   out [B, L, d_out]
```

All three head types use `F.scaled_dot_product_attention` for flash/efficient
kernel dispatch on GPU.

---

## 2. Module structure

```
src/bloom_arch/
  posenc/
    fourier.py           -- FourierCoordEncoder (log-spaced sin/cos)
    genome_codebook.py   -- GenomeCodebook (deterministic u_g, r_g buffers)
    encoder.py           -- PosEncoder (composes codebook + Fourier)
  attention/
    sequence_heads.py    -- SequenceHeads (Q,K from n; V from s)
    position_heads.py    -- PositionHeads (Q=K from pos encoding; V from s)
    standard_heads.py    -- StandardHeads (Q,K,V from full feature)
    block.py             -- BloomAttentionBlock (top-level)
```

---

## 3. The positional encoding

A feature $i$ has a coordinate $x_i$ in bp and an integer chromosome index
$g_i \in [0, \text{max\_G})$. The dataloading pipeline is responsible for
mapping `(species, assembly, chrom)` strings to integer indices.

### 3.1 Two channels with disjoint roles

The positional information splits into two parts (spec §3):

- **$u_{g_i}$ — the chromosome baseline code.** Fixed unit-norm vector per
  genome index. Same-chromosome features share $u_g$, so $u_i \cdot u_j = 1$
  on-chromosome and $\approx 0$ off-chromosome (random unit vectors in
  high-dim space are nearly orthogonal).
- **$r_{g_i} \odot \phi(x_i)$ — the sign-gated Fourier encoding.** $\phi(x_i)$
  is a multiscale sinusoidal embedding of the coordinate; $r_g \in \{-1, +1\}^{2K}$
  is a per-chromosome random sign vector that **cancels cross-chromosome
  contributions**.

The dot product splits along these channels:

$$
p_i \cdot p_j = \beta^2 \cdot u_i \cdot u_j + (r_i \odot \phi_i) \cdot (r_j \odot \phi_j)
$$

If features are on the same chromosome, $r_i = r_j$ cancels and the second
term reduces to $\phi(x_i) \cdot \phi(x_j)$, a function of coordinate
distance only. Cross-chromosome features have both terms approximately zero.

### 3.2 Integer-indexed deterministic codebook

`GenomeCodebook` pre-generates all codes at init time using a seeded
`torch.Generator` with `manual_seed(seed + g)` per row. This gives:

- **Determinism**: same seed always produces identical codes.
- **Extensibility**: increasing `max_G` preserves existing codes.
- **No parameters**: codes are stored as non-learnable buffers.
- **GPU-first**: forward is a single buffer index operation.

### 3.3 Multiscale Fourier features

$\phi(x) \in \mathbb{R}^{2K}$ packs sin/cos pairs at log-spaced wavelengths
$\lambda_k \in [\lambda_{min}, \lambda_{max}]$ (spec §6):

$$
\phi(x)[2k] = \sin(2\pi x / \lambda_k) / \sqrt{K}, \quad
\phi(x)[2k+1] = \cos(2\pi x / \lambda_k) / \sqrt{K}
$$

The $1/\sqrt{K}$ normalization bounds the dot product:

$$
\phi(x_i) \cdot \phi(x_j) = (1/K) \sum_k \cos(2\pi (x_i - x_j) / \lambda_k)
$$

This is translation-invariant, peaks at 1 for $\Delta = 0$, and oscillates
around 0 as $\Delta$ grows.

---

## 4. Head types

### 4.1 Sequence heads (spec §3.1)

Q and K are projected from the sequence encoding $n_i$ (from a DNA language
model). V is projected from the content encoding $s_i = [n_i; m_i]$.
Scaling is $1/\sqrt{d_{seq}}$ (the sequence encoding input dimension),
passed explicitly to SDPA.

### 4.2 Position heads (spec §3.2, §4.2)

Q = K = $\sqrt{\tau_h} \cdot [\beta_h \cdot u_{g_i};\; w_h^{tiled} \odot r\_phi_i]$

This construction makes $Q_i \cdot K_j$ reproduce the spec logit formula
exactly. SDPA is called with `scale=1.0` to disable the default $1/\sqrt{d_k}$
factor — $\tau_h$ provides the only scaling.

V is projected from the content encoding $s_i$.

### 4.3 Standard heads (spec §3, type 3)

Q, K, and V are all projected from the full feature encoding
$f_i = [n_i; \beta \cdot u_{g_i}; r\_phi_i; m_i]$. Standard SDPA scaling applies.
The block maintains a shared learnable $\beta$ for constructing $p_i$ in the
standard heads input (separate from the per-head $\beta_h$ in position heads).

---

## 5. Per-head scale selection

Each position head $h$ carries a nonneg weight vector $w_h \in \mathbb{R}^K_{>0}$,
and its kernel is:

$$
K_h(\Delta) = \beta_h^2 + (1/K) \sum_k w_{h,k}^2 \cdot \cos(2\pi \Delta / \lambda_k)
$$

### 5.1 Log-Gaussian initialization

$$
\log w_{h,k} = -\frac{(\log \lambda_k - \log L_h)^2}{2\rho^2}
$$

Normalized so $\|w_h\|^2 = K$. With $\rho = \log(10)/2 \approx 1.15$,
one standard deviation spans a factor of $\sqrt{10}$ in wavelength.

### 5.2 Tying sin/cos

The wavelength weight applies to both sin and cos of the same frequency.
`log_w` is stored as $[H, K]$ and tiled to $[H, 2K]$ via
`repeat_interleave(2, dim=-1)`.

### 5.3 Log-space parameterization

Storing $\log(w_{hk})$ rather than $w_{hk}$:
1. Nonnegativity is free ($w = \exp(\log w) > 0$).
2. Gradient geometry matches Adam's additive updates.

**Note:** `learn_w=True` is not yet implemented. Making $w_h$ learnable
requires reparameterization to maintain the $\|w_h\|^2 = K$ constraint and
non-negativity during training. Currently `log_w` is a fixed buffer.

### 5.4 Softmax temperature $\tau_h$

Per-head $\tau_h > 0$ (stored as `log_tau`) controls attention sharpness.
Parameterized in log-space to maintain positivity. Learnable when
`learn_tau=True`; held fixed at init value otherwise (default).

---

## 6. Q.K^T mapping (the SDPA / Flash path)

Construct:

$$
Q_i^{(h)} = \sqrt{\tau_h} \cdot [\beta_h u_i;\; w_h^{tiled} \odot r\_phi_i]
$$

with $\text{head\_dim} = d_u + 2K$. Then $Q_i \cdot K_j = \ell_{ij}^{(h)}$
exactly. Passing `scale=1.0` to SDPA disables its default $1/\sqrt{d_k}$.

### 6.1 Padding V for flash dispatch

PyTorch's flash-attention backend requires `head_dim(Q) == head_dim(V)`.
Position heads have QK dim $d_u + 2K$ but V dim $d_v$. Setting
`pad_v_for_flash=True` zero-pads V to QK dim, calls SDPA, then slices
back. Mathematically inert — verified by `test_pad_v_parity`.

---

## 7. Default values and meaningful ranges

### 7.1 Encoder (no learnable parameters)

| Symbol | Default | Meaning | Range / Notes |
|--------|---------|---------|---------------|
| `max_G` | - | Max genome/chromosome IDs | Set to your dataset's unique count |
| `d_u` | 32 | Chromosome baseline dim | 16-128. Larger → cleaner orthogonality |
| `K` | 112 | Number of Fourier wavelengths | 32-256. Cross-chrom variance ~ 1/K |
| `lambda_min` | 10 bp | Shortest wavelength | 1-100 bp |
| `lambda_max` | 100 Mb | Longest wavelength | 10 Mb - 1 Gb |
| `seed` | 42 | Codebook seed | Change to get different codes |

### 7.2 Position heads (per-head learnables)

| Symbol | Default | Meaning | Sensible range |
|--------|---------|---------|----------------|
| `n_position_heads` | (required) | Number of position heads | One per scale of interest |
| `target_scales` | log-spaced | Initial center wavelength | Within [10·λ_min, λ_max/10] |
| `rho` | log(10)/2 | Log-Gaussian width | 0.4-2.5 |
| `beta_init` | 0.0 | Initial floor β | β² ranges 0-2 in practice |
| `tau_init` | 1.0 | Initial temperature | τ typically 0.1-10 |
| `learn_tau` | False | If True, tau is learnable | Set True to learn sharpness |
| `learn_w` | False | Reserved (not yet implemented) | Raises NotImplementedError if True |
| `d_v` | 64 | Per-head value dim | 32-128 |

### 7.3 Recommended training-time constraints

- **`log_w`**: weight decay 1e-3 to 1e-2.
- **`beta`**: clamp |β| ≤ 5.
- **`log_tau`**: clamp ∈ [-4, 4] (τ ∈ [0.018, 55]).

### 7.4 Numerical precision

Coordinates in base pairs. Float32 has ~7 digits of mantissa, so at
$x \sim 2 \cdot 10^8$ and $\lambda_{min} = 10$, the phase is at the edge
of float32 precision. The Fourier encoder computes phases in float32;
downstream attention operates in bf16/fp16 where values are bounded.

---

## 8. Design omissions

Three deliberate non-choices:

- **No FFN / LayerNorm / residual** in `BloomAttentionBlock`. It outputs
  after W_O; wrapping with transformer machinery is the caller's job.
- **No untied positional Q/K projections** (spec §5). Position heads keep
  Q = K — they represent the position of the same feature.
- **No random Fourier features mode.** The cosine bank is chosen for exact
  same-chrom cancellation and clean dot-product identity.
