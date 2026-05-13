# Implementation notes

This document covers the mathematical structure of the chromosome-local
positional encoding and typed attention block, the reasoning behind each
design choice, sensible defaults, and ranges to constrain learnable
parameters to during training.

Cross-references in parentheses (e.g. `spec §3`) point to `docs/spec.md`.

---

## 1. The positional encoding

A feature $i$ has a coordinate $x_i$ in bp and a chromosome key
$g(i) = (species, assembly, chrom)$. We want a vector $p_i ∈ R^{d_p}$
whose dot products $p_i · p_j$ encode meaningful position-aware
similarity for the model's attention weights.

### 1.1 Two channels with disjoint roles

$p_i$ splits cleanly into two parts (spec §3):

$$
p_i = [\, \underbrace{β \cdot u_{g(i)}}_{d_u}\ ;\ \underbrace{r_{g(i)} \odot φ(x_i)}_{2K} \,]
$$

- **$u_g$ — the chromosome baseline code.** Fixed unit vector per
  chromosome key. Same-chromosome features share $u_g$, so $u_i · u_j = 1$
  on-chromosome and $≈ 0$ off-chromosome (random unit vectors in
  high-dim space are nearly orthogonal).
- **$r_g ⊙ φ(x_i)$ — the sign-gated Fourier encoding.** $φ(x_i)$ is a
  multiscale sinusoidal embedding of the coordinate; $r_g ∈ {±1}^{2K}$ is
  a per-chromosome random sign vector that **cancels cross-chromosome
  contributions**.

The dot product splits along these channels:

$$
p_i · p_j = β² · u_i·u_j + (r_i ⊙ φ_i) · (r_j ⊙ φ_j)
$$

Why this split? If features $i$ and $j$ are on the same chromosome,
$u_i = u_j$ and $r_i = r_j$, so the sign-gating cancels ($r ⊙ r = 1$)
and the second term reduces to $φ(x_i) · φ(x_j)$, a function of the
coordinate difference. If they are on *different* chromosomes, $u_i · u_j ≈ 0$
and the second term has independent random signs, so it is approximately
zero by Hoeffding's bound (variance shrinks as $1/K$). Cross-chromosome
features are decorrelated by construction, regardless of how close their
coordinates happen to be.

### 1.2 Determinism, not learned

$u_g$ and $r_g$ are **deterministic functions of the chromosome key**,
not learned tables (spec §2, §9). Two reasons:

1. **Generalization to unseen chromosomes/species.** A model trained on
   mouse mm10 will produce stable, well-separated codes for human hg38
   chromosomes at inference time — they were never in the training set
   but `BLAKE2b(salt | "homo_sapiens" | "hg38" | "chr1")` produces the
   same code today as it will on any machine, ever.
2. **No parameter cost.** Genome-scale work touches thousands of
   contigs; a learned table would carry orders of magnitude more
   parameters than the rest of this block.

The salt is versioned (`"posenc_v1"`) so we can change the hash
without silently invalidating an old checkpoint.

### 1.3 Multiscale Fourier features

$φ(x) ∈ R^{2K}$ packs sin/cos pairs at log-spaced wavelengths
$λ_k ∈ [λ_min, λ_max]$ (spec §6):

$$
φ(x)[2k]   = sin(2π x / λ_k) / √K \\
φ(x)[2k+1] = cos(2π x / λ_k) / √K
$$

The $1/√K$ normalization makes the dot product cleanly bounded:

$$
φ(x_i) · φ(x_j) = (1/K) Σ_k cos(2π (x_i - x_j) / λ_k)
$$

This is translation-invariant (depends only on $Δ = x_i − x_j$), peaks
at `1` for $Δ = 0$, and oscillates around `0` as $Δ$ grows. The number
of wavelengths $K$ controls (a) how densely the log-frequency range is
sampled, and (b) the variance of cross-chromosome dot products
($~ 1/K$ from the central limit theorem; verified by test T4).

---

## 2. Per-head scale selection (spec §12)

The unweighted kernel $(1/K) Σ_k cos(2π Δ / λ_k)$ is broadband — it
sees all scales at once. A single head treating "5 bp away" and
"5 Mb away" as both "same chromosome" is rarely what we want for
genomic problems, where regulatory grammar lives at ~10 bp, enhancer
contacts at ~10 kb, and TADs at ~1 Mb.

Per-head wavelength weighting fixes this. Each spatial head $h$ carries
a nonnegative weight vector $w_h ∈ R^K_{>0}$, and its kernel is

$$
K_h(Δ) = β_h² + (1/K) Σ_k w_{h,k}² · cos(2π Δ / λ_k)
$$

$w_{h,k}$ shapes how much wavelength $λ_k$ contributes to head $h$'s
attention. Initializing $w_h$ as a log-Gaussian centered at a target
wavelength $L_h$ gives the head a tunable band-pass character around
that scale.

### 2.1 Log-Gaussian initialization

$$
log w_{h,k} = -((log λ_k - log L_h)²) / (2 ρ²)
$$

so $w_h$ is a Gaussian over log-wavelengths centered at $L_h$ with
width $ρ$ (in nats). With $ρ = log(10)/2 ≈ 1.15$, one standard
deviation is a factor of $√10 ≈ 3.16×$ in wavelength; the full
width at half maximum is roughly one decade. Heads with this width
produce kernels that drop by 50% within roughly one decade of their
target scale, which is the regime shown in
`docs/figures/kernel_sweep.png`.

### 2.2 Tying sin/cos

The wavelength weight applies to *both* the sin and cos components of
the same wavelength — they are two views of the same frequency. The
implementation stores `log_w` of shape $[H, K]$ and tiles to
$[H, 2K]$ via `repeat_interleave(2, dim=-1)` on every forward.
Test T11 (`test_wavelength_weights_tied_across_sin_cos`) guards this
invariant.

### 2.3 Why log-space?

Storing $log w_{h,k}$ rather than $w_{h,k}$ has two benefits:

1. **Nonnegativity is free.** $w = exp(log_w)$ is positive for any real
   `log_w`. If $w$ were stored directly, training could push it
   negative and invert the kernel's sign — meaningful only by accident.
2. **Gradient geometry.** A multiplicative change in $w$ corresponds to
   an additive change in `log_w`. The optimizer (Adam, AdamW, etc.) is
   well-suited to additive steps; "shrink this head's weight by 30%"
   becomes a $log(0.7) ≈ -0.36$ step on `log_w`, irrespective of the
   current value.

### 2.4 Same-chromosome baseline $β_h$

$β_h²$ is the **floor** of $K_h(Δ)$ — the value the kernel decays
toward at large $Δ$ on the same chromosome (spec §8). It pulls all
same-chromosome features toward each other in attention space,
regardless of distance, by an amount controlled by $β_h$. Off-
chromosome features have $u_i · u_j ≈ 0$, so they never see this
floor.

$β_h$ is a learned per-head scalar. Initialize moderately
(`beta_init=1.0` → floor of `1.0`; values in the kernel-sweep figure
were generated with `beta_init=0.3` → floor of `0.09`, which keeps
the visualization legible). Over training, $β_h$ can rise (head leans
more heavily on chromosome identity than coordinate) or fall (head
relies almost entirely on coordinate). Spec §8 anticipates $β$ shrinking
toward zero over the course of learning as the model learns more
informative within-chromosome structure.

### 2.5 Softmax temperature $τ_h$

The pre-softmax logit is

$$
ℓ_{ij}^{(h)} = τ_h · ( β_h² u_i·u_j + Σ_d w_tiled[h,d]² · r_phi_i[d] · r_phi_j[d] )
$$

Per-head learnable $τ_h > 0$ controls **how sharply** the kernel
differences translate into attention mass:

- $τ_h$ large → softmax is sharp, attention concentrates on the few
  features with the highest kernel value.
- $τ_h$ small → softmax is diffuse, attention smears across many
  features.
- $τ_h → 0$ → uniform attention (kernel is ignored).

Like $w_h$, $τ_h$ is parameterized in log-space (`log_tau`) to keep it
positive.

There is a known redundancy: doubling $w_h$ and quartering $τ_h$
leaves the logit unchanged (logits are linear in $τ$ and quadratic in
$w$). We break the redundancy by initializing $w_h$ from the
log-Gaussian ($w_h ~ O(1)$) and $τ_h = 1$. Training is free to move
both, and L2 regularization on `log_w` + `log_tau` will share the
load between them.

---

## 3. Q.K^T mapping (the SDPA / Flash path)

`SpatialAttentionSDPA` rewrites the section-12 logit as a single
inner product so it fits PyTorch's `scaled_dot_product_attention`
API. Construct

$$
Q_i^(h) = √τ_h · [ β_h u_i ; w_tiled^(h) ⊙ r_phi_i ] \\
K_j^(h) = √τ_h · [ β_h u_j ; w_tiled^(h) ⊙ r_phi_j ]
$$

with $head_dim = d_u + 2K$. Then $Q_i^(h) · K_j^(h) = ℓ_{ij}^(h)$
exactly. Passing $scale=1.0$ to SDPA disables its default $1/√d_k$
factor — $τ_h$ is the only scaling we want.

### 3.1 Padding V for flash dispatch

PyTorch's flash-attention backend requires
`head_dim(Q) == head_dim(K) == head_dim(V)`. Our QK head dim is
$d_u + 2K$ (256 with defaults); our V head dim is $d_v$ (typically
64). The asymmetry disqualifies flash.

The fix is to zero-pad V along the head dim up to `head_dim_qk`,
call SDPA, then slice the output back to $d_v$. Padding with zeros
is mathematically inert under the softmax-weighted sum SDPA
performs — the padded V columns produce padded output columns,
which we discard. Cost is a ~4× larger V tile per forward; no extra
parameters; no change to gradients.

Set `pad_v_for_flash=True` to enable. On NVIDIA L4 (sm_89, PyTorch
2.6+cu124), this is what causes FLASH_ATTENTION to dispatch in
bf16/fp16. The EFFICIENT_ATTENTION backend works regardless of
padding.

---

## 4. Default values and meaningful ranges

The defaults below are spec section 10's converged hyperparameters,
chosen for typical genome-scale features. Anything marked
*learnable* is also free to train; the **range** column gives values
that have a sensible biological or numerical interpretation. Values
outside these ranges are not forbidden but should prompt a
double-check.

### 4.1 Encoder (no learnable parameters)

| symbol      | default     | meaning                                     | range / notes |
|---|---|---|---|
| $d_u$       | 32          | chromosome baseline dim                     | 16-128. Larger → cleaner cross-chrom orthogonality but cheap; 32 is plenty for ~50 contigs/species. |
| $K$         | 112         | number of Fourier wavelengths               | 32-256. Cross-chrom dot product variance ~ 1/K. K=32 has visible noise; K=112 is well into the asymptotic regime. |
| $λ_min$     | 10 bp       | shortest wavelength                         | 1-100 bp. Bounded below by float32 precision at large coords. |
| $λ_max$     | 100 Mb      | longest wavelength                          | 10 Mb - 1 Gb. Set to roughly chromosome size for your organism. |
| $salt$      | "posenc_v1" | hash salt                                   | Bump when changing the hash function or its inputs. |

### 4.2 SpatialAttention (per-head learnables)

| symbol      | default                   | meaning                                            | sensible range after training |
|---|---|---|---|
| $H$         | 4-6                       | number of spatial heads                            | One per scale of interest. |
| `target_scales` | e.g. [100 bp ... 10 Mb] geometric | initial center wavelength $L_h$            | Within $[10·λ_min, λ_max/10]$. Outside this, the log-Gaussian falls onto fewer than two wavelengths and the head becomes near-flat. |
| $rho$       | $log(10)/2 ≈ 1.15$        | log-Gaussian width                                 | 0.4-2.5 (0.2-1 decade FWHM). Narrower → sharper scale selectivity but less robustness. |
| `beta_init` | 1.0                       | initial floor $β_h$                                | $β_h^2$ ranges 0-2 in practice. > ~2 means the head is mostly chromosome-presence regardless of coordinate. |
| `tau_init`  | 1.0                       | initial softmax temperature                        | $τ_h$ typically 0.1-10. > ~50 will saturate softmax; < ~0.01 yields near-uniform attention. |
| $d_v$       | 64                        | per-head value dim                                 | 32-128 typical. 64 keeps SDPA's V-padded mode at the 256 flash head dim limit with K=112. |

### 4.3 Recommended training-time constraints

These are not hard limits but make optimization better behaved:

- **`log_w`**: weight decay `1e-3` to `1e-2` keeps $w_h$ from drifting
  to extreme magnitudes that would dominate $τ_h$.
- **$beta$**: clamp $|β_h| ≤ 5$ if training pushes it large. The
  floor scales as $β²$, so values much above 5 effectively saturate
  attention to "same chromosome" regardless of $w_h$.
- **`log_tau`**: clamp $log_tau ∈ [-4, 4]$ (i.e. $τ_h ∈ [0.018, 55]$).
  Above ~50, attention is one-hot; below ~0.02, it is uniform — both
  regimes kill gradients for the rest of the head.

These can be implemented as `torch.clamp_` in a post-step hook, or as
soft penalties via weight decay on the log-parameters.

### 4.4 Numerical precision

$coords$ is in base pairs. Float32 has ~7 decimal digits of mantissa,
so at $x ~ 2 · 10^8$ (typical chromosome length) and $λ_min = 10$,
the phase $2π x / λ_min ≈ 1.3 · 10^8$ rad is at the edge of float32
precision. Two consequences:

1. **Tests use float64 for end-to-end translation invariance.**
   Test `test_fourier_dot_depends_only_on_distance` runs in
   `torch.float64`; a separate short-range float32 test
   ($λ_min = 100$, $x < 10^6$) covers the model's training-time regime.
2. **bf16/fp16 forward is fine.** The phases are computed in float32
   on the buffer side; bf16 is used downstream in the matmul/softmax,
   where the relevant numbers are bounded $|·| ≤ 1$ per cosine.

If you ever need precise behavior at the very ends of a chromosome
with $λ_min < 10$, run the encoder in float64 and cast to bf16 only
after $r_phi$ is constructed.

---

## 5. What this design omits

Three deliberate non-choices, in case the next iteration revisits them:

- **No FFN / LayerNorm / residual** in `TypedMultiHeadAttention`. It
  is just $Content + Spatial → concat → W_O$. Wrapping it with the
  usual transformer machinery is the caller's job; the pure block
  makes the attention behavior testable at initialization.
- **No untied positional Q/K projections** (spec §5). v1 keeps Q and
  K equal — they are the *position* of the same feature, so untying
  would invite a learned drift away from the spec's symmetry. A
  future variant might add a per-head learned mixing of `(u, r_phi)`
  before the SDPA call.
- **No random-Fourier-features mode.** Spec §12.5 notes that the
  cosine bank is oscillatory; a strictly monotone-decay kernel would
  use Random Fourier Features against an RBF target. We chose the
  cosine bank for cleanliness — exact same-chrom cancellation,
  trivial dot-product identity — and accept that distant features
  see a kernel with small oscillations rather than a clean
  exponential decay.
