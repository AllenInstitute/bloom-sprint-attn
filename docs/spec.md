# Spec: Chromosome-Local Positional Encoding for Genome-Wide Feature Bags

## 1. Problem summary

We have a transformer model that receives a bag of genome-derived features per example/cell. Each feature corresponds to an underlying DNA sequence and genomic coordinate. The input order is arbitrary and should not carry meaning; the model ultimately pools over features, so the architecture should behave as permutation-invariant or permutation-equivariant before pooling.

Each feature should carry two kinds of information:

1. **Sequence/content information**
   Encoded from the underlying nucleotide sequence using a DNA sequence encoder.

2. **Chromosome-local spatial information**
   Encoded from the feature's chromosome/contig identity and coordinate, so that position-based attention can preferentially aggregate information from nearby features on the same chromosome.

The goal is to let some attention heads use genomic position to determine attention weights, while using sequence embeddings as the values being aggregated:

$$
\text{position determines attention weights; sequence provides values.}
$$

We want to avoid supplying an explicit pairwise genomic distance matrix or an explicit pairwise same-chromosome mask.

---

## 2. Constraints and design considerations

### 2.1 Input structure

* Each example/cell contains a sampled bag of genomic features.
* Features are genome-wide and may come from many chromosomes.
* Input feature order is arbitrary and nonsensical.
* The transformer should not rely on feature index as a positional signal.

### 2.2 Biological/geometric constraints

* Genomic distance is meaningful primarily within a chromosome or contig.
* Absolute distance/proximity is sufficient (i.e. no directionality).
* Cross-chromosome coordinate similarity should not induce false proximity. For example, `chr1:10Mb` and `chr2:10Mb` should not be treated as genomically close.
* The model should support multiple species, potentially with different chromosome counts and chromosome names.
* We do not have human data for training, but we want eventual generalization to human.
* Cross-species biological associations should primarily be learned through sequence/content embeddings, not through learned chromosome identity or synteny priors.

### 2.3 Modeling constraints

* Avoid explicit pairwise distance matrices.
* Avoid explicit pairwise chromosome masks for position-derived attention.
* Avoid learned chromosome lookup tables that assume a fixed set of chromosomes.
* Use a dimensionality budget on the order of 512-1024 total feature dimensions, with a positional channel likely in the 128-512 range.
* Maintain a clean decomposition between sequence/content similarity and spatial proximity, at least in early layers.

---

## 3. Chosen high-level design

For each feature $i$, compute:

$$
s_i = \text{DNAEncoder}(\text{seq}_i)
$$

where $s_i$ is the sequence/content embedding.

Compute a chromosome-local positional encoding:

$$
p_i = [\beta u_{g(i)};\ r_{g(i)} \odot \phi(x_i)]
$$

where:

* $x_i$ is the coordinate of feature $i$ along its chromosome or contig.
* $g(i)$ is a chromosome/contig key, e.g. `(species, assembly, chromosome)`.
* $\phi(x_i)$ is a multiscale Fourier encoding of the coordinate.
* $r_{g(i)} \in {-1, +1}^{d_\phi}$ is a fixed deterministic random sign vector for chromosome/contig $g(i)$.
* $u_{g(i)} \in \mathbb{R}^{d_u}$ is a fixed deterministic random unit vector or sign-derived unit vector for chromosome/contig $g(i)$.
* $\beta$ controls the strength of same-chromosome preference.
* $\odot$ denotes elementwise multiplication.

The model then uses typed attention heads:

1. **Content heads**
   Attention weights are derived from sequence embeddings; values are sequence embeddings.

2. **Spatial heads**
   Attention weights are derived from positional encodings; values are sequence embeddings.

Optionally, later layers may include mixed heads that can use both sequence and position, but the first implementation should prioritize clean separation.

---

## 4. Why not use additive sequence + position embeddings?

A standard transformer might use:

$$
z_i = s_i + p_i
$$

This immediately mixes sequence and position. Attention logits become:

$$
(W_Q(s_i+p_i))^\top(W_K(s_j+p_j))
$$

which expands into sequence-sequence, position-position, and sequence-position cross terms. These cross terms may be useful later, but they make it harder to preserve a clean distinction between sequence similarity and spatial proximity.

Instead, we prefer to keep sequence and position as separate channels and define attention heads that explicitly choose which channel determines attention weights.

---

## 5. Attention head definitions

### 5.1 Content heads

Content heads use sequence embeddings for queries, keys, and values:

$$
q_i^{(h)} = W_Q^{(h)} s_i
$$

$$
k_j^{(h)} = W_K^{(h)} s_j
$$

$$
v_j^{(h)} = W_V^{(h)} s_j
$$

The attention logits are:

$$
\ell_{ij}^{(h)} = \frac{(q_i^{(h)})^\top k_j^{(h)}}{\sqrt{d_h}}
$$

These heads learn relationships among features based on sequence/regulatory content. They are also the primary path for cross-species biological generalization.

### 5.2 Spatial heads

Spatial heads use positional encodings for queries and keys, but sequence embeddings for values:

$$
q_i^{(h)} = p_i
$$

$$
k_j^{(h)} = p_j
$$

$$
v_j^{(h)} = W_V^{(h)} s_j
$$

A simple initial logit form is:

$$
\ell_{ij}^{(h)} = \tau_h \cdot p_i^\top p_j
$$

where $\tau_h$ is a learned scalar temperature per spatial head.

Then:

$$
A_{ij}^{(h)} = \text{softmax}_j(\ell_{ij}^{(h)})
$$

$$
o_i^{(h)} = \sum_j A_{ij}^{(h)} v_j^{(h)}
$$

This means position determines where each feature attends, while sequence/content determines what information is passed.

### 5.3 Optional spatial projections

If more flexibility is needed, spatial heads may use a tied projection:

$$
q_i^{(h)} = W_P^{(h)} p_i
$$

$$
k_j^{(h)} = W_P^{(h)} p_j
$$

Using the same projection for queries and keys better preserves the intended positional geometry than untied dense $W_Q$ and $W_K$ matrices.

For the first implementation, prefer either:

1. no projection, only learned temperature; or
2. tied projection initialized near identity.

Avoid untied dense positional Q/K projections in the first version, because they can weaken the intended same-chromosome geometric bias.

---

## 6. Multiscale Fourier coordinate encoding

For a coordinate $x$, define wavelengths:

$$
\lambda_1, \lambda_2, \dots, \lambda_K
$$

Then:

$$
\phi(x) = [\sin(2\pi x/\lambda_1),\ \cos(2\pi x/\lambda_1),\ \dots,\ \sin(2\pi x/\lambda_K),\ \cos(2\pi x/\lambda_K)]
$$

Thus:

$$
d_\phi = 2K
$$

The wavelengths should be log-spaced:

$$
\lambda_k = \lambda_{\min}\left(\frac{\lambda_{\max}}{\lambda_{\min}}\right)^{k/(K-1)}
$$

Recommended starting values:

$$
\lambda_{\min} = 10\ \text{bp}
$$

$$
\lambda_{\max} = 100\ \text{Mb} \quad \text{or} \quad 300\ \text{Mb}
$$

If $d_\phi = 224$, then $K=112$ wavelengths.

### 6.1 Dot-product property

For a single sine/cosine pair:

$$
\sin(2\pi x_i/\lambda)\sin(2\pi x_j/\lambda) +
\cos(2\pi x_i/\lambda)\cos(2\pi x_j/\lambda)
=
\cos(2\pi(x_i-x_j)/\lambda)
$$

Therefore, the dot product of two Fourier encodings depends on relative distance:

$$
\phi(x_i)^\top \phi(x_j) = \sum_k \cos(2\pi(x_i-x_j)/\lambda_k)
$$

This gives translationally invariant distance sensitivity across multiple genomic scales.

### 6.2 Normalization

Normalize the Fourier block so dot-product magnitudes are predictable. Possible choices:

$$
\phi(x) \leftarrow \frac{1}{\sqrt{K}}\phi(x)
$$

or normalize the final positional vector:

$$
p_i \leftarrow \frac{p_i}{|p_i|}
$$

For the first implementation, normalize $\phi(x)$ by $1/\sqrt{K}$, and use learned spatial-head temperatures $\tau_h$.

---

## 7. Chromosome-local sign-gated encoding

The sign-gated coordinate component is:

$$
r_g \odot \phi(x)
$$

where $r_g$ is a fixed deterministic random sign vector for chromosome/contig key $g$.

For two features on the same chromosome, $g_i=g_j=g$:

$$
(r_g \odot \phi(x_i))^\top(r_g \odot \phi(x_j))
= \phi(x_i)^\top \phi(x_j)
$$

The signs cancel, so within-chromosome coordinate geometry is preserved exactly.

For two features on different chromosomes, $g_i \neq g_j$:

$$
(r_{g_i} \odot \phi(x_i))^\top(r_{g_j} \odot \phi(x_j)) \approx 0
$$

because independent random sign vectors decorrelate the coordinate encodings.

This gives a soft implicit same-chromosome restriction without an explicit pairwise mask.

---

## 8. Role of the $\beta u_g$ term

The term $\beta u_g$ provides a same-chromosome baseline preference.

Without this term, cross-chromosome positional similarities are approximately mean-zero. However, in a softmax, zero logits can still receive substantial attention mass, especially when many cross-chromosome features are present.

Define:

$$
p_i = [\beta u_{g_i};\ r_{g_i}\odot\phi(x_i)]
$$

Then:

$$
p_i^\top p_j =
\beta^2 u_{g_i}^\top u_{g_j}
+
(r_{g_i}\odot\phi(x_i))^\top(r_{g_j}\odot\phi(x_j))
$$

For same chromosome:

$$
p_i^\top p_j = \beta^2 + \phi(x_i)^\top\phi(x_j)
$$

assuming $u_g$ is unit-normalized.

For different chromosomes:

$$
p_i^\top p_j \approx 0
$$

Thus:

* $r_g\odot\phi(x)$ encodes distance within a chromosome.
* $\beta u_g$ says whether two features are likely from the same chromosome.

The $\beta u_g$ term is optional but recommended if spatial heads show too much cross-chromosome attention leakage.

For the first implementation, include the term but make $\beta$ configurable. It can be:

1. a fixed scalar hyperparameter;
2. a learned global scalar;
3. a learned per-spatial-head scalar.

The preferred first version is a learned per-spatial-head scalar initialized to a moderate value.

---

## 9. Deterministic generation of chromosome codes

Both $r_g$ and $u_g$ should be fixed per chromosome/contig key. They should not be learned and should not be resampled per batch or epoch.

Use a deterministic hash-based generator:

$$
g = \text{species} \mid \text{assembly} \mid \text{chromosome/contig}
$$

$$
r_g = \text{HashToSigns}(g)
$$

$$
u_g = \text{HashToUnitVector}(g)
$$

Requirements:

* Same key always produces the same $r_g$ and $u_g$.
* Different chromosome/contig keys produce approximately independent vectors.
* The hash salt/version should be fixed for reproducibility.
* New species/chromosomes at test time can be handled by applying the same deterministic generation procedure.

Example conceptual pseudocode:

```python
def chrom_sign_vector(species, assembly, chrom, dim, salt="posenc_v1"):
    key = f"{salt}|{species}|{assembly}|{chrom}|sign"
    rng = seeded_rng_from_hash(key)
    return rng.choice([-1.0, 1.0], size=dim)


def chrom_unit_vector(species, assembly, chrom, dim, salt="posenc_v1"):
    key = f"{salt}|{species}|{assembly}|{chrom}|unit"
    rng = seeded_rng_from_hash(key)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)
```

Do not generate $r_g$ independently per feature. The same-chromosome cancellation property requires the same $r_g$ for all features on chromosome $g$.

---

## 10. Recommended first implementation

### 10.1 Dimensions

Assume a model width around 768, with sequence and position channels separated.

Example:

$$
s_i \in \mathbb{R}^{512}
$$

$$
p_i \in \mathbb{R}^{256}
$$

For $p_i$:

| Component             | Dimensions | Purpose                                        |
| --------------------- | ---------: | ---------------------------------------------- |
| $\beta u_g$           |         32 | Same-chromosome baseline preference            |
| $r_g \odot \phi(x_i)$ |        224 | Multiscale within-chromosome distance encoding |

Thus:

$$
d_u = 32
$$

$$
d_\phi = 224
$$

$$
K = 112
$$

### 10.2 Coordinate encoding

Use log-spaced wavelengths:

$$
\lambda_{\min}=10\ \text{bp}
$$

$$
\lambda_{\max}=100\ \text{Mb}
$$

or optionally:

$$
\lambda_{\max}=300\ \text{Mb}
$$

Normalize $\phi(x)$ by $1/\sqrt{K}$.

### 10.3 Spatial attention

Initial spatial-head logits:

$$
\ell_{ij}^{(h)} = \tau_h \cdot p_i^\top p_j
$$

Values:

$$
v_j^{(h)} = W_V^{(h)}s_j
$$

Output:

$$
o_i^{(h)} = \sum_j \text{softmax}_j(\ell_{ij}^{(h)})v_j^{(h)}
$$

Start with no learned positional Q/K projection. Add tied projection only if needed.

### 10.4 Content attention

Use standard sequence-derived attention heads in parallel:

$$
\ell_{ij}^{(h)} = \frac{(W_Q^{(h)}s_i)^\top(W_K^{(h)}s_j)}{\sqrt{d_h}}
$$

$$
v_j^{(h)} = W_V^{(h)}s_j
$$

These heads should be responsible for content-driven associations and cross-species generalization.

---

## 11. Implementation tests

The first implementation should include tests that verify the positional encoding behaves as intended before training.

### 11.1 Determinism test

For the same `(species, assembly, chrom)` key, repeated calls should produce identical $r_g$ and $u_g$.

Expected:

$$
r_g^{(1)} = r_g^{(2)}
$$

$$
u_g^{(1)} = u_g^{(2)}
$$

### 11.2 Same-chromosome Fourier preservation test

For two coordinates $x_i, x_j$ on the same chromosome:

$$
(r_g\odot\phi(x_i))^\top(r_g\odot\phi(x_j))
=
\phi(x_i)^\top\phi(x_j)
$$

Expected numerical error should be near floating-point tolerance.

### 11.3 Cross-chromosome decorrelation test

For different chromosome keys $g_a\neq g_b$, evaluate:

$$
(r_{g_a}\odot\phi(x_i))^\top(r_{g_b}\odot\phi(x_j))
$$

across many random coordinate pairs.

Expected:

* Mean should be near zero.
* Variance should decrease as $d_\phi$ increases.
* Cross-chromosome similarities should be much lower than nearby same-chromosome similarities after applying temperature and $\beta u_g$.

### 11.4 Same-chromosome attention preference test

Construct a synthetic bag with features from multiple chromosomes. For a query feature, include:

* nearby features on the same chromosome;
* far features on the same chromosome;
* features on different chromosomes with similar raw coordinates.

Expected for spatial heads:

1. nearby same-chromosome features receive high attention;
2. far same-chromosome features receive lower attention;
3. different-chromosome features do not receive high attention solely because their raw coordinates are similar.

### 11.5 Permutation equivariance test

Permute the order of input features and verify that pre-pooling outputs permute correspondingly, and pooled outputs are invariant.

Expected:

* Transformer outputs before pooling should be equivariant to input permutation.
* Final pooled representation should be invariant to input permutation.

### 11.6 New chromosome/species test

Create a chromosome key not seen during training or initialization, such as a synthetic species/assembly/chromosome string.

Expected:

* Encoding should be generated successfully.
* Same-key features preserve within-chromosome distance geometry.
* Different-key features are decorrelated from existing chromosomes.

---

## 12. Distance-decay control and spatial length scales

The spatial encoding should support different attention heads specializing to different genomic distance scales. The base Fourier encoding provides a bank of possible spatial wavelengths, but without head-specific weighting all spatial heads see the same multiscale positional kernel, differing only by temperature.

### 12.1 Base positional kernel

Ignoring the chromosome code term, the Fourier component gives the following same-chromosome identity:

```text
(r_g * phi(x_i)) dot (r_g * phi(x_j)) = phi(x_i) dot phi(x_j)
```

For the Fourier bank:

```text
phi(x_i) dot phi(x_j) = sum_k cos(2*pi*(x_i - x_j)/lambda_k)
```

Spatial similarity is therefore controlled by the available wavelengths `lambda_k`. Small wavelengths provide sensitivity to short-range distances; large wavelengths provide sensitivity to long-range distances.

A raw sum of cosines is not guaranteed to be a monotonic distance-decay function. It provides multiscale distance sensitivity, not necessarily clean monotonic decay.

### 12.2 Head-specific scale weighting

Each spatial head should have its own weighting over Fourier bands:

```text
p_i^(h) = [beta_h * u_g(i); r_g(i) * (w_h * phi(x_i))]
```

where:

* `h` indexes the spatial attention head;
* `w_h` is a head-specific nonnegative weighting over Fourier dimensions;
* sine/cosine dimensions for the same wavelength should share the same scalar weight;
* `beta_h` is a head-specific same-chromosome baseline strength.

The spatial logit becomes:

```text
logit_ij^(h) = tau_h * dot(p_i^(h), p_j^(h))
```

For same-chromosome features, this gives:

```text
logit_ij^(h) proportional to beta_h^2 + sum_k w_h,k^2 * cos(2*pi*(x_i - x_j)/lambda_k)
```

Thus, each spatial head can emphasize a different genomic distance regime.

### 12.3 Initializing heads to different length scales

For the first implementation, initialize spatial heads with fixed or learnable log-Gaussian windows over wavelengths. For head `h`, choose a target length scale `L_h`, then define:

```text
w_h,k = exp(-((log(lambda_k) - log(L_h))^2) / (2*rho^2))
```

where `rho` controls how broad the scale window is.

Example target scales:

| Spatial head | Target scale `L_h` | Interpretation                           |
| ------------ | -----------------: | ---------------------------------------- |
| H1           |             100 bp | Very local sequence-neighborhood effects |
| H2           |               1 kb | Local regulatory neighborhood            |
| H3           |              10 kb | Short-range cis-regulatory context       |
| H4           |             100 kb | Enhancer/promoter-scale neighborhood     |
| H5           |               1 Mb | Broader chromosomal domain-scale context |
| H6           |              10 Mb | Long-range same-chromosome context       |

The weights `w_h` can either be:

1. fixed, preserving interpretable scale-specific heads; or
2. learned after initialization, allowing heads to adapt their spatial kernels.

The preferred first implementation is to initialize `w_h` with log-Gaussian windows and make them learnable under a mild regularization or parameterization that keeps them nonnegative.

### 12.4 Temperature and sharpness

The learned scalar `tau_h` controls how sharply the spatial kernel affects softmax attention:

```text
A_ij^(h) = softmax_j(tau_h * K_ij^(h))
```

Higher `tau_h` makes attention more selective; lower `tau_h` makes it more diffuse. Temperature does not by itself choose the length scale; length scale is primarily controlled by `w_h` and the wavelengths `lambda_k`.

### 12.5 Practical note on monotonic decay

The Fourier dot-product kernel is oscillatory. If explicitly monotonic decay with distance is required, a later version can replace or augment the deterministic Fourier bank with random Fourier features approximating RBF kernels at head-specific length scales:

```text
K_h(delta) approx exp(-delta^2 / (2*sigma_h^2))
```

For the first implementation, use the log-spaced Fourier bank with head-specific scale weighting, then empirically inspect attention as a function of genomic distance.

---

## 13. Key open hyperparameters

| Hyperparameter         |        Initial value | Notes                              |
| ---------------------- | -------------------: | ---------------------------------- |
| $d_s$                  |                  512 | Sequence embedding dimension       |
| $d_p$                  |                  256 | Positional channel dimension       |
| $d_u$                  |                   32 | Same-chromosome code dimension     |
| $d_\phi$               |                  224 | Fourier coordinate dimension       |
| $K$                    |                  112 | Number of Fourier wavelengths      |
| $\lambda_{\min}$       |                10 bp | Finest coordinate scale            |
| $\lambda_{\max}$       |     100 Mb or 300 Mb | Coarsest coordinate scale          |
| $\beta$                | configurable/learned | Same-chromosome baseline strength  |
| $\tau_h$               |              learned | Spatial attention temperature      |
| Spatial Q/K projection |       none initially | Use tied projection only if needed |

---

## 13. Summary of the converged implementation

The first implementation will use separate sequence and position channels. Sequence embeddings will carry biological content and support cross-species generalization. Position encodings will only provide a within-chromosome spatial prior.

For each feature:

$$
s_i = \text{DNAEncoder}(\text{seq}_i)
$$

$$
p_i = [\beta u_{g(i)};\ r_{g(i)}\odot\phi(x_i)]
$$

Spatial attention heads use:

$$
q_i = p_i
$$

$$
k_j = p_j
$$

$$
v_j = W_Vs_j
$$

This achieves the desired behavior:

* no explicit pairwise distance matrix;
* no explicit pairwise chromosome mask;
* arbitrary chromosome counts across species;
* same-chromosome distance sensitivity;
* suppression of cross-chromosome coordinate aliasing;
* sequence-driven cross-species biological transfer;
* compatibility with permutation-invariant feature bags.
