# Relative position attention masking

## 1. Problem overview

A bidirectional transformer model receives a bag of genome-derived features per cell. Each feature consists of an identifier for the underlying genomic sequence (`d_seq` dimensions), genomic coordinate relevant to that sequence (`d_pos` dimensions), and the measured value (`d_val` dimensions). The transformer output is a per-cell representation. We want the design to accommodate cell-level data from different species.

To generalize this across data from different species, genomic sequence identifiers (derived from DNA language models) and values don't need special treatment. However genomic coordinates should be used only in a relative sense, without relying on synteny priors. We will do this by constructing attention heads at different length scales for this. 

Let's say each batch for the transformer is of shape `(B, L, E)`. `B` is batch, `L` is the context length. `E = d_seq + d_pos + d_val`. 

The genomic coordinates consist of a species identifier (e.g. `mm`, `hg`), a chromosome identifier `(1, 2, 3, ... X, Y, M)`, and a coordinate along the chromosome. 

The desired behavior is:
 - interaction between features on the same chromosomes is encouraged.
 - interaction between features across different chromosomes is discouraged.
 - closeness along the chromosome for any feature pair is determined by the difference in coordinates between them.
 - different attention heads should allow for interactions at different length scales. 

**Examples:**
 - `chr1:1000` and `chr1:1100` are as close as `chr5:2000` and `chr5:2100`
 - `chr1:1000` and `chr2:1000` are not close.
 
**Note:**
`E` is ~512-1024 and `d_pos` is ~128-512.

## 2. Design

For each feature $i$, let $n_i$ be the sequence encoding (derived from a DNA model) and $m_i$ be the measured-value encoding (typically an MLP). Then the model computes content encoding $s_i$ from the sequence encoding and measured-value encoding for that feature:

$$
s_i = [n_i; m_i]
$$

Calculate the position encoding for feature $i$ on genome/chromosome $g_i$:

$$
p_i = [\beta u_{g_i}; r_{g_i} \odot \phi(x_i)]
$$

 - $x_i$ is the coordinate of feature $i$ along its chromosome or contig.
 - $\phi(x_i) \in \mathbb{R}^{2K}$ is a multiscale Fourier encoding of the coordinate. 
 - $g_i$ is a genome/chromosome identifier (`int`) derived from `(species, assembly, chromosome)`. 
 - $r_{g_i} \in \{-1, +1\}^{2K}$ is a fixed deterministic random sign vector per $g_i$
 - $u_{g_i} \in \mathbb{R}^{d_u}$ is a fixed deterministic random unit vector for $g_i$. 
 - $\beta$ learnable same-chromosome preference.
 - $\odot$  is elementwise multiplication

The model then uses a block with custom attention heads:

1. **Sequence heads**
 - `k, q` are derived from sequence encoding.
 - `v` from content encoding.

2. **Position heads**
 - `k, q` are derived from position encoding.
 - `v` from content encoding.

Optionally, later layers may include mixed heads that can use both sequence and position, but the first implementation should prioritize clean separation.


## 3. Attention head definitions

### 3.1 Sequence heads

Sequence heads use sequence embeddings for queries, keys, and values:

$$
q_i^{(h)} = W_Q^{(h)} n_i \\
k_j^{(h)} = W_K^{(h)} n_j \\
v_j^{(h)} = W_V^{(h)} s_j
$$

Attention logits:

$$
\ell_{ij}^{(h)} = \frac{(q_i^{(h)})^\top k_j^{(h)}}{\sqrt{d_n}}
$$

Here $d_n$ is the sequence encoding dimensionality. These heads learn relationships among features based on sequence encodings of the DNA model. Note that the values are both the sequence encoding and the measured-value encoding. 

### 3.2 Position heads

Position heads use positional encodings for queries and keys, but content embeddings for values:

$$
q_i^{(h)} = p_i^{h} \\
k_j^{(h)} = p_j^{h} \\
v_j^{(h)} = W_V^{(h)} s_j
$$

Attention logits:

$$
\ell_{ij}^{(h)} = \tau_h \cdot p_i^{(h)\top} p_j^{(h)}
$$

Note that position encodings are constructed per head - so that they attend over a pre-determined length scale. We also don't use projections ($W_K, W_Q$) for the position heads. Instead $p^{(h)}$ can be viewed already as the already projected version. 

**Note:**
 $\phi(x)$ is scaled by $\frac{1}{\sqrt{K}}$ so that $\phi(x_i)^\top \phi(x_j) \in [-1, 1]$. With this normalization, $\tau_h$ is initialized to $\frac{1}{\sqrt{2K}}$ so that the logit magnitude is $O(1)$ at initialization.

## 4. Sinusoidal coordinate encoding

For a coordinate $x$, define K wavelengths:

$$
\lambda_0, \lambda_2, \dots, \lambda_{K-1} \\
\lambda_k = \lambda_{\min}\left(\frac{\lambda_{\max}}{\lambda_{\min}}\right)^{k/(K-1)}
$$

Then the un-weighted position coordinates are:

$$
\phi(x) = [\sin(2\pi x/\lambda_0),\ \cos(2\pi x/\lambda_0),\ \sin(2\pi x/\lambda_1),\ \cos(2\pi x/\lambda_1),\ \dots,\ \sin(2\pi x/\lambda_{K-1}),\ \cos(2\pi x/\lambda_{K-1})]
$$

Thus $d_\phi = 2K$

### 4.1 Dot-product property

For a single sine/cosine pair:

$$
\begin{aligned}
\sin(2\pi x_i/\lambda)\sin(2\pi x_j/\lambda) +
\cos(2\pi x_i/\lambda)\cos(2\pi x_j/\lambda)
&=
\cos(2\pi(x_i-x_j)/\lambda) \\
&= \cos(2\pi(\Delta_{ij})/\lambda) 
\end{aligned}
$$

Therefore, the dot product of two such encodings depends on relative distance in the coordinates (i.e. translational invariance):

$$
\phi(x_i)^\top \phi(x_j) = \sum_k \cos(2\pi(\Delta_{ij})/\lambda_k)
$$

### 4.2 Per-head length-scale weighting

We introduce a non-negative weight vector $w_h \in \mathbb{R}_{\ge 0}^{K}$ over wavelengths. The per-head position encoding is:

$$
p_i^{(h)} = [\beta_h u_{g_i};\ r_{g_i} \odot (w_h \odot \phi(x_i))]
$$

The sin and cosine terms for each wavelength share the same scalar weight. The same-chromosome logit becomes:

$$
\ell_{ij}^{(h)} \propto \beta_h^2 \delta_{g_{i}g_{j}} + \delta_{g_{i}g_{j}}\sum_k w_{h,k}^2 \cos\!\left(\frac{2\pi(\Delta_{ij})}{\lambda_k}\right)
$$

Wavelengths with large $w_{h,k}$ dominate the kernel shape; wavelengths with $w_{h,k} \approx 0$ are effectively 0 for that head. The $\delta_{g_{i}g_{j}}$ term is a property of the way $r$ is constructed (described below). Similarly the construction of $u_g$ (also described below) returns $\delta_{g_{i}g_{j}}$.  

**Per-head initialization:** A target length scale $L_h$ is used per to  initialize $w_h$ as a Gaussian window:

$$
w_{h,k} = \exp\!\left(-\frac{(\ln \lambda_k - \ln L_h)^2}{2\rho^2}\right)
$$

where $\rho$ controls the spectral bandwidth in log-space. 


## 5. Sign-gating of sinusoidal encodings

We only want features within the chromosome to interact. This is achieved by sign-gating the sinusoidal coordinate component. We can think of this as a random, fixed rotation in high dimensional space, unique to each chromosome. Let $R_g$ be a diagonal matrix constructed from the entries of the vector $r_g$. Then 

$$
r_g \odot \phi(x) = R_g\phi(x)
$$

The dot product of the position encoding then contains terms 

$$
\begin{aligned}
(R_{g_i}\phi(x_i))^T(R_{g_j}\phi(x_j)) &= \phi(x_i)^TR_{g_i}^TR_{g_j}\phi(x_j) \\
&= \phi(x_i)^T\delta_{g_{i} g_{j}}\phi(x_j)
\end{aligned}
$$


## 6. Role of the $\beta u_g$ term

The term $\beta u_g$ provides an optional same-chromosome bias to complement the design of $r_g$. Limited dimensionality of the position encodings leads to the cross-chromosome terms not cancelling out fully. 

The key idea is to define $u_g$ that are orthogonal. The recipe for this is to define randomly oriented features in high-dimensional space. We practically achieve this by sampling the entries of each $u_g$ from $\mathcal{N(0,1)}$ and unit-normalizing it.

$u_g$ terms in the dot product for the position encoding are simply 

$$
\beta^{2} u_{g_i}^\top u_{g_j} \approx \beta^{2} \delta_{g_i g_j}
$$

### 7. Interface


### 8. Multi-head module

Default target scales:

| Position head | Target scale `L_h` | Interpretation                          |
| ------------ | -----------------: | ---------------------------------------- |
| H1           |             100 bp | Very local sequence-neighborhood effects |
| H2           |               1 kb | Local regulatory neighborhood            |
| H3           |              10 kb | Short-range cis-regulatory context       |
| H4           |             100 kb | Enhancer/promoter-scale neighborhood     |
| H5           |               1 Mb | Broader chromosomal domain-scale context |
| H6           |              10 Mb | Long-range same-chromosome context       |

Default is to fix `L_h`. Make implementation flexible to enable `L_h` to be learnable, but constrained to not deviate across scales handled in other heads. 

Weights `w_h` could also be learned, but that is our of scope for now. 


## 9. Key hyperparameters

| Hyperparameter         |        Initial value | Notes                              |
| ---------------------- | -------------------: | ---------------------------------- |
| $d_s$                  |                  512 | Sequence embedding dimension       |
| $d_p$                  |                  256 | Positional channel dimension       |
| $d_u$                  |                   32 | Same-chromosome code dimension     |
| $d_\phi$               |                  224 | Fourier coordinate dimension       |
| $K$                    |                  112 | Number of Fourier wavelengths      |
| $\lambda_{\min}$       |               100 bp | Finest coordinate scale            |
| $\lambda_{\max}$       |               100 Mb | Coarsest coordinate scale          |
| $\beta$                |         configurable | Same-chromosome baseline strength  |
| $\tau_h$               |         configurable | Position attention temperature     |
| $\rho$                  |         configurable | Position attention temperature     |

 
 - User sets $d_p$: raise error if not divisible by 2
 - User sets $d_u$: default is 0
 - $K$ is derived using $d_p - d_u$. 
 - $\beta$: initialized only if $d_u$ is non-zero. If initialized, set to 1 and learnable is false. Learnability set through flag (flag value does not matter if not initialized)
 - $\tau_h$: default value is 1, and not learnable. Learnability set through flag. 
 - $\rho$: default value
