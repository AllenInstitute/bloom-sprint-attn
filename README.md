# bloom-arch

Chromosome-local positional encoding and typed multi-head attention for
transformer models that operate on bags of genome-derived features per cell.

Each feature carries a content embedding `s_i = [n_i; m_i]` (from a DNA
encoder and a measured-value encoder) and a genomic locus `(species,
assembly, chromosome, coordinate)`. The attention block splits into three
head types:

- **Sequence heads** — Q, K from DNA sequence encodings; V from content.
- **Position heads** — Q = K from positional encoding (no learned
  projections); V from content. Same-chromosome features attract;
  cross-chromosome features cancel via sign-gating.
- **Standard heads** — Q, K, V from the full feature `[n; p; m]`.

Each position head has a fixed wavelength window centered at a chosen
length scale (100 bp ... 10 Mb), so a single block can resolve sequence-,
enhancer-, and TAD-level locality simultaneously.

See `docs/spec.md` for the design rationale and `docs/implementation.md`
for the mathematical details, parameter ranges, and tuning notes.

## Quick start

```python
import torch
from bloom_arch.attention import BloomAttentionBlock

B, L = 4, 32         # batch, features per cell
d_seq = 128          # sequence encoding dim (from DNA LM)
d_val = 64           # measured-value encoding dim

block = BloomAttentionBlock(
    d_seq=d_seq,
    d_val=d_val,
    d_u=32,                   # chromosome baseline dim
    K=112,                    # Fourier wavelengths
    d_v=64,                   # per-head value dim
    n_sequence_heads=2,
    n_position_heads=4,       # one per length scale of interest
    n_standard_heads=2,
    d_out=256,
    max_G=30,                 # max distinct chromosomes in a batch
)

n = torch.randn(B, L, d_seq)          # sequence encoding
x = torch.rand(B, L) * 2e8            # genomic coordinates in bp
g = torch.randint(0, 24, (B, L))      # integer chromosome indices
m = torch.randn(B, L, d_val)          # measured-value encoding

# Optional: mask padding positions (True = padding, excluded as keys)
key_padding_mask = torch.zeros(B, L, dtype=torch.bool)
key_padding_mask[:, -4:] = True       # last 4 positions are padding

out = block(n, x, g, m, key_padding_mask=key_padding_mask)  # [B, L, 256]
```

Notes on the interface:

- `x` contains raw base-pair coordinates as floats. Float32 has ~7 decimal
  digits, which is at the edge of precision for coordinates ~2e8 bp with
  shortest wavelength 10 bp.
- `g` is an integer tensor of chromosome indices. The dataloading pipeline
  maps `(species, assembly, chrom)` tuples to integers. The codebook
  generates deterministic codes for indices 0..max_G-1.
- Position heads derive target length scales automatically (log-spaced
  between `lam_min*10` and `lam_max/10`) unless `target_scales` is given.

## Layout

```
src/bloom_arch/
├── posenc/
│   ├── fourier.py           # FourierCoordEncoder (log-spaced sin/cos)
│   ├── genome_codebook.py   # GenomeCodebook (deterministic u_g, r_g)
│   └── encoder.py           # PosEncoder (composes codebook + Fourier)
└── attention/
    ├── sequence_heads.py    # SequenceHeads (Q,K from n; V from s)
    ├── position_heads.py    # PositionHeads (Q=K from pos encoding; V from s)
    ├── standard_heads.py    # StandardHeads (Q,K,V from full feature)
    └── block.py             # BloomAttentionBlock (top-level)
tests/                       # unit and integration tests
docs/
├── spec.md                  # design spec
└── implementation.md        # math, parameter ranges, tuning notes
```

## Install & test

```bash
pip install -e ".[dev]"
pytest -v tests
```
