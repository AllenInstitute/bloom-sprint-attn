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
