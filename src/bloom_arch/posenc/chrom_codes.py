"""Deterministic chromosome codebook (spec §7, §9).

For every (species, assembly, chromosome) key we generate two stateless vectors:

- ``r_g``  : a fixed ±1 sign vector of length ``d_phi`` used to gate the Fourier
            coordinate encoding so that within-chromosome dot products preserve
            φ(x_i)·φ(x_j) exactly while cross-chromosome dot products
            decorrelate (spec §7).
- ``u_g``  : a fixed unit-norm vector of length ``d_u`` used as the
            "same-chromosome baseline" code (spec §8).

Both vectors are derived from a BLAKE2b hash of ``salt|species|assembly|chrom|kind``,
which gives identical results across processes and devices. They are NOT
learnable: ``ChromCodeBook`` has zero parameters and zero buffers, and a fresh
codebook with the same salt reproduces the same vectors for the same keys
without any state transfer.

The book caches generated codes in a per-instance dictionary keyed by
``(species, assembly, chrom)`` so a batch of N features that references only
M unique chromosomes hashes M times, not N.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import torch
from torch import nn

ChromKey = tuple[str, str, str]  # (species, assembly, chrom)


class ChromCodeBook(nn.Module):
    """Deterministic, parameter-free codebook keyed by (species, assembly, chrom).

    Parameters
    ----------
    d_u : int
        Dimension of the same-chromosome baseline vector u_g (spec §8, §10.1).
    d_phi : int
        Dimension of the sign vector r_g. Must equal the Fourier encoding
        dimension (2*K). Spec §10.1 default: 224.
    salt : str
        Versioning salt mixed into every hash. Change this to invalidate every
        cached code (e.g. when bumping the hashing convention).

    Notes
    -----
    The module derives ``r_g`` from independent draws of {-1, +1} per dimension
    and ``u_g`` from a unit-normalized standard normal of length ``d_u``. Two
    different (species, assembly, chrom) keys produce approximately independent
    sign vectors, satisfying the cross-chromosome decorrelation requirement of
    spec §11.3.

    The codebook has **no learnable parameters and no registered buffers**.
    Codes are generated on CPU (since string hashing is CPU-only) and the
    output tensors are moved to the requested device/dtype in ``forward``.
    """

    def __init__(self, d_u: int = 32, d_phi: int = 224, salt: str = "posenc_v1") -> None:
        super().__init__()
        if d_u <= 0:
            raise ValueError(f"d_u must be positive, got {d_u}")
        if d_phi <= 0 or d_phi % 2 != 0:
            raise ValueError(f"d_phi must be a positive even integer, got {d_phi}")
        self.d_u = d_u
        self.d_phi = d_phi
        self.salt = salt
        self._cache: dict[ChromKey, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _hash_seed(s: str) -> int:
        """Map a string to a 64-bit unsigned int via BLAKE2b."""
        digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)

    def _generate(self, key: ChromKey) -> tuple[torch.Tensor, torch.Tensor]:
        """Hash a key into (u_g, r_g). Pure function of (salt, key, d_u, d_phi)."""
        species, assembly, chrom = key
        base = f"{self.salt}|{species}|{assembly}|{chrom}"
        sign_seed = self._hash_seed(base + "|sign")
        unit_seed = self._hash_seed(base + "|unit")

        rng_sign = np.random.default_rng(sign_seed)
        rng_unit = np.random.default_rng(unit_seed)

        # ±1 sign vector — use integer draw then map for tight control of distribution.
        r_np = rng_sign.integers(0, 2, size=self.d_phi).astype(np.float32) * 2.0 - 1.0

        # Unit-norm Gaussian vector.
        u_np = rng_unit.standard_normal(size=self.d_u).astype(np.float32)
        norm = float(np.linalg.norm(u_np))
        if norm == 0.0:
            # astronomically unlikely; resample defensively.
            u_np = np.ones(self.d_u, dtype=np.float32) / np.sqrt(self.d_u)
        else:
            u_np = u_np / norm

        return torch.from_numpy(u_np), torch.from_numpy(r_np)

    def _get(self, key: ChromKey) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        u, r = self._generate(key)
        self._cache[key] = (u, r)
        return u, r

    def clear_cache(self) -> None:
        """Drop the in-memory code cache (useful for tests)."""
        self._cache.clear()

    def forward(
        self,
        chrom_keys: Sequence[Sequence[ChromKey]],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up codes for a batch of feature bags.

        Parameters
        ----------
        chrom_keys : list of length B, each a list of length N
            ``chrom_keys[b][n] = (species, assembly, chrom)`` for feature ``n``
            in batch element ``b``. All rows must be the same length N.
        device : torch device, optional
            Device to place the output on. Codes are always generated on CPU
            then moved.
        dtype : torch.dtype
            Output dtype. Defaults to float32.

        Returns
        -------
        u : Tensor of shape [B, N, d_u]
            Per-feature same-chromosome baseline vectors.
        r : Tensor of shape [B, N, d_phi]
            Per-feature ±1 sign vectors.
        """
        B = len(chrom_keys)
        if B == 0:
            return (
                torch.empty(0, 0, self.d_u, device=device, dtype=dtype),
                torch.empty(0, 0, self.d_phi, device=device, dtype=dtype),
            )
        N = len(chrom_keys[0])
        for row in chrom_keys:
            if len(row) != N:
                raise ValueError("chrom_keys must be rectangular (all rows same length)")

        # Build a unique-key table to amortize hashing.
        flat: list[ChromKey] = [tuple(k) for row in chrom_keys for k in row]  # type: ignore[misc]
        uniq_to_idx: dict[ChromKey, int] = {}
        inv = torch.empty(B * N, dtype=torch.long)
        for i, k in enumerate(flat):
            idx = uniq_to_idx.get(k)
            if idx is None:
                idx = len(uniq_to_idx)
                uniq_to_idx[k] = idx
            inv[i] = idx

        U_uniq = torch.empty(len(uniq_to_idx), self.d_u)
        R_uniq = torch.empty(len(uniq_to_idx), self.d_phi)
        for k, idx in uniq_to_idx.items():
            u, r = self._get(k)
            U_uniq[idx] = u
            R_uniq[idx] = r

        u_out = U_uniq.index_select(0, inv).view(B, N, self.d_u).to(device=device, dtype=dtype)
        r_out = R_uniq.index_select(0, inv).view(B, N, self.d_phi).to(device=device, dtype=dtype)
        return u_out, r_out
