"""Probe which SDPA backend dispatches for ``SpatialAttentionSDPA`` on this device.

PyTorch's :func:`torch.nn.functional.scaled_dot_product_attention` chooses
among four backends:

  * ``FLASH_ATTENTION`` - fused kernel, fp16/bf16, CUDA SM >= 8.0,
    per-head dim <= 256 (PyTorch >= 2.2).
  * ``EFFICIENT_ATTENTION`` - memory-efficient kernel (xFormers lineage).
  * ``CUDNN_ATTENTION`` - when available in the cuDNN runtime.
  * ``MATH`` - fallback that works anywhere with no fusion.

We force each backend in turn and report whether it accepts our input. With
the spec's default dims, the QK head dim is ``d_u + 2K = 32 + 224 = 256`` -
right at the flash upper limit, so this diagnostic is the cleanest way to
confirm whether flash dispatches in practice.

Usage
-----
$ python scripts/check_sdpa_backends.py
"""

from __future__ import annotations

import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from bloom_arch.attention import SpatialAttentionSDPA
from bloom_arch.posenc import ChromLocalPosEncoder

# Bench config - match the spec defaults so the diagnostic reflects realistic shapes.
D_S = 512
D_V = 64
H = 6
K = 112
TARGET_SCALES = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


def header(s: str) -> None:
    print(f"\n=== {s} ===")


def _build_layer(device: str, dtype: torch.dtype) -> tuple[ChromLocalPosEncoder, SpatialAttentionSDPA]:
    encoder = ChromLocalPosEncoder(d_u=32, K=K, lambda_min=10.0, lambda_max=1.0e8).to(device)
    layer = SpatialAttentionSDPA(
        d_s=D_S, d_v=D_V, H=H, K=K, lambdas=encoder.fourier.lambdas,
        target_scales=TARGET_SCALES,
    ).to(device).to(dtype)
    return encoder, layer


def probe(dtype: torch.dtype, device: str, B: int = 2, N: int = 64) -> None:
    encoder, layer = _build_layer(device, dtype)
    coords = (torch.rand(B, N) * 1.0e7).to(device)
    keys = [[("musmus", "mm10", "chr1")] * N for _ in range(B)]
    pos = encoder(coords, keys)
    u_ = pos["u"].to(dtype)
    r_ = pos["r_phi"].to(dtype)
    s = torch.randn(B, N, D_S, device=device, dtype=dtype)

    head_dim_qk = layer.d_u + 2 * layer.K
    print(
        f"device={device}, dtype={dtype}, head_dim(Q,K)={head_dim_qk}, "
        f"head_dim(V)={layer.d_v}, B={B}, N={N}, H={layer.H}"
    )

    backends = [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]
    for name, backend in backends:
        try:
            with sdpa_kernel(backend):
                out = layer(u_, r_, s)
            ok = torch.isfinite(out).all().item()
            tag = "PASS" if ok else "PASS-BUT-NONFINITE"
            print(f"  [{tag}] {name}: out={tuple(out.shape)} dtype={out.dtype}")
        except Exception as e:
            msg = str(e).strip().splitlines()[0][:140]
            print(f"  [FAIL] {name}: {msg}")


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA build:  {torch.version.cuda}")
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        major, minor = torch.cuda.get_device_capability(0)
        print(f"Compute cap: sm_{major}{minor}")
    else:
        print("CUDA not available - only MATH backend on CPU is meaningful.")

    if torch.cuda.is_available():
        header("bfloat16 on CUDA (flash-attention's preferred dtype)")
        probe(torch.bfloat16, "cuda")

        header("float16 on CUDA")
        probe(torch.float16, "cuda")

        header("float32 on CUDA (flash refuses fp32 by design)")
        probe(torch.float32, "cuda")

    header("float32 on CPU (only MATH applies)")
    probe(torch.float32, "cpu")

    return 0


if __name__ == "__main__":
    sys.exit(main())
