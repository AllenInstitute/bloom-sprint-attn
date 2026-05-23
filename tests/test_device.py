"""Regression tests: SpatialAttention must construct cleanly on CUDA.

The bug we are guarding against: ``_default_log_gaussian_init`` originally
built ``log_L`` on the default (CPU) device while ``lambdas`` could already
be on CUDA, producing a cross-device subtraction error during ``.to('cuda')``
of the encoder followed by ``SpatialAttention(lambdas=encoder.fourier.lambdas)``.
"""

from __future__ import annotations

import pytest
import torch

from bloom_arch.attention import SpatialAttention, SpatialAttentionSDPA
from bloom_arch.posenc import ChromLocalPosEncoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spatial_attention_constructs_from_cuda_lambdas():
    encoder = ChromLocalPosEncoder(d_u=32, K=64, lambda_min=10.0, lambda_max=1.0e8).cuda()
    # This used to raise "tensors not on the same device" because log_L was
    # built on CPU while lambdas had been moved to CUDA by the .cuda() above.
    layer = SpatialAttention(
        d_s=32,
        d_v=8,
        H=3,
        K=64,
        lambdas=encoder.fourier.lambdas,
        target_scales=[1e3, 1e4, 1e5],
    )
    # Init params should land on the same device as lambdas.
    assert layer.log_w.device == encoder.fourier.lambdas.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spatial_attention_sdpa_constructs_from_cuda_lambdas():
    encoder = ChromLocalPosEncoder(d_u=32, K=64, lambda_min=10.0, lambda_max=1.0e8).cuda()
    layer = SpatialAttentionSDPA(
        d_s=32,
        d_v=8,
        H=3,
        K=64,
        lambdas=encoder.fourier.lambdas,
        target_scales=[1e3, 1e4, 1e5],
    )
    assert layer.log_w.device == encoder.fourier.lambdas.device
