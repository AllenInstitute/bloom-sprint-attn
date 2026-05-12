"""Smoke test — confirms the package is importable and torch is wired up."""

import bloom_arch


def test_import_version():
    assert bloom_arch.__version__ == "0.1.0"


def test_torch_available():
    import torch  # noqa: F401

    # Don't require CUDA here — keeps the test runnable on any box.
    assert hasattr(torch, "Tensor")
