#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for Scrambled Subsampled Randomized Fourier Transform (SSRFT).

See `[TYUC2019, 3.2] <https://arxiv.org/abs/1902.08651>`_.
"""

import pytest
import torch

from skerch.ssrft import SSRFT
from skerch.utils import BadShapeError

from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def f64_rtol():
    """Relative error tolerance for float64.

    .. note::

      Float32 tolerance goes to zero for larger shapes (which is where SSVD
      makes sense), so we only test f64. Use f32 at own risk!
    """
    result = {torch.float64: 1e-10}
    return result


@pytest.fixture
def square_shapes():
    """Test shapes for square matrices."""
    result = [
        (1, 1),
        (10, 10),
        (100, 100),
        (10_000, 10_000),
        (1_000_000, 1_000_000),
    ]
    return result


@pytest.fixture
def fat_shapes():
    """Test shapes for fat (wide) matrices."""
    result = [
        (1, 10),
        (10, 100),
        (100, 1_000),
        (10_000, 100_000),
        (1_000_000, 10_000_000),
    ]
    return result


# ##############################################################################
# # TESTS
# ##############################################################################
def test_no_nans(torch_devices, f64_rtol, rng_seeds, square_shapes):
    """Tests that SSRFT yields no NaNs."""
    for seed in rng_seeds:
        for h, w in square_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            for device in torch_devices:
                for dtype, _rtol in f64_rtol.items():
                    x = torch.randn(w, dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = y @ ssrft
                    #
                    assert not x.isnan().any(), f"{ssrft, device, dtype}"
                    assert not y.isnan().any(), f"{ssrft, device, dtype}"
                    assert not xx.isnan().any(), f"{ssrft, device, dtype}"


def test_invertible(torch_devices, f64_rtol, rng_seeds, square_shapes):
    """Invertibility/orthogonality of quare SSRFT.

    Tests that, when input and output dimensionality are the same, the SSRFT
    operator is orthogonal, i.e. we can recover the input exactly via an
    adjoint operation.

    Also tests that it works for mat-vec and mat-mat formats.
    """
    for seed in rng_seeds:
        for h, w in square_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            for device in torch_devices:
                for dtype, rtol in f64_rtol.items():
                    # matvec
                    x = torch.randn(w, dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = y @ ssrft
                    #
                    assert torch.allclose(
                        x, xx, rtol=rtol
                    ), f"MATVEC: {ssrft, device, dtype}"
                    # matmat
                    x = torch.randn((w, 2), dtype=dtype).to(device)
                    y = ssrft @ x
                    xx = (y.T @ ssrft).T
                    #
                    assert torch.allclose(
                        x, xx, rtol=rtol
                    ), f"MATMAT: {ssrft, device, dtype}"
                    # matmat-shape tests
                    assert len(y.shape) == 2
                    assert len(xx.shape) == 2
                    assert y.shape[-1] == 2
                    assert xx.shape[-1] == 2


def test_seed_consistency(torch_devices, f64_rtol, rng_seeds, square_shapes):
    """Seed consistency of SSRFT.

    Test that same seed and shape lead to same operator with same results,
    and different otherwise.
    """
    for seed in rng_seeds:
        for h, w in square_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            ssrft_same = SSRFT((h, w), seed=seed)
            ssrft_diff = SSRFT((h, w), seed=seed + 1)
            for device in torch_devices:
                for dtype, _rtol in f64_rtol.items():
                    # matvec
                    x = torch.randn(w, dtype=dtype).to(device)
                    assert ((ssrft @ x) == (ssrft_same @ x)).all()
                    # here, dim=1 may indeed result in same output, since
                    # there are no permutations or index-pickings, so 50/50.
                    # therefore we ignore that case.
                    if x.numel() > 1:
                        assert ((ssrft @ x) != (ssrft_diff @ x)).any()


def test_device_consistency(torch_devices, f64_rtol, rng_seeds, square_shapes):
    """Seed consistency of SSRFT across different devices.

    Test that same seed and shape lead to same operator with same results,
    even when device is different.
    """
    for seed in rng_seeds:
        for h, w in square_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            for dtype, rtol in f64_rtol.items():
                # apply SSRFT on given devices and check results are equal
                x = torch.randn(w, dtype=dtype)
                y = [(ssrft @ x.to(device)).cpu() for device in torch_devices]
                for yyy in y:
                    assert torch.allclose(
                        yyy, y[0], rtol=rtol
                    ), "SSRFT inconsistency among devices!"


def test_unsupported_tall_ssrft(rng_seeds, fat_shapes):
    """Tail SSRFT linops are not supported."""
    for seed in rng_seeds:
        for h, w in fat_shapes:
            with pytest.raises(BadShapeError):
                # If this line throws a BadShapeError, the test passes
                SSRFT((w, h), seed=seed)


def test_input_shape_mismatch(rng_seeds, fat_shapes, torch_devices, f64_rtol):
    """Test case for SSRFT shape consistency."""
    for seed in rng_seeds:
        for h, w in fat_shapes:
            ssrft = SSRFT((h, w), seed=seed)
            for device in torch_devices:
                for dtype, _rtol in f64_rtol.items():
                    # forward matmul
                    x = torch.empty(w + 1, dtype=dtype).to(device)
                    with pytest.raises(BadShapeError):
                        ssrft @ x
                    # adjoint matmul
                    x = torch.empty(h + 1, dtype=dtype).to(device)
                    with pytest.raises(BadShapeError):
                        x @ ssrft
