#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`recovery`."""


import pytest
import torch
from skerch.utils import gaussian_noise, svd
from skerch.synthmat import RandomLordMatrix
from skerch.recovery import singlepass
from . import rng_seeds, torch_devices


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 3e-5,
        torch.complex64: 1e-5,
        torch.float64: 1e-10,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def parallel_modes():
    result = (None, "mp")
    return result


@pytest.fixture
def complex_dtypes_tols():
    """Error tolerances for each complex dtype."""
    result = {
        torch.complex64: 1e-5,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def iid_noise_linop_types():
    """Class names for all noise linops to be tested.

    :returns: Collection of pairs ``(lop_type, is_complex_only)``
    """
    result = {
        (GaussianNoiseLinOp, False),
        (RademacherNoiseLinOp, False),
        (PhaseNoiseLinOp, True),
    }
    return result


@pytest.fixture
def iid_hw_and_autocorr_tolerances():
    """Error tolerances for each complex dtype."""
    hw = (20, 20)
    delta_at_least = 0.7
    nondelta_at_most = 0.5
    return hw, delta_at_least, nondelta_at_most


@pytest.fixture
def ssrft_hw_and_autocorr_tolerances():
    """Error tolerances for each complex dtype."""
    hw = (30, 10)
    delta_at_least = 0.7
    nondelta_at_most = 0.5
    return hw, delta_at_least, nondelta_at_most


# ##############################################################################
# # PERFORM MEASUREMENTS
# ##############################################################################
def test_recovery_formal(rng_seeds, torch_devices, dtypes_tols):
    """Test case for recovery (formal and correctness).

    For torch/numpy inputs, and for all recovery methods, test:
    * xxx
    *
    * Output is in matching device and dtype


    1. our target linop is a lr matrix


    we need both sketches and the mop
    everything should have different types
    """
    hw, rank, meas = (50, 50), 5, 25
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                # torch identity, low-rank matrix and noisy matrices
                I1 = torch.eye(meas, dtype=dtype, device=device)
                mat1, _ = RandomLordMatrix.exp(
                    hw,
                    rank,
                    decay=100,
                    diag_ratio=0.0,
                    symmetric=False,
                    psd=False,
                    seed=seed,
                    dtype=dtype,
                    device=device,
                )
                U1, S1, Vh1 = svd(mat1)
                right1 = gaussian_noise(
                    (hw[1], meas), seed=seed + 100, device=device, dtype=dtype
                )
                left1 = gaussian_noise(
                    (meas, hw[0]), seed=seed + 101, device=device, dtype=dtype
                )
                #
                # SINGLEPASS
                # measurements and recovery
                Y1 = mat1 @ right1  # tall
                Z1 = left1 @ mat1  # fat
                U1rec, S1rec, Vh1rec = singlepass(Y1, Z1, right1)
                # correctness of result
                assert torch.allclose(
                    mat1, (U1 * S1) @ Vh1, atol=tol
                ), "Incorrect singlepass torch recovery!"
                # orthogonality of recovered U, V
                assert torch.allclose(
                    I1, U1rec.H @ U1rec, atol=tol
                ), "Singlepass torch U not orthogonal?"
                assert torch.allclose(
                    I1, Vh1rec @ Vh1rec.H, atol=tol
                ), "Singlepass torch V not orthogonal?"
                # correctness of recovered svals
                assert torch.allclose(
                    S1[:meas], S1rec, atol=tol
                ), "Incorrect singlepass torch svals!"
                # matching device and type
                assert (
                    U1rec.device == mat1.device
                ), "Incorrect U singlepass torch device!"
                assert (
                    S1rec.device == mat1.device
                ), "Incorrect S singlepass torch device!"
                assert (
                    Vh1rec.device == mat1.device
                ), "Incorrect V singlepass torch device!"
                assert (
                    U1rec.dtype == mat1.dtype
                ), "Incorrect U singlepass torch dtype!"
                assert (
                    S1rec.dtype == mat1.real.dtype
                ), "Incorrect S singlepass torch dtype!"
                assert (
                    Vh1rec.dtype == mat1.dtype
                ), "Incorrect V singlepass torch dtype!"
