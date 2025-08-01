#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`skerch.utils`."""


import pytest
import torch
import numpy as np

from skerch.utils import torch_dtype_as_str, complex_dtype_to_real
from skerch.utils import uniform_noise, gaussian_noise, rademacher_noise
from skerch.utils import randperm, rademacher_flip
from skerch.utils import COMPLEX_DTYPES, phase_noise, phase_shift
from skerch.utils import qr, pinv

from . import rng_seeds, torch_devices, autocorrelation_test_helper


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 1e-5,
        torch.complex64: 1e-5,
        torch.float64: 1e-10,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def rand_dims_samples(request):
    """Shapes to test linop_to_matrix"""
    if request.config.getoption("--quick"):
        num_samples = 3
    else:
        num_samples = 10
    result = [
        (1000, num_samples),
    ]
    return result


# ##############################################################################
# # DTYPES
# ##############################################################################
def test_dtype_utils():
    """Test case for dtype manipulation utils"""
    dtype = torch.float16
    assert "float16" == torch_dtype_as_str(dtype), "Bad str(dtype) conversion!"
    assert torch.float16 == complex_dtype_to_real(
        dtype
    ), "Bad real(dtype) cast!"
    dtype = torch.float32
    assert "float32" == torch_dtype_as_str(dtype), "Bad str(dtype) conversion!"
    assert torch.float32 == complex_dtype_to_real(
        dtype
    ), "Bad real(dtype) cast!"
    dtype = torch.float64
    assert "float64" == torch_dtype_as_str(dtype), "Bad str(dtype) conversion!"
    assert torch.float64 == complex_dtype_to_real(
        dtype
    ), "Bad real(dtype) cast!"
    #
    dtype = torch.complex32
    assert "complex32" == torch_dtype_as_str(
        dtype
    ), "Bad str(type) conversion!"
    assert torch.float16 == complex_dtype_to_real(
        dtype
    ), "Bad real(dtype) cast!"
    dtype = torch.complex64
    assert "complex64" == torch_dtype_as_str(
        dtype
    ), "Bad str(dtype) conversion!"
    assert torch.float32 == complex_dtype_to_real(
        dtype
    ), "Bad real(dtype) cast!"
    dtype = torch.complex128
    assert "complex128" == torch_dtype_as_str(
        dtype
    ), "Bad str(dtype) conversion!"
    assert torch.float64 == complex_dtype_to_real(
        dtype
    ), "Bad real(dtype) cast!"
    #
    dtype = torch.uint8
    assert "uint8" == torch_dtype_as_str(dtype), "Bad str(dtype) conversion!"
    assert torch.uint8 == complex_dtype_to_real(dtype), "Bad real(dtype) cast!"
    dtype = torch.int32
    assert "int32" == torch_dtype_as_str(dtype), "Bad str(dtype) conversion!"
    assert torch.int32 == complex_dtype_to_real(dtype), "Bad real(dtype) cast!"
    dtype = torch.int64
    assert "int64" == torch_dtype_as_str(dtype), "Bad str(dtype) conversion!"
    assert torch.int64 == complex_dtype_to_real(dtype), "Bad real(dtype) cast!"


# ##############################################################################
# # NOISE SOURCES
# ##############################################################################
def test_noise_sources(
    rng_seeds, dtypes_tols, torch_devices, rand_dims_samples
):
    """Test case for noise sources in ``utils``.

    Note that SSRFT noise is implemented and tested in ``measurements`` and
    not here.

    For all devices and datatypes, and a variety of shapes:
    * Same-seed consistency
    * Quasi-delta autocorelation

    For phase_noise, and phase_shift also test that:
    * conj generates same-seed conjugate
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for dims, num_samples in rand_dims_samples:
                    for n in range(num_samples):
                        sd = seed + n
                        # uniform
                        noise1 = uniform_noise(dims, sd, dtype, device)
                        noise2 = uniform_noise(dims, sd, dtype, device)
                        noise3 = uniform_noise(dims, sd + 1, dtype, device)
                        autocorrelation_test_helper(noise1)
                        assert torch.allclose(
                            noise1, noise2, atol=tol
                        ), "Same seed, different noise? (uniform)"
                        assert not torch.allclose(
                            noise1, noise3, atol=tol
                        ), "Different seed, same noise? (uniform)"
                        # gaussian
                        noise1 = gaussian_noise(dims, 0, 1, sd, dtype, device)
                        noise2 = gaussian_noise(dims, 0, 1, sd, dtype, device)
                        noise3 = gaussian_noise(
                            dims, 0, 1, sd + 1, dtype, device
                        )
                        autocorrelation_test_helper(noise1)
                        assert torch.allclose(
                            noise1, noise2, atol=tol
                        ), "Same seed, different noise? (Gaussian)"
                        assert not torch.allclose(
                            noise1, noise3, atol=tol
                        ), "Different seed, same noise? (gaussian)"
                        # rademacher
                        noise1 = rademacher_noise(dims, sd, device)
                        noise2 = rademacher_noise(dims, sd, device)
                        noise3 = rademacher_noise(dims, sd + 1, device)
                        autocorrelation_test_helper(noise1.to(dtype))
                        assert torch.allclose(
                            noise1, noise2, atol=tol
                        ), "Same seed, different noise? (Rademacher)"
                        assert not torch.allclose(
                            noise1, noise3, atol=tol
                        ), "Different seed, same noise? (Rademacher)"
                        ones = torch.ones(dims, dtype=dtype, device=device)
                        # rademacher flip
                        noise1, m1 = rademacher_flip(
                            ones, sd, inplace=False, rng_device=device
                        )
                        noise2, _ = rademacher_flip(
                            ones, sd, inplace=False, rng_device=device
                        )
                        noise3, _ = rademacher_flip(
                            ones, sd + 1, inplace=False, rng_device=device
                        )
                        autocorrelation_test_helper(noise1)
                        assert torch.allclose(
                            noise1, noise2, atol=tol
                        ), "Same seed, different noise? (Rademacher flip)"
                        assert not torch.allclose(
                            noise1, noise3, atol=tol
                        ), "Different seed, same noise? (Rademacher flip)"
                        assert (
                            noise1 == m1
                        ).all(), (
                            "Inconsistent output and mask in Rademacher flip?"
                        )
                        # randperm
                        perm1a = randperm(dims, sd, device, inverse=False)
                        perm1b = randperm(dims, sd, device, inverse=True)
                        perm2 = randperm(dims, sd, device, inverse=False)
                        perm3 = randperm(dims, sd + 1, device, inverse=False)
                        arange = torch.arange(dims, device=device)
                        autocorrelation_test_helper(perm1a.to(dtype))
                        assert (
                            perm1a[perm1b] == arange
                        ).all(), "Wrong permutation inverse?"
                        assert (
                            perm1b[perm1a] == arange
                        ).all(), "Wrong permutation inverse? (commuted)"
                        assert torch.allclose(
                            perm1a, perm2, atol=tol
                        ), "Same seed, different noise? (permutation)"
                        assert not torch.allclose(
                            perm1a, perm3, atol=tol
                        ), "Different seed, same noise? (permutation)"
                        # complex noise
                        if dtype in COMPLEX_DTYPES:
                            # phase noise
                            noise1 = phase_noise(
                                dims, sd, dtype, device, conj=False
                            )
                            noise2 = phase_noise(
                                dims, sd, dtype, device, conj=False
                            )
                            noise3 = phase_noise(
                                dims, sd + 1, dtype, device, conj=False
                            )
                            autocorrelation_test_helper(noise1)
                            assert torch.allclose(
                                noise1, noise2, atol=tol
                            ), "Same seed, different noise? (uniform)"
                            assert not torch.allclose(
                                noise1, noise3, atol=tol
                            ), "Different seed, same noise? (uniform)"
                            assert torch.allclose(
                                noise1.conj(),
                                phase_noise(
                                    dims, sd, dtype, device, conj=True
                                ),
                                atol=tol,
                            ), "phase_noise not conjugating correctly?"
                            # phase shift
                            ones = torch.ones(dims, dtype=dtype, device=device)
                            noise1, m1 = phase_shift(
                                ones,
                                sd,
                                inplace=False,
                                rng_device=device,
                                conj=False,
                            )
                            noise2, _ = phase_shift(
                                ones,
                                sd,
                                inplace=False,
                                rng_device=device,
                                conj=False,
                            )
                            noise3, _ = phase_shift(
                                ones,
                                sd + 1,
                                inplace=False,
                                rng_device=device,
                                conj=False,
                            )
                            autocorrelation_test_helper(noise1)
                            assert torch.allclose(
                                noise1, noise2, atol=tol
                            ), "Same seed, different noise? (phase shift)"
                            assert not torch.allclose(
                                noise1, noise3, atol=tol
                            ), "Different seed, same noise? (phase shift)"
                            assert (
                                noise1 == m1
                            ).all(), "Inconsistent output and phase shift?"
                            #
                            noise1_conj, m1_conj = phase_shift(
                                ones,
                                sd,
                                inplace=False,
                                rng_device=device,
                                conj=True,
                            )
                            assert torch.allclose(
                                noise1.conj(),
                                noise1_conj,
                                atol=tol,
                            ), "incorrect phase_shift conjutage output?"
                            assert torch.allclose(
                                m1.conj(),
                                m1_conj,
                                atol=tol,
                            ), "phase_shift not conjugating mask correctly?"


# ##############################################################################
# # MATRIX ROUTINE WRAPPERS
# ##############################################################################
def test_qr(rng_seeds, torch_devices, dtypes_tols):
    """Test case for QR wrapper (formal and correctness)"""
    # fat matrices trigger error
    with pytest.raises(ValueError):
        _ = qr(torch.zeros((5, 10)), in_place_q=False, return_R=False)
    with pytest.raises(ValueError):
        _ = qr(torch.zeros((5, 10)), in_place_q=True, return_R=False)
    with pytest.raises(ValueError):
        _ = qr(torch.zeros((5, 10)), in_place_q=False, return_R=True)
    #
    hw = (20, 5)
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                tnsr = gaussian_noise(hw, 0, 1, seed, dtype, device)
                arr = tnsr.cpu().numpy()
                Itnsr = torch.eye(hw[1], dtype=dtype, device=device)
                Iarr = Itnsr.cpu().numpy()
                Q1, R1 = qr(tnsr, in_place_q=False, return_R=True)
                Q2, R2 = qr(arr, in_place_q=False, return_R=True)
                # test correctness of QR decomposition
                assert torch.allclose(
                    tnsr, Q1 @ R1, atol=tol
                ), "Incorrect torch QR?"
                assert np.allclose(
                    arr, Q2 @ R2, atol=tol
                ), "Incorrect numpy QR?"
                # test orthogonality of Q
                assert torch.allclose(
                    Itnsr, Q1.H @ Q1, atol=tol
                ), "Torch Q not orthogonal?"
                assert np.allclose(
                    Iarr, Q2.conj().T @ Q2, atol=tol
                ), "Numpy Q not orthogonal?"
                # test correctness of in-place
                Q1b = qr(tnsr, in_place_q=True, return_R=False)
                Q2b = qr(arr, in_place_q=True, return_R=False)
                assert Q1b is tnsr, "In-place not returning tensor!"
                assert Q2b is arr, "In-place not returning array!"
                assert torch.allclose(
                    tnsr, Q1b, atol=tol
                ), "Incorrect torch in-place QR?"
                assert np.allclose(
                    arr, Q2b, atol=tol
                ), "Incorrect torch in-place QR?"


def test_pinv(rng_seeds, torch_devices, dtypes_tols):
    """Test case for QR wrapper (formal and correctness)"""
    hw = (10, 10)
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                tnsr = gaussian_noise(hw, 0, 1, seed, dtype, device)
                arr = tnsr.cpu().numpy()
                Itnsr = torch.eye(hw[1], dtype=dtype, device=device)
                Iarr = Itnsr.cpu().numpy()
                #
                tnsr_inv = pinv(tnsr)
                arr_inv = pinv(arr)
                #
                assert torch.allclose(
                    Itnsr, tnsr @ tnsr_inv, atol=tol
                ), "Incorrect torch pinv?"
                assert torch.allclose(
                    Itnsr, tnsr_inv @ tnsr, atol=tol
                ), "Incorrect torch pinv (adj)?"
                assert np.allclose(
                    Iarr, arr @ arr_inv, atol=tol
                ), "Incorrect numpy pinv?"
                assert np.allclose(
                    Iarr, arr_inv @ arr, atol=tol
                ), "Incorrect numpy pinv (adj)?"
