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
from skerch.utils import qr, pinv, lstsq, svd, eigh, htr
from skerch.utils import subdiag_hadamard_pattern, serrated_hadamard_pattern
from skerch.utils import truncate_decomp

from . import rng_seeds, torch_devices
from . import autocorrelation_test_helper, svd_test_helper, eigh_test_helper


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
def dtypes_tols_badcond():
    """Error tolerances for each dtype (f32 looser due to pinv/lstsq)"""
    result = {
        torch.float32: 1e-2,  # not very small due to pinv/lstsq
        torch.complex64: 1e-2,  # not very small due to pinv/lstsq
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


@pytest.fixture
def hadamard_testcases():
    """Inputs and expected outputs for subdiag and serrated hadamard.

    Returns tuples in the form:
    ``(input, sub1, sub2, sub3, sub4, serr1, serr2, serr3, serr4)``, where:
    * ``sub1`` contains subdiag outputs for idxs=[i]
    * ``sub2`` contains subdiag outputs for idxs=[-1]
    * ``sub3`` contains subdiag outputs for idxs=[0, ..., i]
    * ``sub4`` contains subdiag outputs for idxs=[0, ..., -i]
    * ``serr1`` contains serrated outputs for blocksize=i, lower, with diag
    * ``serr1`` contains serrated outputs for blocksize=i, lower, without diag
    * ``serr1`` contains serrated outputs for blocksize=i, upper, with diag
    * ``serr1`` contains serrated outputs for blocksize=i, upper, without diag
    """
    case1 = (
        [1, 2, 3, 4, 5, 6, 7],
        # subdiag: idx=i
        [
            [1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 0, 1, 2, 3, 4, 5],
            [0, 0, 0, 1, 2, 3, 4],
            [0, 0, 0, 0, 1, 2, 3],
            [0, 0, 0, 0, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        # subdiag: idx=-1
        [
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 0],
            [3, 4, 5, 6, 7, 0, 0],
            [4, 5, 6, 7, 0, 0, 0],
            [5, 6, 7, 0, 0, 0, 0],
            [6, 7, 0, 0, 0, 0, 0],
            [7, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        # subdiag: idxs = 0, ... i
        [
            [1, 2, 3, 4, 5, 6, 7],
            [1, 3, 5, 7, 9, 11, 13],
            [1, 3, 6, 9, 12, 15, 18],
            [1, 3, 6, 10, 14, 18, 22],
            [1, 3, 6, 10, 15, 20, 25],
            [1, 3, 6, 10, 15, 21, 27],
            [1, 3, 6, 10, 15, 21, 28],
            [1, 3, 6, 10, 15, 21, 28],
        ],
        # subdiag: idxs = 0, ... -i
        [
            [1, 2, 3, 4, 5, 6, 7],
            [3, 5, 7, 9, 11, 13, 7],
            [6, 9, 12, 15, 18, 13, 7],
            [10, 14, 18, 22, 18, 13, 7],
            [15, 20, 25, 22, 18, 13, 7],
            [21, 27, 25, 22, 18, 13, 7],
            [28, 27, 25, 22, 18, 13, 7],
            [28, 27, 25, 22, 18, 13, 7],
        ],
        # serrated: lower, with diag
        [
            [1, 2, 3, 4, 5, 6, 7],  # blocksize=1
            [1, 3, 3, 7, 5, 11, 7],  # blocksize=2
            [1, 3, 6, 4, 9, 15, 7],  # blocksize=3
            [1, 3, 6, 10, 5, 11, 18],
            [1, 3, 6, 10, 15, 6, 13],
            [1, 3, 6, 10, 15, 21, 7],
            [1, 3, 6, 10, 15, 21, 28],
        ],
        # serrated: lower, without diag (like z1 but subtractin 1st row)
        [
            [0, 0, 0, 0, 0, 0, 0],  # blocksize=1
            [0, 1, 0, 3, 0, 5, 0],  # blocksize=2
            [0, 1, 3, 0, 4, 9, 0],  # blocksize=3
            [0, 1, 3, 6, 0, 5, 11],
            [0, 1, 3, 6, 10, 0, 6],
            [0, 1, 3, 6, 10, 15, 0],
            [0, 1, 3, 6, 10, 15, 21],
        ],
        # serrated: upper, with diag
        [
            [1, 2, 3, 4, 5, 6, 7],  # blocksize=1
            [3, 2, 7, 4, 11, 6, 7],  # blocksize=2
            [6, 5, 3, 15, 11, 6, 7],  # blocksize=3
            [10, 9, 7, 4, 5, 6, 7],
            [15, 14, 12, 9, 5, 6, 7],
            [21, 20, 18, 15, 11, 6, 7],
            [28, 27, 25, 22, 18, 13, 7],
        ],
        # serrated: upper, without diag
        [
            [0, 0, 0, 0, 0, 0, 0],  # blocksize=1
            [2, 0, 4, 0, 6, 0, 0],  # blocksize=2
            [5, 3, 0, 11, 6, 0, 0],  # blocksize=3
            [9, 7, 4, 0, 0, 0, 0],
            [14, 12, 9, 5, 0, 0, 0],
            [20, 18, 15, 11, 6, 0, 0],
            [27, 25, 22, 18, 13, 7, 0],
        ],
    )
    #
    result = [case1]
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
    # unknown dtype:
    with pytest.raises(ValueError):
        complex_dtype_to_real("float1234567891011")


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
                            # phase shift: inplace
                            phase_shift(
                                ones,
                                sd,
                                inplace=True,
                                rng_device=device,
                                conj=False,
                            )
                            assert (
                                ones == noise1
                            ).all(), "phase_shift not inplace?"


# ##############################################################################
# # MATRIX ROUTINE WRAPPERS
# ##############################################################################
def test_qr(rng_seeds, torch_devices, dtypes_tols):
    """Test case for QR wrapper (formal and correctness).

    * Fat matrices trigger error
    * ``Q @ R`` equals given matrix
    * ``Q`` is orthogonal
    * In-place operation yields correct result
    * Output is in matching device and dtype
    """
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
                # test R is upper triangular
                assert torch.allclose(
                    R1, R1.triu(), atol=tol
                ), "Torch R not upper triangular?"
                assert np.allclose(
                    R2, np.triu(R2), atol=tol
                ), "Numpy R not upper triangular?"
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
                # matching device and dtype
                assert Q1.device == tnsr.device, "Incorrect torch Q device?"
                assert R1.device == tnsr.device, "Incorrect torch R device?"
                assert Q1.dtype == tnsr.dtype, "Incorrect torch Q dtype?"
                assert R1.dtype == tnsr.dtype, "Incorrect torch R dtype?"
                assert Q2.dtype == arr.dtype, "Incorrect numpy Q dtype?"
                assert R2.dtype == arr.dtype, "Incorrect numpy R dtype?"


def test_pinv_lstsq(rng_seeds, torch_devices, dtypes_tols_badcond):
    """Test case for matrix inversion wrappers (formal and correctness).

    * ``pinv`` yields the correct inverse
    * ``lstsq`` yields equivalent result to multiplying with ``pinv``.
    * Output is in matching device and dtype
    """
    hw = (10, 10)
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols_badcond.items():
                tnsr = gaussian_noise(hw, 0, 1, seed, dtype, device)
                arr = tnsr.cpu().numpy()
                Itnsr = torch.eye(hw[1], dtype=dtype, device=device)
                Iarr = Itnsr.cpu().numpy()
                # pinv correctness
                tnsr_inv = pinv(tnsr)
                arr_inv = pinv(arr)
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
                # lstsq correctness
                # tnsr2 = gaussian_noise(hw, 0, 1, seed + 1, dtype, device)
                tnsr2 = torch.eye(len(tnsr), dtype=dtype, device=device)
                arr2 = tnsr2.cpu().numpy()
                tinv2 = lstsq(tnsr, tnsr2)
                ainv2 = lstsq(arr, arr2)
                assert torch.allclose(
                    tinv2, tnsr_inv @ tnsr2, atol=tol
                ), "Incorrect torch lstsq?"
                assert np.allclose(
                    ainv2, arr_inv @ arr2, atol=tol
                ), "Incorrect numpy lstsq?"
                # matching device and dtype
                assert (
                    tnsr_inv.device == tnsr.device
                ), "Incorrect torch pinv device?"
                assert (
                    tnsr_inv.dtype == tnsr.dtype
                ), "Incorrect torch pinv dtype?"
                assert (
                    arr_inv.dtype == arr.dtype
                ), "Incorrect numpy pinv dtype?"
                #
                assert (
                    tinv2.device == tnsr.device
                ), "Incorrect torch lstsq device?"
                assert (
                    tinv2.dtype == tnsr.dtype
                ), "Incorrect torch lstsq dtype?"
                assert ainv2.dtype == arr.dtype, "Incorrect numpy lstsq dtype?"


def test_svd(rng_seeds, torch_devices, dtypes_tols):
    """Test case for singular val decomp wrapper (formal and correctness)

    * Singular bases are orthogonal
    * Singular values are given as real vectors of nonascending nonneg values
    * ``U @ diag(S) @ Vh`` equals given matrix
    * Output is in matching device and dtype (svals are always floats)
    """
    hw = (10, 10)
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                tnsr = gaussian_noise(hw, 0, 1, seed, dtype, device)
                arr = tnsr.cpu().numpy()
                Itnsr = torch.eye(min(hw), dtype=dtype, device=device)
                Iarr = Itnsr.cpu().numpy()
                #
                u1, s1, vh1 = svd(tnsr)
                u2, s2, vh2 = svd(arr)
                #
                try:
                    svd_test_helper(tnsr, Itnsr, u1, s1, vh1, tol)
                except AssertionError as ae:
                    raise AssertionError("Error in torch SVD!") from ae
                try:
                    svd_test_helper(arr, Iarr, u2, s2, vh2, tol)
                except AssertionError as ae:
                    raise AssertionError("Error in numpy SVD!") from ae


def test_eigh(rng_seeds, torch_devices, dtypes_tols):
    """Test case for Hermitian eigdecomp wrapper (formal and correctness)

    * Eigenbasis is orthogonal
    * Eigenvalues are given as real vectors by descending magnitude/value
    * ``Q @ diag(Lambda) @ Q.H`` equals given matrix
    * Output is in matching device and dtype
    """
    hw = (10, 10)
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                tnsr = gaussian_noise(hw, 0, 1, seed, dtype, device)
                tnsr = tnsr + tnsr.H  # make it hermitian
                arr = tnsr.cpu().numpy()
                Itnsr = torch.eye(min(hw), dtype=dtype, device=device)
                Iarr = Itnsr.cpu().numpy()
                for by_mag in (True, False):
                    ews1, evs1 = eigh(tnsr, by_descending_magnitude=by_mag)
                    ews2, evs2 = eigh(arr, by_descending_magnitude=by_mag)
                    try:
                        eigh_test_helper(tnsr, Itnsr, ews1, evs1, tol, by_mag)
                    except AssertionError as ae:
                        raise AssertionError("Error in torch EIGH!") from ae
                    try:
                        eigh_test_helper(arr, Iarr, ews2, evs2, tol, by_mag)
                    except AssertionError as ae:
                        raise AssertionError("Error in numpy EIGH!") from ae


def test_htr(rng_seeds, torch_devices, dtypes_tols):
    """Test case for Hermitian eigdecomp wrapper (formal and correctness)

    * Works for torch and numpy
    * Works for vectors and matrices with no warning
    * force_copy returns a brand new copy
    """
    h, w = (10, 2)
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                m1 = gaussian_noise((h, w), 0, 1, seed, dtype, device)
                v1 = gaussian_noise(h, 0, 1, seed + 1, dtype, device)
                m2 = m1.cpu().numpy()
                v2 = v1.cpu().numpy()
                # copy transpositions
                m1H = htr(m1, in_place=False)
                v1H = htr(v1, in_place=False)
                m2H = htr(m2, in_place=False)
                v2H = htr(v2, in_place=False)
                # correctness
                assert (
                    m1.H == m1H
                ).all(), "Incorrect torch mat transp! (inplace=False)"
                assert (
                    v1.conj() == v1H
                ).all(), "Incorrect torch vec transp! (inplace=False)"
                assert (
                    m2.conj().T == m2H
                ).all(), "Incorrect np mat transp! (inplace=False)"
                assert (
                    v2.conj() == v2H
                ).all(), "Incorrect np vec transp! (inplace=False)"
                # check that a new copy was returned: modify the returned H and
                # check it is different to original
                m1H += 1
                v1H += 1
                m2H += 1
                v2H += 1
                assert (m1.conj().T != m1H).all(), "Forcecopy torch mat error?"
                assert (v1.conj() != v1H).all(), "Forcecopy torch vec error?"
                assert (m2.conj().T != m2H).all(), "Forcecopy np mat error?"
                assert (v2.conj() != v2H).all(), "Forcecopy np vec error?"
                # in-place transpositions
                m1b = htr(m1, in_place=True)
                v1b = htr(v1, in_place=True)
                m2b = htr(m2, in_place=True)
                v2b = htr(v2, in_place=True)
                # correctness: ori.T = ori_T_conj because conj was inplace
                assert (
                    m1.T == m1b
                ).all(), "Incorrect torch mat transp! (inplace=True)"
                assert (
                    v1 == v1b
                ).all(), "Incorrect torch vec transp! (inplace=True)"
                assert (
                    m2.T == m2b
                ).all(), "Incorrect np mat transp! (inplace=True)"
                assert (
                    v2 == v2b
                ).all(), "Incorrect np vec transp! (inplace=True)"
                # check that just a view was returned: modify the returned H
                # and correctness tests still hold
                m1b += 1
                v1b += 1
                m2b += 1
                v2b += 1
                assert (m1.T == m1b).all(), "View torch mat error?"
                assert (v1 == v1b).all(), "View torch vec error?"
                assert (m2.T == m2b).all(), "View np mat error?"
                assert (v2 == v2b).all(), "View np vec error?"


# ##############################################################################
# # MEASUREMENT HADAMARD PATTERNS
# ##############################################################################
def test_hadamard_patterns(dtypes_tols, torch_devices, hadamard_testcases):
    """Test case for measurement Hadamard patterns in ``utils``.

    For all devices and datatypes, with and without FFT, sample a random vector
    and check that ``subdiag_hadamard_pattern`` produces the right shift for
    all possible idxs.

    Then, check that ``serrated_hadamard_pattern`` produces the right block
    shifts for all combinations:
    * blocksize
    * with_main_diagonal
    * lower
    * use_fft
    """
    for device in torch_devices:
        for dtype, tol in dtypes_tols.items():
            for x, y1, y2, y3, y4, z1, z2, z3, z4 in hadamard_testcases:
                dims = len(x)
                x = torch.tensor(x, dtype=dtype, device=device)
                y1 = torch.tensor(y1, dtype=dtype, device=device)
                y2 = torch.tensor(y2, dtype=dtype, device=device)
                y3 = torch.tensor(y3, dtype=dtype, device=device)
                y4 = torch.tensor(y4, dtype=dtype, device=device)
                z1 = torch.tensor(z1, dtype=dtype, device=device)
                z2 = torch.tensor(z2, dtype=dtype, device=device)
                z3 = torch.tensor(z3, dtype=dtype, device=device)
                z4 = torch.tensor(z4, dtype=dtype, device=device)
                if dtype in COMPLEX_DTYPES:
                    x = x + 1j * x
                    y1 = y1 + 1j * y1
                    y2 = y2 + 1j * y2
                    y3 = y3 + 1j * y3
                    y4 = y4 + 1j * y4
                    z1 = z1 + 1j * z1
                    z2 = z2 + 1j * z2
                    z3 = z3 + 1j * z3
                    z4 = z4 + 1j * z4
                # empty idxs list raises error
                with pytest.raises(ValueError):
                    subdiag_hadamard_pattern(x, [])
                # nonpositive blocksize raises error
                with pytest.raises(ValueError):
                    serrated_hadamard_pattern(x, 0)
                with pytest.raises(ValueError):
                    serrated_hadamard_pattern(x, -1)
                with pytest.raises(ValueError):
                    serrated_hadamard_pattern(x, len(x) + 1)
                # test vectorized outputs
                xrep = x.unsqueeze(0).repeat(5, 1)
                xrep1a = subdiag_hadamard_pattern(xrep, [1], use_fft=False)
                xrep1b = subdiag_hadamard_pattern(xrep, [1], use_fft=True)
                xrep2 = serrated_hadamard_pattern(xrep, 3, use_fft=False)
                assert (
                    xrep1a - y1[1]
                ).norm() < tol, "Inconsistent subdiag vectorization!"
                assert (
                    xrep1b - y1[1]
                ).norm() < tol, "Inconsistent subdiag vectorization! (fft)"
                assert (
                    xrep2 - z1[2]
                ).norm() < tol, "Inconsistent serrated vectorization!"
                # subdiag_hadamard_pattern
                for i in range(dims + 1):
                    for fft in (False, True):
                        msg = " (FFT)" if fft else ""
                        # subdiag with i produces a right shift
                        w = subdiag_hadamard_pattern(x, [i], use_fft=fft)
                        assert torch.allclose(y1[i], w, atol=tol), (
                            "Wrong subdiag positive shift!" + msg
                        )
                        # subdiag with -i produces a left shift
                        w = subdiag_hadamard_pattern(x, [-i], use_fft=fft)
                        assert torch.allclose(y2[i], w, atol=tol), (
                            "Wrong subdiag negative shift!" + msg
                        )
                        # subdiag with 0,1...i is a right-cumulative shift
                        idxs = torch.arange(i + 1)
                        w = subdiag_hadamard_pattern(x, idxs, use_fft=fft)
                        assert torch.allclose(y3[i], w, atol=tol), (
                            "Wrong cumulative positive shift!" + msg
                        )
                        # subdiag with 0,-1...-i is a left-cumulative shift
                        w = subdiag_hadamard_pattern(x, -idxs, use_fft=fft)
                        assert torch.allclose(y4[i], w, atol=tol), (
                            "Wrong cumulative negative shift!" + msg
                        )
                # serrated_hadamard_pattern
                for i in range(dims):
                    for fft in (False,):
                        msg = " (FFT)" if fft else ""
                        # serrated pattern: lower, with main diag
                        w = serrated_hadamard_pattern(
                            x,
                            i + 1,
                            with_main_diagonal=True,
                            lower=True,
                            use_fft=fft,
                        )
                        assert torch.allclose(z1[i], w, atol=tol), (
                            "Wrong serrated lower with main diag!" + msg
                        )
                        # serrated pattern: lower, without main diag
                        w = serrated_hadamard_pattern(
                            x,
                            i + 1,
                            with_main_diagonal=False,
                            lower=True,
                            use_fft=fft,
                        )
                        assert torch.allclose(z2[i], w, atol=tol), (
                            "Wrong serrated lower without main diag!" + msg
                        )
                        # serrated pattern: upper, with main diag
                        w = serrated_hadamard_pattern(
                            x,
                            i + 1,
                            with_main_diagonal=True,
                            lower=False,
                            use_fft=fft,
                        )
                        assert torch.allclose(z3[i], w, atol=tol), (
                            "Wrong serrated upper with main diag!" + msg
                        )
                        # serrated pattern: upper, without main diag
                        w = serrated_hadamard_pattern(
                            x,
                            i + 1,
                            with_main_diagonal=False,
                            lower=False,
                            use_fft=fft,
                        )
                        assert torch.allclose(z4[i], w, atol=tol), (
                            "Wrong serrated upper without main diag!" + msg
                        )


# ##############################################################################
# # RECOVERY UTILS
# ##############################################################################
@pytest.fixture
def truncate_testcases():
    """Inputs and expected outputs for subdiag and serrated hadamard.

    Returns tuples in the form ``(shape, k)``
    """
    result = [
        ((20, 20), 10),
    ]
    return result


def test_truncate_decomp(dtypes_tols, torch_devices, truncate_testcases):
    """Test case for measurement Hadamard patterns in ``utils``.

    For all devices and datatypes, with and without FFT, sample a random vector
    and check that ``subdiag_hadamard_pattern`` produces the right shift for
    all possible idxs.


    * k <= raises error
    * Output has expected shape and content
    * Modifying outputs doesn't alter inputs if copy=True, does otherwise
    """
    import itertools

    for device in torch_devices:
        for dtype, tol in dtypes_tols.items():
            for shape, k in truncate_testcases:
                mat = gaussian_noise(
                    shape,
                    0,
                    1,
                    seed=0,
                    dtype=dtype,
                    device=device,
                )
                U, S, Vh = torch.linalg.svd(mat)
                # k <= 0 raises error
                with pytest.raises(ValueError):
                    _ = truncate_decomp(0, U, S, Vh)
                # output has expected shape and content
                for u, s, vh in itertools.product(
                    *((x, None) for x in (U, S, Vh))
                ):
                    u_out, s_out, vh_out = truncate_decomp(k, u, s, vh)
                    #
                    if u is None:
                        assert u_out is None, "u None u_out not None?"
                    else:
                        assert (u_out == u[:, :k]).all(), "incorrect u_out?"
                    #
                    if s is None:
                        assert s_out is None, "s None s_out not None?"
                    else:
                        assert (s_out == s[:k]).all(), "incorrect s_out?"
                    #
                    if vh is None:
                        assert vh_out is None, "vh None vh_out not None?"
                    else:
                        assert (vh_out == vh[:k, :]).all(), "incorrect vh_out?"
                # modifying by-copy doesn't alter input
                u_out, s_out, vh_out = truncate_decomp(
                    max(shape), U, S, Vh, copy=True
                )
                u_out *= 0
                s_out *= 0
                vh_out *= 0
                assert torch.dist(U, u_out) == torch.norm(
                    U
                ), "U modified by-copy?"
                assert torch.dist(S, s_out) == torch.norm(
                    S
                ), "S modified by-copy?"
                assert torch.dist(Vh, vh_out) == torch.norm(
                    Vh
                ), "Vh modified by-copy?"
                # modifying by-reference alters input
                u_out, s_out, vh_out = truncate_decomp(
                    max(shape), U, S, Vh, copy=False
                )
                u_out *= 0
                s_out *= 0
                vh_out *= 0
                assert torch.norm(U) == 0, "U not modified by-ref?"
                assert torch.norm(S) == 0, "S not modified by-ref?"
                assert torch.norm(Vh) == 0, "Vh not modified by-ref?"
