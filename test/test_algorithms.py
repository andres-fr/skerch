#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`algorithms`."""


import pytest
import torch
import numpy as np

from skerch.utils import COMPLEX_DTYPES
from skerch.synthmat import RandomLordMatrix
from skerch.algorithms import ssvd, seigh, xdiag
from . import rng_seeds, torch_devices, max_mp_workers
from . import svd_test_helper, eigh_test_helper


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 1e-3,
        torch.complex64: 1e-3,
        torch.float64: 3e-8,
        torch.complex128: 3e-8,
    }
    return result


@pytest.fixture
def ssvd_recovery_shapes(request):
    """Tuples in the form ``((height, width), rank, outermeas, innermeas)."""
    result = [
        ((30, 30), 5, 15, 20),
        ((20, 30), 5, 15, 20),
        ((30, 20), 5, 15, 20),
        ((200, 150), 20, 35, 40),
    ]
    if request.config.getoption("--quick"):
        result = result[:3]
    return result


@pytest.fixture
def seigh_recovery_shapes(request):
    """Tuples in the form ``(dims, rank, outermeas, innermeas)."""
    result = [
        (30, 5, 15, 20),
        (200, 20, 35, 40),
    ]
    if request.config.getoption("--quick"):
        result = result[:1]
    return result


@pytest.fixture
def xdiag_recovery_shapes(request):
    """Tuples in the form ``(dims, rank, outermeas)."""
    result = [
        (30, 5, 15),
        (200, 20, 35),
    ]
    if request.config.getoption("--quick"):
        result = result[:1]
    return result


@pytest.fixture
def noise_types():
    """Collection of tuples ``(noise_type, is_complex_only)``"""
    result = [
        ("rademacher", False),
        ("gaussian", False),
        ("ssrft", False),
        ("phase", True),
    ]
    return result


# ##############################################################################
# # HELPERS
# ##############################################################################
class BasicMatrixLinOp:
    """Intentionally simple linop, only supporting ``shape`` and @."""

    def __init__(self, matrix):
        """ """
        self.matrix = matrix
        self.shape = matrix.shape

    def __matmul__(self, x):
        """ """
        return self.matrix @ x

    def __rmatmul__(self, x):
        """ """
        return x @ self.matrix


# ##############################################################################
# # DISPATCHERS
# ##############################################################################


# ##############################################################################
# # SSVD
# ##############################################################################
def test_ssvd_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    ssvd_recovery_shapes,
    noise_types,
    max_mp_workers,
):
    """Correctness test case for SSVD:

    Runs SSVD on all devices/dtypes/noisemats/recoveries, on a few low-rank
    linops, and tests that:

    * The SSVD is actually close to the matrix
    * ``U, V`` have orthonormal columns
    * Recovered ``S`` is a vector
    * The recovered ``S`` are nonnegative and in descending order
    * The devices and dtypes all match
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for hw, rank, outermeas, innermeas in ssvd_recovery_shapes:
                    mat, _ = RandomLordMatrix.exp(
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
                    I = torch.eye(outermeas, dtype=dtype, device=device)
                    lop = BasicMatrixLinOp(mat)
                    #
                    for noise_type, complex_only in noise_types:
                        if dtype not in COMPLEX_DTYPES and complex_only:
                            # this noise type does not support reals,
                            # skip this iteration
                            continue
                        for recovery_type in (
                            "singlepass",
                            "nystrom",
                            f"oversampled_{innermeas}",
                        ):
                            errmsg = (
                                "SSVD error! "
                                "{(seed, device, dtype, (hw, rank, outermeas, "
                                "innermeas), noise_type, recovery_type)})"
                            )
                            # run SSVD
                            U, S, Vh = ssvd(
                                lop,
                                device,
                                dtype,
                                outermeas,
                                seed + 1,
                                noise_type,
                                recovery_type,
                                max_mp_workers=max_mp_workers,
                            )
                            # test that output is correct and SVD-like
                            try:
                                svd_test_helper(mat, I, U, S, Vh, tol)
                            except AssertionError as ae:
                                raise AssertionError(errmsg) from ae


# ##############################################################################
# # SEIGH
# ##############################################################################
def test_seigh_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    seigh_recovery_shapes,
    noise_types,
    max_mp_workers,
):
    """Correctness test case for SEIGH:

    Runs SEIGH on all devices/dtypes/noisemats/recoveries, on a few low-rank
    linops, and tests that:

    * The EIGH is actually close to the matrix
    * Eigenvectors are orthonormal columns
    * Recovered eigvals given as a vector
    * The recovered eigvals are by descending magnitude/value
    * The devices and dtypes all match
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for dims, rank, outermeas, innermeas in seigh_recovery_shapes:
                    hw = (dims, dims)
                    mat, _ = RandomLordMatrix.exp(
                        hw,
                        rank,
                        decay=100,
                        diag_ratio=0.0,
                        symmetric=True,
                        psd=False,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    I = torch.eye(outermeas, dtype=dtype, device=device)
                    lop = BasicMatrixLinOp(mat)
                    #
                    for noise_type, complex_only in noise_types:
                        if dtype not in COMPLEX_DTYPES and complex_only:
                            # this noise type does not support reals,
                            # skip this iteration
                            continue
                        for recovery_type in (
                            "singlepass",
                            "nystrom",
                            f"oversampled_{innermeas}",
                        ):
                            # run SEIGh
                            for by_mag in (True, False):
                                ews, evs = seigh(
                                    lop,
                                    device,
                                    dtype,
                                    outermeas,
                                    seed + 1,
                                    noise_type,
                                    recovery_type,
                                    max_mp_workers=max_mp_workers,
                                    by_mag=by_mag,
                                )
                            # test that output is correct and SVD-like
                            try:
                                eigh_test_helper(mat, I, ews, evs, tol, by_mag)
                            except AssertionError as ae:
                                matconf = (hw, rank, outermeas, innermeas)
                                errmsg = (
                                    f"SEIGH error! (by_mag={by_mag}, {seed}, "
                                    f"{device}, {dtype}, {matconf}, "
                                    f"{noise_type}, {recovery_type})"
                                )
                                raise AssertionError(errmsg) from ae


# ##############################################################################
# # XDIAG/DIAGPP
# ##############################################################################
def test_xdiag_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    xdiag_recovery_shapes,
    noise_types,
    max_mp_workers,
):
    """Correctness test case for XDIAG/DIAGPP:

    Runs XDIAG on all devices/dtypes/noisemats/recoveries, on a few low-rank
    linops, and tests that:


    * The SSVD is actually close to the matrix
    * ``U, V`` have orthonormal columns
    * Recovered ``S`` is a vector
    * The recovered ``S`` are nonnegative and in descending order
    * The devices and dtypes all match
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for (
                    dims,
                    rank,
                    outermeas,
                ) in xdiag_recovery_shapes:
                    hw = (dims, dims)
                    mat, _ = RandomLordMatrix.exp(
                        hw,
                        rank,
                        decay=0,
                        diag_ratio=0.0,
                        symmetric=False,
                        psd=False,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    lop = BasicMatrixLinOp(mat)
                    diag = mat.diag()
                    #
                    for noise_type, complex_only in noise_types:
                        if dtype not in COMPLEX_DTYPES and complex_only:
                            # this noise type does not support reals,
                            # skip this iteration
                            continue
                        #
                        diag, _ = xdiag(
                            lop,
                            device,
                            dtype,
                            outermeas,
                            seed,
                            noise_type,
                            max_mp_workers,
                            diagpp=False,
                        )
                        breakpoint()
                        errmsg = (
                            "SSVD error! "
                            "{(seed, device, dtype, (hw, rank, outermeas, "
                            "innermeas), noise_type, recovery_type)})"
                        )
                        # run SSVD
                        U, S, Vh = ssvd(
                            lop,
                            device,
                            dtype,
                            outermeas,
                            seed + 1,
                            noise_type,
                            recovery_type,
                            max_mp_workers=max_mp_workers,
                        )
                        # test that output is correct and SVD-like
                        try:
                            svd_test_helper(mat, I, U, S, Vh, tol)
                        except AssertionError as ae:
                            raise AssertionError(errmsg) from ae
