#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for noisy measurements.

* IID (Gaussian, Rademacher, Phase)
* SSRFT (transform and linop)
"""


import time
import warnings
from functools import partial

import pytest
import torch

from skerch.linops import CompositeLinOp, TransposedLinOp, linop_to_matrix
from skerch.measurements import (
    SSRFT,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
    RademacherNoiseLinOp,
    SsrftNoiseLinOp,
)
from skerch.utils import (
    COMPLEX_DTYPES,
    BadSeedError,
    BadShapeError,
    gaussian_noise,
)

from . import (
    autocorrelation_test_helper,
    rng_seeds,
    torch_devices,
)


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
    hw = (50, 50)
    delta_at_least = 0.7
    nondelta_at_most = 0.5
    return hw, delta_at_least, nondelta_at_most


@pytest.fixture
def ssrft_hw_and_autocorr_tolerances():
    """Error tolerances for each complex dtype."""
    hw = (50, 40)
    delta_at_least = 0.7
    nondelta_at_most = 0.5
    return hw, delta_at_least, nondelta_at_most


# ##############################################################################
# # HELPERS
# ##############################################################################
class BasicMatrixLinOp:
    """Intentionally simple linop, only supporting ``shape`` and @.

    :param delay: Artificially introduced when computing  ``@``, in seconds.
    """

    def __init__(self, matrix, delay=0):
        """ """
        self.matrix = matrix
        self.shape = matrix.shape
        self.delay = delay

    def __matmul__(self, x):
        """ """
        if self.delay > 0:
            time.sleep(self.delay)
        return self.matrix @ x

    def __rmatmul__(self, x):
        """ """
        if self.delay > 0:
            time.sleep(self.delay)
        return x @ self.matrix


# ##############################################################################
# # IID
# ##############################################################################
def test_iid_measurements_formal(
    rng_seeds, torch_devices, dtypes_tols, iid_noise_linop_types
):
    """Formal test case for iid measurement linops.

    For every iid noise linop tests:
    * Repr creates correct strings
    * Register triggers error if overlapping seeds are used if active
    * Get_vector triggers error for invalid index, and returns right dtype and
      device otherwise
    * Deterministic behaviour (fwd and adjoint): running twice is same
    * Seed consistency (including same linop in different device)
    * Output is in requested datatype and device

    For complex_only linops, it also tests:
    * Providing a noncomplex dtype raises a ``ValueError``
    """
    # correct string conversion
    hw = (3, 3)  # only matters for strings
    lop = RademacherNoiseLinOp(hw, 0, by_row=False, register=False)
    s = "<RademacherNoiseLinOp(3x3), by col, blocksize=1, seed=0>"
    assert str(lop) == s, "Unexpected repr for Rademacher noise linop!"
    lop = GaussianNoiseLinOp(hw, 0, by_row=False, register=False)
    s = (
        "<GaussianNoiseLinOp(3x3), by col, blocksize=1, seed=0, "
        "mean=0.0, std=1.0>"
    )
    assert str(lop) == s, "Unexpected repr for Gaussian noise linop!"
    lop = PhaseNoiseLinOp(hw, 0, by_row=False, register=False)
    s = "<PhaseNoiseLinOp(3x3), by col, blocksize=1, seed=0, conj=False>"
    assert str(lop) == s, "Unexpected repr for Phase noise linop!"
    #
    for lop_type, complex_only in iid_noise_linop_types:
        lop = lop_type((5, 5), seed=0, by_row=False)
        # register triggers for overlapping seeds regardless of other factors
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=0, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((20, 5), seed=1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, by_row=True)
        lop_type.REGISTER.clear()  # clear register for this lop_type
        # invalid index triggers error
        with pytest.raises(ValueError):
            lop.get_block(-1, torch.float32, "cpu")
        with pytest.raises(ValueError):
            lop.get_block(1e12, torch.float32, "cpu")
        #
        hw = (100, 2)
        for seed in rng_seeds:
            for dtype, tol in dtypes_tols.items():
                # complex-only linops raise value error for noncomplex dtype
                if complex_only and dtype not in {
                    torch.complex32,
                    torch.complex64,
                    torch.complex128,
                }:
                    clop = lop_type(hw, 0, register=False)
                    with pytest.raises(ValueError):
                        _ = clop @ torch.ones(hw[1], dtype=dtype)
                    continue
                # seed consistency across devices (if CUDA is available)
                if torch.cuda.is_available():
                    lop = lop_type(hw, seed, by_row=False, register=False)
                    mat1 = linop_to_matrix(lop, dtype, "cpu", adjoint=False)
                    mat2 = linop_to_matrix(lop, dtype, "cuda", adjoint=False)
                    mat3 = linop_to_matrix(lop, dtype, "cpu", adjoint=True)
                    mat4 = linop_to_matrix(lop, dtype, "cuda", adjoint=True)
                    assert torch.allclose(
                        mat1, mat2.cpu(), atol=tol
                    ), "Mismatching linop across devices!"
                    assert torch.allclose(
                        mat1, mat3.cpu(), atol=tol
                    ), "Mismatching linop across devices!"
                    assert torch.allclose(
                        mat1, mat4.cpu(), atol=tol
                    ), "Mismatching linop across devices!"
                else:
                    warnings.warn(
                        "Warning! cross-device tests didn't run "
                        "because CUDA is not available in this device",
                        stacklevel=2,
                    )
                # deterministic behaviour and seed consistency
                for device in torch_devices:
                    lop1 = lop_type(hw, seed, by_row=False, register=False)
                    lop2 = lop_type(hw, seed, by_row=False, register=False)
                    lop3 = lop_type(hw, seed + 5, by_row=False, register=False)
                    mat1a = linop_to_matrix(lop1, dtype, device, adjoint=False)
                    mat1b = linop_to_matrix(lop1, dtype, device, adjoint=False)
                    mat1c = linop_to_matrix(lop1, dtype, device, adjoint=True)
                    mat2 = linop_to_matrix(lop2, dtype, device, adjoint=False)
                    mat3 = linop_to_matrix(lop3, dtype, device, adjoint=False)
                    assert (
                        mat1a == mat1b
                    ).all(), f"Nondeterministic linop? {lop1}"
                    assert (
                        mat1a == mat1c
                    ).all(), f"Different fwd and adjoint? {lop1}"
                    assert (
                        mat1a == mat2
                    ).all(), f"Same seed, differentl linop? {lop1}"
                    #
                    for col in mat1a.H:
                        cosim = abs(col @ mat3) / (col.norm() ** 2)
                        assert (
                            cosim < 0.5
                        ).all(), "Different seeds, similar vectors? {lop1}"
                    # dtype and device check
                    assert mat1a.dtype == dtype, "Mismatching dtype!"
                    assert mat1a.device.type == device, "Mismatching device!"


def test_iid_measurements_correctness(  # noqa:C901
    rng_seeds,
    torch_devices,
    dtypes_tols,
    iid_noise_linop_types,
    iid_hw_and_autocorr_tolerances,
):
    """Test case for correctness of iid measurement linops.

    For each iid linop on all dtypes and devices tests that:
    * Columns/rows behave like iid noise (delta autocorrelation)
    * Fwd and adj matmul with Identity and onehot vectors lead to same linop
    * Correctness of to_matrix
    * Transposed linop is correct (fwd and adj)
    """
    hw, delta_at_least, nondelta_at_most = iid_hw_and_autocorr_tolerances
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                Ileft = torch.eye(hw[1], dtype=dtype, device=device)
                Iright = torch.eye(hw[0], dtype=dtype, device=device)
                for lop_type, complex_only in iid_noise_linop_types:
                    if complex_only and dtype not in {
                        torch.complex32,
                        torch.complex64,
                        torch.complex128,
                    }:
                        continue
                    #
                    for by_row in (False, True):
                        brs = "(by row)" if by_row else "(by col)"
                        lop = lop_type(
                            hw,
                            seed,
                            by_row=by_row,
                            register=False,
                            blocksize=max(hw),
                        )
                        mat1a = lop @ Iright
                        mat1b = lop @ Iright
                        mat1c = lop.to_matrix(dtype, device)
                        mat1d = linop_to_matrix(
                            lop, dtype, device, adjoint=False
                        )
                        mat1e = linop_to_matrix(
                            lop, dtype, device, adjoint=True
                        )
                        # Columns/rows behave like iid noise (delta autocorr)
                        for x in mat1a:  # x is a row
                            try:
                                autocorrelation_test_helper(
                                    x, delta_at_least, nondelta_at_most
                                )
                            except AssertionError as ae:
                                raise AssertionError(
                                    "IID autocorr error (row)"
                                ) from ae
                        for x in mat1a.H:  # x is a column
                            try:
                                autocorrelation_test_helper(
                                    x, delta_at_least, nondelta_at_most
                                )
                            except AssertionError as ae:
                                raise AssertionError(
                                    "IID autocorr error (col)"
                                ) from ae
                        # to_matrix, fwd and adj matmul lead to same linop
                        assert (
                            mat1a == mat1b
                        ).all(), f"Mismatching fwd/adj {lop_type} {brs}"
                        assert (
                            mat1a == mat1c
                        ).all(), f"Wrong to_matrix {lop_type} {brs}"
                        assert (
                            mat1a == mat1d
                        ).all(), f"Wrong linop_to_matrix {lop_type} {brs}"
                        assert (
                            mat1a == mat1e
                        ).all(), f"Wrong adj linop_to_matrix {lop_type} {brs}"
                        # transposed linop is correct
                        lopT = TransposedLinOp(lop)
                        matTa = lopT @ Ileft
                        matTb = Iright @ lopT
                        assert torch.allclose(
                            mat1a.H, matTa, atol=tol
                        ), "Wrong iid transposition?"
                        assert torch.allclose(
                            mat1a.H, matTb, atol=tol
                        ), "Wrong iid transposition? (adjoint)"


def test_phasenoise_conj_unit(rng_seeds, torch_devices, complex_dtypes_tols):
    """Test case for conjugation of ``PhaseNoise`` linop.

    For all seeds, devices and complex dtypes, tests that:
    * Conj of linop produces elementwise conjugated entries
    * linop entries have unit norm
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in complex_dtypes_tols.items():
                lop = PhaseNoiseLinOp((5, 5), seed, conj=False, register=False)
                lop_conj = PhaseNoiseLinOp(
                    (5, 5), seed, conj=True, register=False
                )
                mat = linop_to_matrix(lop, dtype, device, adjoint=False)
                mat_conj = linop_to_matrix(
                    lop_conj, dtype, device, adjoint=False
                )
                assert (mat.conj() == mat_conj).all(), "Wrong conj linop?"
                #
                prod = mat * mat_conj
                assert prod.imag.norm() <= tol, "wrong conjugations?"
                assert torch.allclose(
                    prod.real, torch.ones_like(prod.real), atol=tol
                ), "Phasenoise not unit norm?"


# ##############################################################################
# # SSRFT
# ##############################################################################
def test_ssrft_formal(rng_seeds, torch_devices, dtypes_tols):
    """Formal test case for SSRFT functionality.

    For the SSRFT transform and/or linop (wherever applicable), tests:
    * Repr creates correct strings
    * Register triggers error if overlapping seeds are used if active
    * Non-orthogonal normalization raises NotImplementederror
    * Too large or too small out_dims for ssrft
    * Invalid shape to SSRFT linop triggers error (must be square or fat)
    * Get_vector triggers error for invalid index, and returns right dtype and
      device otherwise
    * @ output is of same dtype and device as input
    * Deterministic behaviour (fwd and adjoint): running twice is same
    * Seed consistency
    """
    hw = (10, 10)
    # correct string conversion (linop)
    lop = SsrftNoiseLinOp(hw, 0, norm="ortho", register=False)
    s = "<SsrftNoiseLinOp(10x10), by col, blocksize=1, seed=0, norm=ortho>"
    assert str(lop) == s, "Unexpected repr for SSRFT noise linop!"
    # register overlap triggers error
    lop = SsrftNoiseLinOp((3, 3), seed=12345, register=True)
    with pytest.raises(BadSeedError):
        _ = SsrftNoiseLinOp((3, 3), seed=12345, register=True)
    SsrftNoiseLinOp.REGISTER.clear()
    # non-orthogonal norm raises error (transform)
    with pytest.raises(NotImplementedError):
        SSRFT.ssrft(torch.ones(10), out_dims=10, seed=0, norm="XXXX")
    with pytest.raises(NotImplementedError):
        SSRFT.issrft(torch.ones(10), out_dims=10, seed=0, norm="XXXX")
    # non-orthogonal norm raises error (linop)
    lop = SsrftNoiseLinOp(hw, 0, norm="XXXX", register=False)
    with pytest.raises(NotImplementedError):
        lop @ torch.ones(10)
    with pytest.raises(NotImplementedError):
        torch.ones(10) @ lop
    # empty input raises BadShapeError (linop)
    lop = SsrftNoiseLinOp(hw, 0, register=False)
    with pytest.raises(BadShapeError):
        lop @ torch.tensor(0)
    with pytest.raises(BadShapeError):
        torch.tensor(0) @ lop
    with pytest.raises(BadShapeError):
        lop @ torch.zeros(0)
    with pytest.raises(BadShapeError):
        torch.zeros(0) @ lop
    # too large or too small out_dims raises error (transform)
    with pytest.raises(ValueError):
        SSRFT.ssrft(torch.ones(10), out_dims=-1)
    with pytest.raises(ValueError):
        SSRFT.ssrft(torch.ones(10), out_dims=0)
    with pytest.raises(ValueError):
        SSRFT.ssrft(torch.ones(10), out_dims=11)
    with pytest.raises(ValueError):
        SSRFT.issrft(torch.ones(10), out_dims=9)
    # invalid shapes raise error (linop)
    with pytest.raises(BadShapeError):
        _ = SsrftNoiseLinOp((0, 0), 0, register=False)
    with pytest.raises(BadShapeError):
        _ = SsrftNoiseLinOp((5, 10), 0, register=False)  # fat
    lop = SsrftNoiseLinOp(hw, 0)
    with pytest.raises(BadShapeError):
        lop @ torch.ones(4)
    with pytest.raises(BadShapeError):
        torch.ones(4) @ lop
    # get_block errors for invalid index
    lop = SsrftNoiseLinOp((5, 3), 0, blocksize=2, register=False)
    with pytest.raises(ValueError):
        lop.get_block(-1, torch.float32, "cpu")
    with pytest.raises(ValueError):
        lop.get_block(2, torch.float32, "cpu")
    # get_block and matmul provide right dtype and device
    lop = SsrftNoiseLinOp((5, 3), 0, register=False)
    for device in torch_devices:
        for dtype in dtypes_tols.keys():
            v = lop.get_block(1, dtype, device)
            assert v.dtype == dtype, "Invalid get_block dtype!"
            assert v.device.type == device, "Invalid get_block device!"
            # output is of same dtype and device as input
            w = lop @ torch.ones(3, dtype=dtype, device=device)
            assert w.dtype == dtype, "Mismatching output dtype!"
            assert w.device.type == device, "Mismatching output device!"
            w = torch.ones(5, dtype=dtype, device=device) @ lop
            assert w.dtype == dtype, "Mismatching output dtype (adj)!"
            assert w.device.type == device, "Mismatching output device (adj)!"
    # deterministic behaviour and seed consistency
    hw = (20, 20)
    for seed in rng_seeds:
        for dtype, tol in dtypes_tols.items():
            for device in torch_devices:
                Ileft = torch.eye(hw[1], dtype=dtype, device=device)
                Iright = torch.eye(hw[0], dtype=dtype, device=device)
                lop1 = SsrftNoiseLinOp(hw, seed, norm="ortho", register=False)
                lop2 = SsrftNoiseLinOp(hw, seed, norm="ortho", register=False)
                lop3 = SsrftNoiseLinOp(
                    hw, seed, norm="ortho", register=False, blocksize=max(hw)
                )
                lop4 = SsrftNoiseLinOp(
                    hw, seed + 1, norm="ortho", register=False
                )
                mat1a = linop_to_matrix(lop1, dtype, device, adjoint=False)
                mat1b = linop_to_matrix(lop1, dtype, device, adjoint=False)
                mat1c = linop_to_matrix(lop1, dtype, device, adjoint=True)
                mat1d = lop3.get_block(0, dtype, device)
                mat1e = lop1.to_matrix(dtype, device)
                mat1f = lop1 @ Iright
                mat1g = Ileft @ lop1
                mat1h = lop3 @ Iright
                mat1i = Ileft @ lop3
                mat2 = linop_to_matrix(lop2, dtype, device, adjoint=False)
                mat4 = linop_to_matrix(lop4, dtype, device, adjoint=False)
                assert torch.allclose(
                    mat1a, mat1b, atol=tol
                ), f"Nondeterministic linop? {lop1}"
                assert torch.allclose(
                    mat1a, mat1c, atol=tol
                ), f"Different fwd and adjoint? {lop1}"
                assert torch.allclose(
                    mat1a, mat1d, atol=tol
                ), f"Inconsistent for different blocksize (get_block)? {lop3}"
                assert torch.allclose(
                    mat1a, mat1e, atol=tol
                ), f"Inconsistent for different blocksize (to_matrix)? {lop3}"
                assert torch.allclose(
                    mat1a, mat1f, atol=tol
                ), f"Wrong matmul? {lop1}"
                assert torch.allclose(
                    mat1a, mat1g, atol=tol
                ), f"Wrong adjoint matmul? {lop1}"
                assert torch.allclose(
                    mat1a, mat1h, atol=tol
                ), f"Inconsistent matmul for different blocksize? {lop1}"
                assert torch.allclose(
                    mat1a, mat1i, atol=tol
                ), f"Inconsistent adj matmul for different blocksize? {lop1}"
                assert torch.allclose(
                    mat1a, mat2, atol=tol
                ), f"Same seed, differentl linop? {lop1}"
                #
                for col in mat1a.H:
                    cosim = abs(col @ mat4) / (col.norm() ** 2)
                    assert (
                        cosim < 0.75
                    ).all(), "Different seeds, similar vectors? {lop1}"
            # seed consistency across devices (if CUDA is available)
            if torch.cuda.is_available():
                lop = SsrftNoiseLinOp(hw, seed, norm="ortho", register=False)
                mat1 = linop_to_matrix(lop, dtype, "cpu", adjoint=False)
                mat2 = linop_to_matrix(lop, dtype, "cuda", adjoint=False)
                mat3 = linop_to_matrix(lop, dtype, "cpu", adjoint=True)
                mat4 = linop_to_matrix(lop, dtype, "cuda", adjoint=True)
                assert torch.allclose(
                    mat1, mat2.cpu(), atol=tol
                ), "Mismatching linop across devices!"
                assert torch.allclose(
                    mat1, mat3.cpu(), atol=tol
                ), "Mismatching linop across devices!"
                assert torch.allclose(
                    mat1, mat4.cpu(), atol=tol
                ), "Mismatching linop across devices!"
            else:
                warnings.warn(
                    "Warning! cross-device tests didn't run "
                    "because CUDA is not available in this device",
                    stacklevel=2,
                )


def test_ssrft_correctness(
    rng_seeds, torch_devices, dtypes_tols, ssrft_hw_and_autocorr_tolerances
):
    """Test case for correctness of SSRFT functionality.

    For the SSRFT transform and/or linop (wherever applicable), tests:
    * Columns/rows behave like iid noise (delta autocorrelation)
    * Matmul and rmatmul with linop (fwd and adj) is same as with matrix
    * Transposed linop is correct (fwd and adj)

    And also:

    * SSRFT linop has orthonormal columns
    * ``issrft(ssrft(x))==x`` and ``ssrft(issrft(x)) == x`` for full dimension
    """
    hw, delta_at_least, nondelta_at_most = ssrft_hw_and_autocorr_tolerances
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                lop = SsrftNoiseLinOp(
                    hw, seed, blocksize=7, norm="ortho", register=False
                )
                mat = linop_to_matrix(lop, dtype, device, adjoint=False)
                tomat = lop.to_matrix(dtype, device)
                # to_matrix is correct
                assert (mat == tomat).all(), "Wrong SSRFT to_matrix?"
                # Columns/rows behave like iid noise (delta autocorr)
                for x in mat:  # x is a row
                    try:
                        autocorrelation_test_helper(
                            x, delta_at_least, nondelta_at_most
                        )
                    except AssertionError as ae:
                        raise AssertionError(
                            "SSRFT autocorr error (row)"
                        ) from ae
                for x in mat.H:  # x is a column
                    try:
                        autocorrelation_test_helper(
                            x, delta_at_least, nondelta_at_most
                        )
                    except AssertionError as ae:
                        raise AssertionError(
                            "SSRFT autocorr error (col)"
                        ) from ae
                # matmul and rmatmul with linop is same as with matrix
                v1 = torch.randn(hw[0], dtype=dtype, device=device)
                v2 = torch.randn(hw[1], dtype=dtype, device=device)
                assert torch.allclose(
                    v1 @ lop, v1 @ mat, atol=tol
                ), "Mismatching adjoint vecmul between lop and mat?"
                assert torch.allclose(
                    lop @ v2, mat @ v2, atol=tol
                ), "Wrong vecmul between lop and mat?"
                # transposed linop is correct
                lopT = TransposedLinOp(lop)
                matTa = linop_to_matrix(lopT, dtype, device, adjoint=False)
                matTb = linop_to_matrix(lopT, dtype, device, adjoint=True)
                assert torch.allclose(
                    mat.H, matTa, atol=tol
                ), "Wrong iid transposition?"
                assert torch.allclose(
                    mat.H, matTb, atol=tol
                ), "Wrong iid transposition? (adjoint)"
                # Orthonormal columns
                assert hw[0] != hw[1], "Tall linop required for this test!"
                assert torch.allclose(
                    mat.H @ mat,
                    torch.eye(hw[1], dtype=mat.dtype, device=mat.device),
                    atol=tol,
                ), "SSRFT columns not orthonormal?"
                # issrft and ssrft invert each other
                w1 = SSRFT.issrft(
                    SSRFT.ssrft(v1, len(v1), seed=seed, norm="ortho"),
                    len(v1),
                    seed=seed,
                    norm="ortho",
                )
                w2 = SSRFT.ssrft(
                    SSRFT.issrft(v2, len(v2), seed=seed, norm="ortho"),
                    len(v2),
                    seed=seed,
                    norm="ortho",
                )
                assert torch.allclose(v1, w1, atol=tol), "issrft(ssrft) != I?"
                assert torch.allclose(v2, w2, atol=tol), "ssrft(issrft) != I?"
