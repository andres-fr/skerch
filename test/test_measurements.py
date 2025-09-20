#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for noisy measurements.

* perform_measurements helper function
* IID (Gaussian, Rademacher, Phase)
* SSRFT (transform and linop)
"""


import time
import warnings
import pytest
import torch
from functools import partial

from skerch.linops import linop_to_matrix, TransposedLinOp, CompositeLinOp

from skerch.measurements import (
    lop_measurement,
    perform_measurements,
    RademacherNoiseLinOp,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
    SSRFT,
    SsrftNoiseLinOp,
)

from skerch.utils import (
    gaussian_noise,
    BadShapeError,
    BadSeedError,
    COMPLEX_DTYPES,
)

from . import (
    rng_seeds,
    torch_devices,
    autocorrelation_test_helper,
    max_mp_workers,
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

    def __matmul__(self, x):
        """ """
        if delay > 0:
            time.sleep(delay)
        return self.matrix @ x

    def __rmatmul__(self, x):
        """ """
        if delay > 0:
            time.sleep(delay)
        return x @ self.matrix


# ##############################################################################
# # PERFORM MEASUREMENTS
# ##############################################################################
def test_perform_measurements_formal():
    """Formal test case for perform_measurements.

    * No parallelization raises warning
    * Unknown parallel mode raises error
    * As-compact-matrix returns idxs and correct shape/device/dtype, and dict
      with also correct features otherwise
    """
    m1, m2 = torch.ones((5, 5)), torch.ones((5, 4))
    idxs = list(range(4))

    def meas_fn(idx, adjoint=False):
        return idx, m2.conj() @ m1 if adjoint else m1 @ m2[:, idx]

    # no parallel raises warning
    with pytest.warns(RuntimeWarning):
        perform_measurements(
            meas_fn,
            idxs,
            adjoint=False,
            parallel_mode=None,
            compact=False,
        )
    # unknown parallel raises error
    with pytest.raises(ValueError):
        perform_measurements(
            meas_fn,
            idxs,
            adjoint=False,
            parallel_mode="...",
            compact=False,
        )
    # correct behavior of as_compact_matrix
    meas_dict = perform_measurements(
        meas_fn,
        idxs,
        adjoint=False,
        parallel_mode=None,
        compact=False,
    )
    meas_idxs, meas_mat = perform_measurements(
        meas_fn,
        idxs,
        adjoint=False,
        parallel_mode=None,
        compact=True,
    )
    #
    assert meas_idxs == list(range(m2.shape[1])), "Incorrect meas_idxs?"
    assert isinstance(meas_mat, torch.Tensor), "meas_mat not a tensor?"
    assert meas_mat.shape == (
        m1.shape[0],
        m2.shape[1],
    ), "Mismatching shape in compact matrix?"
    assert meas_mat.dtype == m1.dtype, "Mismatching dtype: measmat/m1"
    assert meas_mat.dtype == m2.dtype, "Mismatching dtype: measmat/m2"
    assert meas_mat.device == m1.device, "Mismatching device: measmat/m1"
    assert meas_mat.device == m2.device, "Mismatching device: measmat/m2"
    #
    assert isinstance(meas_dict, dict), "meas_dict not a dict?"
    assert sorted(meas_dict) == sorted(idxs), "Wrong idxs in meas_dict?"
    for v in meas_dict.values():
        assert v.dtype == m1.dtype, "Mismatching dtype: measdict/m1"
        assert v.dtype == m2.dtype, "Mismatching dtype: measdict/m2"
        assert v.device == m1.device, "Mismatching device: measdict/m1"
        assert v.device == m2.device, "Mismatching device: measdict/m2"


def test_perform_measurements_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    iid_noise_linop_types,
    parallel_modes,
    max_mp_workers,
):
    """Test case for correctness of perform_measurements:

    For each seed/dtype/device, reates a basic random lop/matrix, and all
    supported measurement lops/matrices.
    Then, compares the mat@mat measurements with the output of
    ``perform_measurements`` on all supported parallelism modes, testing:
    * ``perform_measurements`` yields same output as direct matmul
    * Output does not depend on the type of parallelization
    """
    hw = (10, 10)
    (meas_hw,) = (10, 5)
    #
    for seed in rng_seeds:
        for dtype, tol in dtypes_tols.items():
            for device in torch_devices:
                # create matrix to measure from, and make basic lop
                mat = gaussian_noise(hw, seed=seed, dtype=dtype, device=device)
                lop = BasicMatrixLinOp(mat)
                # create measlops and convert them to matrices
                meas_lops = [
                    RademacherNoiseLinOp(
                        meas_hw,
                        seed,
                        dtype,
                        by_row=False,
                        register=False,
                        blocksize=max(meas_hw),
                    ),
                    GaussianNoiseLinOp(
                        meas_hw,
                        seed,
                        dtype,
                        by_row=False,
                        register=False,
                        blocksize=max(meas_hw),
                    ),
                    SsrftNoiseLinOp(meas_hw, seed, norm="ortho"),
                ]
                if dtype in COMPLEX_DTYPES:
                    meas_lops.append(
                        PhaseNoiseLinOp(
                            meas_hw,
                            seed,
                            dtype,
                            by_row=False,
                            register=False,
                            conj=False,
                        )
                    )
                meas_mats = [
                    linop_to_matrix(l, dtype, device, adjoint=False)
                    for l in meas_lops
                ]
                # begin tests
                for mop, mm in zip(meas_lops, meas_mats):
                    # direct computation of measurements
                    y1 = mat @ mm
                    y2 = mm.H @ mat
                    y3 = mm.H @ y1
                    for parall in parallel_modes:
                        # mp only works on CPU
                        if parall == "mp" and device != "cpu":
                            continue
                        # indirect computation of measurements:
                        meas_fn = partial(
                            lop_measurement,
                            lop=lop,
                            meas_lop=mop,
                            device=device,
                            dtype=dtype,
                        )
                        _, z1 = perform_measurements(
                            meas_fn,
                            range(mop.shape[1]),
                            adjoint=False,
                            parallel_mode=parall,
                            compact=True,
                            max_mp_workers=max_mp_workers,
                        )
                        _, z2 = perform_measurements(
                            meas_fn,
                            range(mop.shape[1]),
                            adjoint=True,
                            parallel_mode=parall,
                            compact=True,
                            max_mp_workers=max_mp_workers,
                        )
                        # test that perform_measurements yields same as direct
                        assert torch.allclose(
                            y1, z1, atol=tol
                        ), "Wrong perform_measurements (fwd)?"
                        assert torch.allclose(
                            y2, z2, atol=tol
                        ), "Wrong perform_measurements (adj)?"
                        # added test for inner measurements:
                        # (mop.H @ lop) @ mop
                        mopTlop = CompositeLinOp(
                            [("mop.H", TransposedLinOp(mop)), ("lop", lop)]
                        )
                        meas_fn = partial(
                            lop_measurement,
                            lop=mopTlop,
                            meas_lop=mop,
                            device=device,
                            dtype=dtype,
                        )
                        _, z3 = perform_measurements(
                            meas_fn,
                            range(mop.shape[1]),
                            adjoint=False,
                            parallel_mode=parall,
                            compact=True,
                            max_mp_workers=max_mp_workers,
                        )
                        assert torch.allclose(
                            y3, z3, atol=tol
                        ), "Wrong inner perform_measurements (fwd)?"
                        # mop.H @ (lop @ mop)
                        lopmop = CompositeLinOp([("lop", lop), ("mop", mop)])
                        meas_fn = partial(
                            lop_measurement,
                            lop=lopmop,
                            meas_lop=mop,
                            device=device,
                            dtype=dtype,
                        )
                        _, z4 = perform_measurements(
                            meas_fn,
                            range(mop.shape[1]),
                            adjoint=True,
                            parallel_mode=parall,
                            compact=True,
                            max_mp_workers=max_mp_workers,
                        )
                        assert torch.allclose(
                            y3, z4, atol=tol
                        ), "Wrong inner perform_measurements (adj)?"

            # on CPU, test that all different parallel modes yield same result
            mat1 = gaussian_noise(hw, seed=seed, dtype=dtype, device="cpu")
            mat2 = gaussian_noise(
                meas_hw, seed=seed + 1, dtype=dtype, device="cpu"
            )
            meas_fn = partial(
                lop_measurement,
                lop=mat1,
                meas_lop=mat2,
                device=mat2.device,
                dtype=mat2.dtype,
            )
            _, meas1a = perform_measurements(
                meas_fn,
                range(mat2.shape[1]),
                adjoint=False,
                parallel_mode=None,
                compact=True,
            )

            _, meas1b = perform_measurements(
                meas_fn,
                range(mat2.shape[1]),
                adjoint=False,
                parallel_mode="mp",
                compact=True,
                max_mp_workers=max_mp_workers,
            )
            assert (meas1a == meas1b).all(), "Parallel mode alters output?"
            #
            _, meas2a = perform_measurements(
                meas_fn,
                range(mat2.shape[1]),
                adjoint=True,
                parallel_mode=None,
                compact=True,
            )

            _, meas2b = perform_measurements(
                meas_fn,
                range(mat2.shape[1]),
                adjoint=True,
                parallel_mode="mp",
                compact=True,
                max_mp_workers=max_mp_workers,
            )
            assert (
                meas2a == meas2b
            ).all(), "Parallel mode alters output? (conj)"


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
    lop = RademacherNoiseLinOp(
        hw, 0, torch.float32, by_row=False, register=False
    )
    s = (
        "<RademacherNoiseLinOp(3x3, seed=0, "
        "dtype=torch.float32, by_row=False)>"
    )
    assert str(lop) == s, "Unexpected repr for Rademacher noise linop!"
    lop = GaussianNoiseLinOp(
        hw, 0, torch.float32, by_row=False, register=False
    )
    s = (
        "<GaussianNoiseLinOp(3x3, mean=0.0, std=1.0, seed=0, "
        + "dtype=torch.float32, by_row=False)>"
    )
    assert str(lop) == s, "Unexpected repr for Gaussian noise linop!"
    lop = PhaseNoiseLinOp(hw, 0, torch.complex64, by_row=False, register=False)
    s = (
        "<PhaseNoiseLinOp(3x3, conj=False, seed=0, "
        + "dtype=torch.complex64, by_row=False)>"
    )
    assert str(lop) == s, "Unexpected repr for Phase noise linop!"
    #
    for lop_type, complex_only in iid_noise_linop_types:
        if complex_only:
            dtype1, dtype2 = torch.complex64, torch.complex128
        else:
            dtype1, dtype2 = torch.float32, torch.float64
        lop = lop_type((5, 5), seed=0, dtype=dtype1, by_row=False)
        # register triggers for overlapping seeds regardless of other factors
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=0, dtype=dtype1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, dtype=dtype1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((20, 5), seed=1, dtype=dtype1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, dtype=dtype2, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, dtype=dtype1, by_row=True)
        lop_type.REGISTER.clear()  # clear register for this lop_type
        # invalid index triggers error
        with pytest.raises(ValueError):
            lop.get_vector(idx=-1, device="cpu")
        with pytest.raises(ValueError):
            lop.get_vector(idx=5, device="cpu")
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
                    with pytest.raises(ValueError):
                        _ = lop_type(hw, 0, dtype, register=False)
                    continue
                # seed consistency across devices (if CUDA is available)
                if torch.cuda.is_available():
                    lop = lop_type(
                        hw, seed, dtype, by_row=False, register=False
                    )
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
                        "because CUDA is not available in this device"
                    )
                # deterministic behaviour and seed consistency
                for device in torch_devices:
                    lop1 = lop_type(
                        hw, seed, dtype, by_row=False, register=False
                    )
                    lop2 = lop_type(
                        hw, seed, dtype, by_row=False, register=False
                    )
                    lop3 = lop_type(
                        hw, seed + 5, dtype, by_row=False, register=False
                    )
                    mat1a = linop_to_matrix(
                        lop1, lop1.dtype, device, adjoint=False
                    )
                    mat1b = linop_to_matrix(
                        lop1, lop1.dtype, device, adjoint=False
                    )
                    mat1c = linop_to_matrix(
                        lop1, lop1.dtype, device, adjoint=True
                    )
                    mat2 = linop_to_matrix(
                        lop2, lop1.dtype, device, adjoint=False
                    )
                    mat3 = linop_to_matrix(
                        lop3, lop1.dtype, device, adjoint=False
                    )
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


def test_iid_measurements_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    iid_noise_linop_types,
    iid_hw_and_autocorr_tolerances,
):
    """Test case for correctness of iid measurement linops.

    For each iid linop on all dtypes and devices tests that:
    * Columns/rows behave like iid noise (delta autocorrelation)
    * Fwd and adj matmul lead to same linop
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
                    lop1 = lop_type(
                        hw,
                        seed,
                        dtype,
                        by_row=False,
                        register=False,
                        blocksize=7,  # max(hw),
                    )
                    lop2 = lop_type(
                        hw,
                        seed,
                        dtype,
                        by_row=True,
                        register=False,
                        blocksize=max(hw),
                    )
                    mat1a = lop1 @ Iright
                    mat1b = lop1 @ Iright
                    mat2a = lop2 @ Iright
                    mat2b = Ileft @ lop2
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
                    # fwd and adj matmul lead to same linop
                    assert (
                        mat1a == mat1b
                    ).all(), f"Mismatching fwd/adj {lop_type} (by_col)"
                    assert (
                        mat2a == mat2b
                    ).all(), f"Mismatching fwd/adj {lop_type} (by_row)"
                    # transposed linop is correct
                    lopT = TransposedLinOp(lop1)
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
                lop = PhaseNoiseLinOp(
                    (5, 5), seed, dtype, conj=False, register=False
                )
                lop_conj = PhaseNoiseLinOp(
                    (5, 5), seed, dtype, conj=True, register=False
                )
                mat = linop_to_matrix(lop, lop.dtype, device, adjoint=False)
                mat_conj = linop_to_matrix(
                    lop_conj, lop.dtype, device, adjoint=False
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
    * Non-orthogonal normalization raises NotImplementederror
    * Non-vector or empty imput to ssrft raises BadShapeError
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
    lop = SsrftNoiseLinOp(hw, 0, norm="ortho")
    assert (
        str(lop) == "<SsrftNoiseLinOp(10x10, seed=0)>"
    ), "Unexpected repr for SSRFT noise linop!"
    # non-orthogonal norm raises error (transform)
    with pytest.raises(NotImplementedError):
        SSRFT.ssrft(torch.ones(10), out_dims=10, seed=0, norm="XXXX")
    with pytest.raises(NotImplementedError):
        SSRFT.issrft(torch.ones(10), out_dims=10, seed=0, norm="XXXX")
    # non-orthogonal norm raises error (linop)
    with pytest.raises(NotImplementedError):
        lop = SsrftNoiseLinOp(hw, 0, norm="XXXX")
        lop @ torch.ones(10)
    with pytest.raises(NotImplementedError):
        lop = SsrftNoiseLinOp(hw, 0, norm="XXXX")
        torch.ones(10) @ lop
    # non-vector or empty input raises BadShapeError (transform)
    with pytest.raises(BadShapeError):
        _ = SSRFT.ssrft(torch.zeros(5, 5), 5)
    with pytest.raises(BadShapeError):
        _ = SSRFT.ssrft(torch.tensor(0), 0)
    with pytest.raises(BadShapeError):
        _ = SSRFT.ssrft(torch.zeros(0), 0)
    with pytest.raises(BadShapeError):
        _ = SSRFT.issrft(torch.zeros(5, 5), 5)
    with pytest.raises(BadShapeError):
        _ = SSRFT.issrft(torch.tensor(0), 0)
    with pytest.raises(BadShapeError):
        _ = SSRFT.issrft(torch.zeros(0), 0)
    # empty input raises BadShapeError (linop)
    lop = SsrftNoiseLinOp(hw, 0)
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
        _ = SsrftNoiseLinOp((0, 0), 0)
    with pytest.raises(BadShapeError):
        _ = SsrftNoiseLinOp((5, 10), 0)  # fat
    lop = SsrftNoiseLinOp(hw, 0)
    with pytest.raises(BadShapeError):
        lop @ torch.ones(4)
    with pytest.raises(BadShapeError):
        torch.ones(4) @ lop
    # get_vector errors for invalid index
    lop = SsrftNoiseLinOp((5, 3), 0)
    with pytest.raises(ValueError):
        lop.get_vector(-1, "cpu", torch.float32, by_row=False)
    with pytest.raises(ValueError):
        lop.get_vector(3, "cpu", torch.float32, by_row=False)
    with pytest.raises(ValueError):
        lop.get_vector(-1, "cpu", torch.float32, by_row=True)
    with pytest.raises(ValueError):
        lop.get_vector(5, "cpu", torch.float32, by_row=True)
    # get_vector and matmul provide right dtype and device
    lop = SsrftNoiseLinOp((5, 3), 0)
    for device in torch_devices:
        for dtype in dtypes_tols.keys():
            v = lop.get_vector(0, device, dtype, by_row=True)
            assert v.dtype == dtype, "Invalid get_vector dtype by_row!"
            assert v.device.type == device, "Invalid get_vector device by_row!"
            v = lop.get_vector(0, device, dtype, by_row=False)
            assert v.dtype == dtype, "Invalid get_vector dtype by_col!"
            assert v.device.type == device, "Invalid get_vector device by_col!"
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
                lop1 = SsrftNoiseLinOp(hw, seed, norm="ortho")
                lop2 = SsrftNoiseLinOp(hw, seed, norm="ortho")
                lop3 = SsrftNoiseLinOp(hw, seed + 1, norm="ortho")
                mat1a = linop_to_matrix(lop1, dtype, device, adjoint=False)
                mat1b = linop_to_matrix(lop1, dtype, device, adjoint=False)
                mat1c = linop_to_matrix(lop1, dtype, device, adjoint=True)
                mat2 = linop_to_matrix(lop2, dtype, device, adjoint=False)
                mat3 = linop_to_matrix(lop3, dtype, device, adjoint=False)

                assert torch.allclose(
                    mat1a, mat1b, atol=tol
                ), f"Nondeterministic linop? {lop1}"
                assert torch.allclose(
                    mat1a, mat1c, atol=tol
                ), f"Different fwd and adjoint? {lop1}"
                assert torch.allclose(
                    mat1a, mat2, atol=tol
                ), f"Same seed, differentl linop? {lop1}"
                #
                for col in mat1a.H:
                    cosim = abs(col @ mat3) / (col.norm() ** 2)
                    assert (
                        cosim < 0.75
                    ).all(), "Different seeds, similar vectors? {lop1}"
            # seed consistency across devices (if CUDA is available)
            if torch.cuda.is_available():
                lop = SsrftNoiseLinOp(hw, seed, norm="ortho")
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
                    "because CUDA is not available in this device"
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
                lop = SsrftNoiseLinOp(hw, seed, norm="ortho")
                linop_to_matrix(lop, dtype, device, adjoint=True)
                mat = linop_to_matrix(lop, dtype, device, adjoint=False)
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
