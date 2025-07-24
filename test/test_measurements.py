#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for all noisy measurements

SSRFT

rademacher

gaussian



TODO:
* finish SSRFT test, then test the perform_measurement function:
  - with all meas linops
  - with mocked parallelization (byvector?), and test it is like inline

LATER TODO:

* Implement 3 recovery methods
  - test correctness and formal
* Implement all sketched algorithms as meas-recovery
* HDF5?

"""


import warnings
import pytest
import torch

from skerch.linops import (
    linop_to_matrix,
    TransposedLinOp,
)

from skerch.measurements import (
    perform_measurement,
    RademacherNoiseLinOp,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
    SSRFT,
    SsrftNoiseLinOp,
)

from skerch.utils import BadShapeError, BadSeedError

from . import rng_seeds, torch_devices, autocorrelation_test_helper


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


# ##############################################################################
# # TESTS
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
    hw = (10, 10)
    lop = RademacherNoiseLinOp(
        hw, 0, torch.float32, by_row=False, register=False
    )
    s = (
        "<RademacherNoiseLinOp(10x10, seed=0, "
        "dtype=torch.float32, by_row=False)>"
    )
    assert str(lop) == s, "Unexpected repr for Rademacher noise linop!"
    lop = GaussianNoiseLinOp(
        hw, 0, torch.float32, by_row=False, register=False
    )
    s = (
        "<GaussianNoiseLinOp(10x10, mean=0.0, std=1.0, seed=0, "
        + "dtype=torch.float32, by_row=False)>"
    )
    assert str(lop) == s, "Unexpected repr for Gaussian noise linop!"
    lop = PhaseNoiseLinOp(hw, 0, torch.complex64, by_row=False, register=False)
    s = (
        "<PhaseNoiseLinOp(10x10, conj=False, seed=0, "
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


def test_iid_measurements_correctness():
    """
    EMPTY TEST:
    Sources of iid noise used in the IID measurement linops are already tested
    in utils, and here we assume formal tests are sufficient.
    Add tests here if that is not the case.
    """
    NotImplemented


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
        _ = SsrftNoiseLinOp((10, 5), 0)
    lop = SsrftNoiseLinOp(hw, 0)
    with pytest.raises(BadShapeError):
        lop @ torch.ones(4)
    with pytest.raises(BadShapeError):
        torch.ones(4) @ lop
    # get_vector errors for invalid index
    lop = SsrftNoiseLinOp((3, 5), 0)
    with pytest.raises(ValueError):
        lop.get_vector(-1, torch.float32, "cpu", by_row=False)
    with pytest.raises(ValueError):
        lop.get_vector(5, torch.float32, "cpu", by_row=False)
    with pytest.raises(ValueError):
        lop.get_vector(-1, torch.float32, "cpu", by_row=True)
    with pytest.raises(ValueError):
        lop.get_vector(3, torch.float32, "cpu", by_row=True)
    # get_vector and matmul provide right dtype and device
    lop = SsrftNoiseLinOp((3, 5), 0)
    for device in torch_devices:
        for dtype in dtypes_tols.keys():
            v = lop.get_vector(0, dtype, device, by_row=True)
            assert v.dtype == dtype, "Invalid get_vector dtype by_row!"
            assert v.device.type == device, "Invalid get_vector device by_row!"
            v = lop.get_vector(0, dtype, device, by_row=False)
            assert v.dtype == dtype, "Invalid get_vector dtype by_col!"
            assert v.device.type == device, "Invalid get_vector device by_col!"
            # output is of same dtype and device as input
            w = lop @ torch.ones(5, dtype=dtype, device=device)
            assert w.dtype == dtype, "Mismatching output dtype!"
            assert w.device.type == device, "Mismatching output device!"
            w = torch.ones(3, dtype=dtype, device=device) @ lop
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


def test_ssrft_correctness():
    """
    Test case for correctness of SSRFT functionality.

    For the SSRFT transform and/or linop (wherever applicable), tests:
    * SSRFT rows/columns autocorrelations are quasi-deltas
    * Forward and adjoint linop vecmul is equivalent to the transform
    * Get vector by row and by col yields same matrix, and this matrix is
      identical to applying the linop from either side
    * The matrix is unitary if square

    TEST TRANSPOSITIONS ALSO IN FORMAL FOR ALL LINOPS!
    """
    # autocorrelation_test_helper(noise1)
    pass

    # TODO:
    # * adapt ssrft to also admit complex (phase noise instead of rademacher)
    # * integrate ssrft into a linop
    # * test it!

    # breakpoint()
    # idx=5; aa = torch.zeros(1000); aa[idx] = 1; bb = dct.dct(aa, norm="ortho"); cc=dct2(aa)
    # plt.clf(); plt.plot(cc); plt.show()


# def test_no_nans(torch_devices, f64_rtol, rng_seeds, square_shapes):
#     """Tests that SSRFT yields no NaNs."""
#     for seed in rng_seeds:
#         for h, w in square_shapes:
#             ssrft = SSRFT((h, w), seed=seed)
#             for device in torch_devices:
#                 for dtype, _rtol in f64_rtol.items():
#                     x = torch.randn(w, dtype=dtype).to(device)
#                     y = ssrft @ x
#                     xx = y @ ssrft
#                     #
#                     assert not x.isnan().any(), f"{ssrft, device, dtype}"
#                     assert not y.isnan().any(), f"{ssrft, device, dtype}"
#                     assert not xx.isnan().any(), f"{ssrft, device, dtype}"


# def test_invertible(torch_devices, f64_rtol, rng_seeds, square_shapes):
#     """Invertibility/orthogonality of quare SSRFT.

#     Tests that, when input and output dimensionality are the same, the SSRFT
#     operator is orthogonal, i.e. we can recover the input exactly via an
#     adjoint operation.

#     Also tests that it works for mat-vec and mat-mat formats.
#     """
#     for seed in rng_seeds:
#         for h, w in square_shapes:
#             ssrft = SSRFT((h, w), seed=seed)
#             for device in torch_devices:
#                 for dtype, rtol in f64_rtol.items():
#                     # matvec
#                     x = torch.randn(w, dtype=dtype).to(device)
#                     y = ssrft @ x
#                     xx = y @ ssrft
#                     #
#                     assert torch.allclose(
#                         x, xx, rtol=rtol
#                     ), f"MATVEC: {ssrft, device, dtype}"
#                     # matmat
#                     x = torch.randn((w, 2), dtype=dtype).to(device)
#                     y = ssrft @ x
#                     xx = (y.T @ ssrft).T
#                     #
#                     assert torch.allclose(
#                         x, xx, rtol=rtol
#                     ), f"MATMAT: {ssrft, device, dtype}"
#                     # matmat-shape tests
#                     assert len(y.shape) == 2
#                     assert len(xx.shape) == 2
#                     assert y.shape[-1] == 2
#                     assert xx.shape[-1] == 2
