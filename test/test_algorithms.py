#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`algorithms`."""


import pytest
import torch
import numpy as np
from collections import defaultdict


from skerch.utils import COMPLEX_DTYPES, BadShapeError
from skerch.synthmat import RandomLordMatrix
from skerch.algorithms import SketchedAlgorithmDispatcher, TriangularLinOp
from skerch.algorithms import ssvd, seigh, diagpp, xdiag
from skerch.measurements import GaussianNoiseLinOp
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
def diag_recovery_shapes(request):
    """Fixture to test Diag++ and XDiag.

    Tuples in the form ``(dims, rank, defl, gh_extra, diagpp_tol, xdiag_tol)``
    where the tolerances are in the form ``(full, top_only)``.
    """
    result = [
        # low-rank matrix with good deflation: good d++, decent XD top
        (200, 5, 50, 0, (1e-10, 1e-10), (0.5, 0.1)),
        # now less deflation but we add G-H: while dtop is equally good in
        # both, the few GH measurements seem to hurt much more in d++ than XD
        (200, 10, 9, 9, (2, 0.2), (0.5, 0.3)),
        # now we add more GH measurements to D++, and then it surpasses XD
        (200, 10, 9, 190, (0.25, 0.2), (0.5, 0.3)),
        # just doing a lot of GH measurements also works for D++
        (10, 10, 0, 2_000, (0.05, None), (None, None)),
    ]
    if request.config.getoption("--quick"):
        result = result[:1]
    return result


# @pytest.fixture
# def triang_configs(request):
#     """Configurations for the triangular linear operator.

#     The plan is to grab a few square random matrices and run this thing on them
#     with/without FFT,

#     """
#     result = [1, 3, 10, 30, 100]
#     if request.config.getoption("--quick"):
#         result = result[:4]
#     return result


@pytest.fixture
def lowrank_noise_types():
    """Collection of tuples ``(noise_type, is_complex_only)``"""
    result = [
        ("rademacher", False),
        ("gaussian", False),
        ("ssrft", False),
        ("phase", True),
    ]
    return result


@pytest.fixture
def diag_noise_types():
    """Collection of tuples ``(noise_type, is_complex_only)``"""
    result = [
        ("rademacher", False),
        ("bloated_1.0", False),
        ("phase", True),
    ]
    return result


@pytest.fixture
def triang_iter_stairs():
    """Collection of input, output pairs to test ``_iter_stairs``.

    * Input: pair in the form ``(dims, stair_width)``
    * Output: corresponding expected output in the form ``((0, 3), (3, 6))``
    """
    result = [
        ((5, 1), ((0, 1), (1, 2), (2, 3), (3, 4))),
        ((5, 2), ((0, 2), (2, 4))),
        ((9, 3), ((0, 3), (3, 6))),
        ((10, 3), ((0, 3), (3, 6), (6, 9))),
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


class BloatedGaussianNoiseLinOp(GaussianNoiseLinOp):
    """Gaussian noise plus a signed constant.

    For a given gaussian noise entry ``x``, and a constant ``c``, this
    linop contains the entry ``x+c`` if ``x`` was positive, and ``x-c``
    otherwise.

    :param shift: The shifting constant.

    Used in diag test case to test the ability to incorporate custom noise
    sources via custom dispatchers.
    """

    REGISTER = defaultdict(list)

    def __init__(
        self,
        shape,
        seed,
        dtype,
        by_row=False,
        register=True,
        mean=0.0,
        std=1.0,
        shift=1.0,
    ):
        """Initializer. See class docstring."""
        super().__init__(shape, seed, dtype, by_row, register, mean, std)
        self.shift = shift

    def get_vector(self, idx, device):
        """Samples a vector with standard Gaussian i.i.d. noise.

        See base class definition for details.
        """
        result = super().get_vector(idx, device)
        try:
            result = result + result.sign() * self.shift
        except:
            result.real += result.real.sign() * self.shift
            result.imag += result.imag.sign() * self.shift
        return result


class MyDispatcher(SketchedAlgorithmDispatcher):
    """Used here to test the ability to use custom dispatchers."""

    @staticmethod
    def mop(noise_type, hw, seed, dtype, register=False):
        """ """
        if "bloated" in noise_type:
            shift = float(noise_type.split("_")[-1])
            mop = BloatedGaussianNoiseLinOp(
                hw, seed, dtype, by_row=False, register=register, shift=shift
            )
        else:
            mop = SketchedAlgorithmDispatcher.mop(
                noise_type, hw, seed, dtype, register
            )
        return mop


def relerr(ori, rec):
    """Relative error in the form ``(frob(ori - rec) / frob(ori))**2``."""
    result = (ori - rec).norm() / ori.norm()
    return result**2


# ##############################################################################
# # SSVD
# ##############################################################################
def test_ssvd_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    ssvd_recovery_shapes,
    lowrank_noise_types,
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
                    for noise_type, complex_only in lowrank_noise_types:
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
    lowrank_noise_types,
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
                    for noise_type, complex_only in lowrank_noise_types:
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
def test_diagpp_xdiag_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    diag_recovery_shapes,
    diag_noise_types,
    max_mp_workers,
):
    """Correctness test case for XDIAG/DIAGPP:

    Runs diagonal recoveries on all devices/dtypes/noisemats, on a few
    different settings for rank and measurements, testing that:

    * Retrieved Q matrices are orthogonal
    * Retrieved diagonals are close enough (either deflation or final ones)
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for (
                    dims,
                    rank,
                    defl,
                    gh_extra,
                    (diagpp_full_tol, diagpp_top_tol),
                    (xdiag_full_tol, xdiag_top_tol),
                ) in diag_recovery_shapes:
                    hw = (dims, dims)
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
                    lop = BasicMatrixLinOp(mat)
                    D = mat.diag()
                    I = torch.eye(defl, dtype=dtype, device=device)
                    #
                    for noise_type, complex_only in diag_noise_types:
                        if dtype not in COMPLEX_DTYPES and complex_only:
                            # this noise type does not support reals,
                            # skip this iteration
                            continue
                        # run diagpp
                        diag1, (dtop1, ddefl1, Q1, R1) = diagpp(
                            lop,
                            device,
                            dtype,
                            defl,
                            gh_extra,
                            seed,
                            noise_type,
                            max_mp_workers,
                            dispatcher=MyDispatcher,
                        )
                        # run XDiag
                        if (
                            xdiag_full_tol is not None
                            or xdiag_top_tol is not None
                        ):
                            diag2, (dtop2, ddefl2, Q2, R2) = xdiag(
                                lop,
                                device,
                                dtype,
                                defl,
                                seed,
                                noise_type,
                                max_mp_workers,
                                dispatcher=MyDispatcher,
                            )
                        else:
                            Q2 = None
                        # test Qs are orthogonal
                        if Q1 is not None:
                            QhQ1 = Q1.H @ Q1
                            assert torch.allclose(
                                QhQ1, I, atol=tol
                            ), "Diag++ Q not orthogonal?"
                        if Q2 is not None:
                            QhQ2 = Q2.H @ Q2
                            assert torch.allclose(
                                QhQ2, I, atol=tol
                            ), "XDiag Q not orthogonal?"
                        # test recoveries are correct
                        if diagpp_full_tol is not None:
                            assert (
                                relerr(D, diag1) < diagpp_full_tol
                            ), "Bad full Diag++?"
                        if diagpp_top_tol is not None:
                            assert (
                                relerr(D, dtop1) < diagpp_top_tol
                            ), "Bad top Diag++?"
                        if xdiag_full_tol is not None:
                            assert (
                                relerr(D, diag2) < xdiag_full_tol
                            ), "Bad full XDiag?"
                        if xdiag_top_tol is not None:
                            assert (
                                relerr(D, dtop2) < xdiag_top_tol
                            ), "Bad top XDiag?"


# ##############################################################################
# # TRIANGULAR
# ##############################################################################
def test_triang_formal(
    rng_seeds, torch_devices, dtypes_tols, triang_iter_stairs
):
    """Formal test case for ``TriangularLinOp``.

    * Only square, nonempty linops supported
    * Stair width between 1 and linop dims
    * Non-unitnorm noise raises warning
    * Nonpositive GH measurements raises warning
    * repr method
    * iter_stairs
    * Noise dispatcher and seed consistency
    """
    with pytest.raises(BadShapeError):
        TriangularLinOp(torch.zeros(5, 6))
    with pytest.raises(ValueError):
        TriangularLinOp(torch.zeros(5, 5), stair_width=0)
    with pytest.raises(ValueError):
        TriangularLinOp(torch.zeros(5, 5), stair_width=6)
    with pytest.warns(RuntimeWarning):
        TriangularLinOp(torch.zeros(5, 5), num_gh_meas=10, noise_type="ssrft")
    with pytest.warns(RuntimeWarning):
        TriangularLinOp(torch.zeros(5, 5), num_gh_meas=-1)
    #
    s = "<TriangularLinOp[tensor([[0.]])](lower, with main diag)>"
    assert s == str(TriangularLinOp(torch.zeros(1, 1))), "Wrong repr!"
    #
    for (dims, stair_width), stairs1 in triang_iter_stairs:
        stairs2 = tuple(TriangularLinOp._iter_stairs(dims, stair_width))
        assert stairs1 == stairs2, f"Wrong iter_stairs for {dims, stair_width}"
    #
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                mat = torch.ones(10, 10)
                diag, tril, triu = mat.diag(), mat.tril(), mat.triu()
                lop = TriangularLinOp(
                    mat,
                    stair_width=3,
                    lower=True,
                    with_main_diagonal=True,
                    use_fft=True,
                    num_gh_meas=10000,
                )
                v = torch.ones(10)
                w1 = lop @ v
                w2 = tril @ v
                breakpoint()

                """
                Triang correctness:
                ma
                * also test transpose
                * mp parallelization compatible

                """
