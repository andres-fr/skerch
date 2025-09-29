#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`algorithms`."""


import pytest
import torch
import numpy as np
from collections import defaultdict


from skerch.utils import COMPLEX_DTYPES, BadShapeError, gaussian_noise
from skerch.linops import linop_to_matrix, TransposedLinOp
from skerch.synthmat import RandomLordMatrix
from skerch.algorithms import SketchedAlgorithmDispatcher, TriangularLinOp
from skerch.algorithms import ssvd, seigh, hutchpp, xdiag
from skerch.algorithms import snorm
from skerch.measurements import GaussianNoiseLinOp
from . import rng_seeds, torch_devices
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
def dispatcher_noise_types():
    """ """
    result = ["rademacher", "gaussian", "phase", "ssrft", "bloated_1.0"]
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
        (200, 5, 50, 0, (1e-10, 1e-10), (0.3, 0.1)),
        # now less deflation but we add G-H: while dtop is equally good in
        # both, the few GH measurements seem to hurt much more in d++ than XD
        (200, 10, 9, 50, (0.8, 0.2), (0.35, 0.25)),
        # now we add more GH measurements to D++, and then it surpasses XD
        (200, 10, 9, 190, (0.2, 0.2), (0.35, 0.25)),
        # # just doing a lot of GH measurements also works for D++
        # Commented out: now meas > rank not allowed (just create mat)
        (10, 10, 0, 2_000, (0.05, None), (None, None)),
    ]
    if request.config.getoption("--quick"):
        result = result[:1]
    return result


@pytest.fixture
def trace_recovery_shapes(request):
    """Fixture to test Trace++ and XTrace.

    Tuples in the form ``(dims, rank, defl, gh_extra, diagpp_tol, xdiag_tol)``
    where the tolerances are in the form ``(full, top_only)``.
    """
    result = [
        (1000, 100, 50, 100000, (1e-10, 1e-10), (0.3, 0.1)),
    ]
    return result


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
def diagtrace_noise_types():
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

    * Input: tuple in the form ``(dims, stair_width, reverse)``
    * Output: corresponding expected output in the form ``((0, 3), (3, 6))``
    """
    result = [
        ((5, 1, False), ((0, 1), (1, 2), (2, 3), (3, 4))),
        ((5, 2, False), ((0, 2), (2, 4))),
        ((9, 3, False), ((0, 3), (3, 6))),
        ((10, 3, False), ((0, 3), (3, 6), (6, 9))),
        #
        ((5, 1, True), ((4, 5), (3, 4), (2, 3), (1, 2))),
        ((5, 2, True), ((3, 5), (1, 3))),
        ((9, 3, True), ((6, 9), (3, 6))),
        ((10, 3, True), ((7, 10), (4, 7), (1, 4))),
    ]
    return result


@pytest.fixture
def triang_configs(request):
    """Configurations for the triangular linear operator.

    Tuples in the form ``(dims, step_width, num_gh_meas, frob_err_ratio)``.
    """
    result = [
        (11, 1, 150, 0.15),
        (11, 2, 150, 0.15),
        (11, 10, 150, 0.15),
        (11, 11, 150, 0.15),
        #
        (11, 3, 150, 0.15),
        (11, 4, 150, 0.15),
        (11, 5, 150, 0.15),
        (11, 3, 1000, 0.1),
    ]
    if request.config.getoption("--quick"):
        result = result[:4]
    return result


@pytest.fixture
def norm_noise_types():
    """Collection of tuples ``(noise_type, is_complex_only)``"""
    result = [
        ("rademacher", False),
        ("gaussian", False),
        ("ssrft", False),
        ("bloated_1.0", False),
        ("phase", True),
    ]
    return result


@pytest.fixture
def norm_configs(request):
    """Configurations for the operator norm

    Tuples in the form ``(shape, specdecay, num_meas, op_relerr, frob_relerr)``.
    specdecay: 0.01 slow, 0.1 medium, 0.5 fast
    """

    result = [
        # near-perfect recovery if meas=rank
        ((20, 10), 0.01, 10, 1e-5, 1e-5),
        # also works for non-square
        ((50, 20), 0.1, 15, 0.01, 0.01),
        ((20, 50), 0.1, 15, 0.01, 0.01),
        # if meas shorter but sufficient, recovery still good
        ((200, 200), 0.1, 20, 0.001, 0.01),
    ]
    # if request.config.getoption("--quick"):
    #     result = result[:3]
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
        by_row=False,
        batch=None,
        blocksize=1,
        register=True,
        mean=0.0,
        std=1.0,
        shift=1.0,
    ):
        """Initializer. See class docstring."""
        super().__init__(
            shape, seed, by_row, batch, blocksize, register, mean, std
        )
        self.shift = shift

    def get_block(self, block_idx, input_dtype, input_device):
        """Samples a vector with standard Gaussian i.i.d. noise.

        See base class definition for details.
        """
        result = super().get_block(block_idx, input_dtype, input_device)
        try:
            result = result + result.sign() * self.shift
        except:
            result.real += result.real.sign() * self.shift
            result.imag += result.imag.sign() * self.shift
        return result


class MyDispatcher(SketchedAlgorithmDispatcher):
    """Used here to test the ability to use custom dispatchers."""

    @staticmethod
    def mop(noise_type, hw, seed, dtype, blocksize=1, register=False):
        """ """
        if "bloated" in noise_type:
            shift = float(noise_type.split("_")[-1])
            mop = BloatedGaussianNoiseLinOp(
                hw,
                seed,
                by_row=False,
                batch=None,
                blocksize=blocksize,
                register=register,
                shift=shift,
                mean=0.0,
                std=0.01,
            )
        else:
            mop = SketchedAlgorithmDispatcher.mop(
                noise_type, hw, seed, dtype, blocksize, register
            )
        return mop


def relerr(ori, rec):
    """Relative error in the form ``(frob(ori - rec) / frob(ori))**2``."""
    result = (ori - rec).norm() / ori.norm()
    return result**2


# ##############################################################################
# # DISPATCHER
# ##############################################################################
def test_algo_dispatcher(dispatcher_noise_types):
    """ """
    # unknown recovery raises error
    with pytest.raises(ValueError):
        SketchedAlgorithmDispatcher.recovery("MadeUpRecovery")
    # unknown measurement linop raises error
    with pytest.raises(ValueError):
        SketchedAlgorithmDispatcher.mop(
            "MadeUpMop", (3, 3), 0, torch.float32, 1, False
        )
    # unknown measurement linop triggers warning in unitnorm checker
    with pytest.warns(RuntimeWarning):
        SketchedAlgorithmDispatcher.unitnorm_lop_entries("MadeUpMop")
    # returned mop supports @ and get_blocks
    dims, dtype, device = 5, torch.complex64, "cpu"
    for mop_type in dispatcher_noise_types:
        mop = MyDispatcher.mop(mop_type, (dims, dims), 0, dtype, dims, False)
        I = torch.eye(dims, dtype=dtype, device=device)
        mat1 = mop @ I
        mat2 = I @ mop
        mat3 = list(mop.get_blocks(dtype, device))[0][0]
        assert (mat1 == mat2).all(), "Measurement linop: inconsistent @?"
        assert (
            mat1 == mat2
        ).all(), "Measurement linop: inconsistent get_blocks?"


# ##############################################################################
# # SSVD
# ##############################################################################
def test_ssvd_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    ssvd_recovery_shapes,
    lowrank_noise_types,
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
                            # run SSVD with max blocksize
                            U, S, Vh = ssvd(
                                lop,
                                device,
                                dtype,
                                outermeas,
                                seed + 1,
                                noise_type,
                                recovery_type,
                                meas_blocksize=max(lop.shape),
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
                                    meas_blocksize=dims,
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
def test_hutchpp_xdiag_formal():
    """Formal test case for ``hutchpp`` amd ``xdiag``.

    * Only square, nonempty linops supported
    * defl_dims or x_dims can't be larger than dims.
    * At least 1 measurement must be done, and meas params can't be negative
    * Warning for non-unitnorm noise
    """
    Z, M, H = torch.ones(0, 0), torch.ones(4, 5), torch.ones(5, 5)
    # HUTCHPP
    for diag in (True, False):
        # nonsquare lop
        with pytest.raises(BadShapeError):
            _ = hutchpp(
                M,
                M.device,
                M.dtype,
                defl_dims=1,
                extra_gh_meas=0,
                seed=0,
                return_diag=diag,
            )
        # empty lop
        with pytest.raises(BadShapeError):
            _ = hutchpp(
                Z,
                Z.device,
                Z.dtype,
                defl_dims=1,
                extra_gh_meas=0,
                seed=0,
                return_diag=diag,
            )
        # too many defl
        with pytest.raises(ValueError):
            _ = hutchpp(
                H,
                H.device,
                H.dtype,
                defl_dims=6,
                extra_gh_meas=0,
                seed=0,
                return_diag=diag,
            )
        # zero total measurements
        with pytest.raises(ValueError):
            _ = hutchpp(
                H,
                H.device,
                H.dtype,
                defl_dims=0,
                extra_gh_meas=0,
                seed=0,
                return_diag=diag,
            )
        # negative measurements
        with pytest.raises(ValueError):
            _ = hutchpp(
                H,
                H.device,
                H.dtype,
                defl_dims=-1,
                extra_gh_meas=1,
                seed=0,
                return_diag=diag,
            )
        with pytest.raises(ValueError):
            _ = hutchpp(
                H,
                H.device,
                H.dtype,
                defl_dims=1,
                extra_gh_meas=-1,
                seed=0,
                return_diag=diag,
            )
    # warning for non-unitnorm noise
    with pytest.warns(RuntimeWarning):
        _ = hutchpp(H, H.device, H.dtype, defl_dims=1, noise_type="gaussian")
    # XDIAG
    # nonsquare lop
    with pytest.raises(BadShapeError):
        _ = xdiag(M, M.device, M.dtype, x_dims=1, seed=0)
    # empty lop
    with pytest.raises(BadShapeError):
        _ = xdiag(Z, Z.device, Z.dtype, x_dims=1, seed=0)
    # too many or negative meas
    with pytest.raises(ValueError):
        _ = xdiag(H, H.device, H.dtype, x_dims=6, seed=0)
    with pytest.raises(ValueError):
        _ = xdiag(H, H.device, H.dtype, x_dims=0, seed=0)
    # warning for non-unitnorm noise
    with pytest.warns(RuntimeWarning):
        _ = xdiag(
            H, H.device, H.dtype, x_dims=1, seed=0, noise_type="gaussian"
        )


def test_diagpp_xdiag_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    diag_recovery_shapes,
    diagtrace_noise_types,
):
    """Correctness test case for ``diagpp`` amd ``xdiag``.

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
                    tr = D.sum()
                    I = torch.eye(defl, dtype=dtype, device=device)
                    #
                    for noise_type, complex_only in diagtrace_noise_types:
                        if dtype not in COMPLEX_DTYPES and complex_only:
                            # this noise type does not support reals,
                            # skip this iteration
                            continue
                        # run hutchpp
                        hutch = hutchpp(
                            lop,
                            device,
                            dtype,
                            defl,
                            gh_extra,
                            seed,
                            noise_type,
                            meas_blocksize=dims,
                            dispatcher=MyDispatcher,
                            return_diag=True,
                        )
                        Q1, R1 = hutch["QR"]
                        diag1, dtop1, dgh1 = hutch["diag"]
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
                                meas_blocksize=dims,
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
    dims, stair_width, gh_meas = 10, 3, 10
    # only square linops supported
    with pytest.raises(BadShapeError):
        TriangularLinOp(torch.zeros(5, 6))
    # only nonempty linops supported
    with pytest.raises(BadShapeError):
        TriangularLinOp(torch.zeros(0, 0))
    # Stair width between 1 and dims
    with pytest.raises(ValueError):
        TriangularLinOp(torch.zeros(5, 5), stair_width=0)
    with pytest.raises(ValueError):
        TriangularLinOp(torch.zeros(5, 5), stair_width=6)
    # non-unitnorm noise raises warning
    with pytest.warns(RuntimeWarning):
        TriangularLinOp(torch.zeros(5, 5), num_gh_meas=10, noise_type="ssrft")
    # nonpositive GH measurements raise warning
    with pytest.warns(RuntimeWarning):
        TriangularLinOp(torch.zeros(5, 5), num_gh_meas=-1)
    # repr test
    s = "<TriangularLinOp[tensor([[0.]])](lower, with main diag)>"
    assert s == str(TriangularLinOp(torch.zeros(1, 1))), "Wrong repr!"
    # iter stairs test
    for (dims, stair_width, rev), stairs1 in triang_iter_stairs:
        stairs2 = tuple(TriangularLinOp._iter_stairs(dims, stair_width, rev))
        assert stairs1 == stairs2, f"Wrong iter_stairs for {dims, stair_width}"
    #
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                mat = gaussian_noise((dims, dims), 0, 1, seed, dtype, device)
                for lower in (True, False):
                    for diag in (True, False):
                        lop1a = TriangularLinOp(
                            mat,
                            stair_width=stair_width,
                            lower=lower,
                            with_main_diagonal=diag,
                            use_fft=False,
                            num_gh_meas=gh_meas,
                            seed=seed,
                        )
                        lop1b = TriangularLinOp(
                            mat,
                            stair_width=stair_width,
                            lower=lower,
                            with_main_diagonal=diag,
                            use_fft=False,
                            num_gh_meas=gh_meas,
                            seed=seed,
                        )
                        lop2 = TriangularLinOp(
                            mat,
                            stair_width=stair_width,
                            lower=lower,
                            with_main_diagonal=diag,
                            use_fft=False,
                            num_gh_meas=gh_meas,
                            seed=seed + 1,
                        )
                        mat1a = linop_to_matrix(
                            lop1a, dtype, device, adjoint=False
                        )
                        mat1a_ = linop_to_matrix(
                            lop1a, dtype, device, adjoint=False
                        )
                        # **commented on purpose: by design, triang linop
                        # fwd and adj are not same**
                        # mat1j = linop_to_matrix(
                        #     lop1a, dtype, device, adjoint=True
                        # )
                        mat1b = linop_to_matrix(
                            lop1b, dtype, device, adjoint=False
                        )
                        mat2 = linop_to_matrix(
                            lop2, dtype, device, adjoint=False
                        )
                        # running same linop twice yields same output
                        assert torch.allclose(
                            mat1a, mat1a_, atol=tol
                        ), "Nondeterministic linop?"
                        # **commented on purpose: by design, triang linop
                        # fwd and adj are not same**
                        # # adjoint to_matrix is same as forward
                        # assert torch.allclose(
                        #     mat1a, mat1j, atol=tol
                        # ), "Inconsistent fwd/adj linop?"
                        # same-seed linop yields same output
                        assert torch.allclose(
                            mat1a, mat1b, atol=tol
                        ), "Same seed, different result?"
                        # different-seed linop yields different output
                        assert (
                            torch.isclose(mat1a, mat2, atol=tol).sum()
                            / mat2.numel()
                            < 0.9
                        ), "Different seed, similar results?"


def test_triang_correctness(
    rng_seeds, torch_devices, dtypes_tols, triang_configs
):
    """Test case for correctness of ``TriangularLinOp``.

    For a variety of random matrices, checks that tri matvecs are close enough:
    * As lower and upper triangle
    * with and without main diag
    * Via forward and adjoint matmul
    * For a variety of stair widths
    * Transposed linop
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for dims, width, gh_meas, errtol in triang_configs:
                    mat = gaussian_noise(
                        (dims, dims), 0, 1, seed, dtype, device
                    )
                    # lower/upper matrices, with/without diag
                    mat1 = mat.tril(-1)
                    mat1d = mat.tril(0)
                    mat2 = mat.triu(1)
                    mat2d = mat.triu(0)
                    # lower/upper linops, with/without diag
                    lop1 = TriangularLinOp(
                        mat,
                        stair_width=3,
                        lower=True,
                        with_main_diagonal=False,
                        use_fft=False,
                        num_gh_meas=gh_meas,
                    )
                    lop1d = TriangularLinOp(
                        mat,
                        stair_width=3,
                        lower=True,
                        with_main_diagonal=True,
                        use_fft=False,
                        num_gh_meas=gh_meas,
                    )
                    lop2 = TriangularLinOp(
                        mat,
                        stair_width=3,
                        lower=False,
                        with_main_diagonal=False,
                        use_fft=False,
                        num_gh_meas=gh_meas,
                    )
                    lop2d = TriangularLinOp(
                        mat,
                        stair_width=3,
                        lower=False,
                        with_main_diagonal=True,
                        use_fft=False,
                        num_gh_meas=gh_meas,
                    )
                    # test vector
                    v = gaussian_noise(dims, 0, 1, seed + 1, dtype, device)
                    # forward matmul
                    v1 = lop1 @ v
                    v1d = lop1d @ v
                    v2 = lop2 @ v
                    v2d = lop2d @ v
                    assert (
                        relerr(mat1 @ v, v1) < errtol
                    ), "Wrong lower/noDiag trilop! (fwd)"
                    assert (
                        relerr(mat1d @ v, v1d) < errtol
                    ), "Wrong lower/withDiag trilop! (fwd)"
                    assert (
                        relerr(mat2 @ v, v2) < errtol
                    ), "Wrong upper/noDiag trilop! (fwd)"
                    assert (
                        relerr(mat2d @ v, v2d) < errtol
                    ), "Wrong upper/withDiag trilop! (fwd)"
                    # adjoint matmul
                    v1 = v @ lop1
                    v1d = v @ lop1d
                    v2 = v @ lop2
                    v2d = v @ lop2d
                    assert (
                        relerr(v @ mat1, v1) < errtol
                    ), "Wrong lower/noDiag trilop! (adj)"
                    assert (
                        relerr(v @ mat1d, v1d) < errtol
                    ), "Wrong lower/withDiag trilop! (adj)"
                    assert (
                        relerr(v @ mat2, v2) < errtol
                    ), "Wrong upper/noDiag trilop! (adj)"
                    assert (
                        relerr(v @ mat2d, v2d) < errtol
                    ), "Wrong upper/withDiag trilop! (adj)"
                    # transposed
                    lop1dT = TransposedLinOp(lop1d)
                    lop2dT = TransposedLinOp(lop2d)
                    assert (
                        (lop1dT @ v.conj()).conj() == v1d
                    ).all(), "Wrong triangular transpose? (lower)"
                    assert (
                        (lop2dT @ v.conj()).conj() == v2d
                    ).all(), "Wrong triangular transpose? (upper)"


# ##############################################################################
# # NORMS
# ##############################################################################
def test_norm_formal():
    """Formal test case for ``snorm``.

    * nore measurements than shape raises error
    * unsupported norm type raises error
    """
    mat = torch.empty(5, 5)
    # too little measurements
    with pytest.raises(ValueError):
        _ = snorm(
            mat,
            mat.device,
            mat.dtype,
            num_meas=0,
            seed=0,
            noise_type="gaussian",
            norm_types=("fro", "op"),
        )
    # too many measurements
    with pytest.raises(ValueError):
        _ = snorm(
            mat,
            mat.device,
            mat.dtype,
            num_meas=len(mat) + 1,
            seed=0,
            noise_type="gaussian",
            norm_types=("fro", "op"),
        )
    # unknown norm type
    with pytest.raises(ValueError):
        _ = snorm(
            mat,
            mat.device,
            mat.dtype,
            num_meas=len(mat) + 1,
            seed=0,
            noise_type="gaussian",
            norm_types=("fro", "op", "fernandez"),
        )


def test_norm_correctness(
    rng_seeds, torch_devices, dtypes_tols, norm_configs, norm_noise_types
):
    """Test case for correctness of sketched norms.


    For all seeds, devices, dtypes, noise types and shape/spectrum cases,
    computes ``snorm`` and checks all results are within given tolerances.
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for shape, specdecay, num_meas, operr, froberr in norm_configs:
                    mat, _ = RandomLordMatrix.exp(
                        shape,
                        1,
                        decay=specdecay,
                        diag_ratio=0.0,
                        symmetric=False,
                        psd=False,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    mat *= 123  # to ensure norm is not always 1
                    svals = torch.linalg.svdvals(mat)
                    frob = svals.norm()
                    opnorm = svals.max()
                    lop = BasicMatrixLinOp(mat)
                    for noise_type, complex_only in norm_noise_types:
                        if dtype not in COMPLEX_DTYPES and complex_only:
                            # this noise type does not support reals,
                            # skip this iteration
                            continue
                        # run sketched norms and test they are within relerr
                        snorms, _ = snorm(
                            lop,
                            device,
                            dtype,
                            num_meas,
                            seed,
                            noise_type,
                            adj_meas=None,
                            dispatcher=MyDispatcher,
                            norm_types=("fro", "op"),
                        )
                        assert (
                            abs(opnorm - snorms["op"]) / opnorm <= operr
                        ), "Wrong operator snorm!"
                        assert (
                            abs(frob - snorms["fro"]) / frob <= froberr
                        ), "Wrong Frobenius snorm!"


# ##############################################################################
# # TRACEPP/XTRACE
# ##############################################################################
def test_tracepp_xtrace_correctness(
    rng_seeds,
    torch_devices,
    dtypes_tols,
    trace_recovery_shapes,
    diagtrace_noise_types,
):
    """Correctness test case for ``diagpp`` amd ``xdiag``.

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
                ) in trace_recovery_shapes:
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
                    tr = D.sum()
                    I = torch.eye(defl, dtype=dtype, device=device)
                    #
                    for noise_type, complex_only in diagtrace_noise_types:
                        if dtype not in COMPLEX_DTYPES and complex_only:
                            # this noise type does not support reals,
                            # skip this iteration
                            continue
                        # run hutchpp
                        hutch = hutchpp(
                            lop,
                            device,
                            dtype,
                            defl,
                            gh_extra,
                            seed,
                            noise_type,
                            meas_blocksize=dims
                            + gh_extra,  #######################
                            dispatcher=MyDispatcher,
                            return_diag=True,
                        )
                        Q1, R1 = hutch["QR"]
                        tr1, trtop1, trgh1 = hutch["tr"]

                        print(
                            ">>>",
                            noise_type,
                            torch.dist(tr, tr1).item(),
                            tr.item(),
                            tr1.item(),
                        )
                        breakpoint()
