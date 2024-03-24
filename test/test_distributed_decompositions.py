#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for distributed sketched matrix decompositions."""


import tempfile

import pytest
import torch

from skerch.a_posteriori import a_posteriori_error
from skerch.decompositions import truncate_core
from skerch.synthmat import SynthMat

from . import rng_seeds, torch_devices  # noqa: F401
from .dseigh import mock_dseigh
from .dssvd import mock_dssvd


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def atol_exp():
    """Absolute error tolerances for exp-decaying test matrices."""
    result = {torch.float64: 1e-7, torch.float32: 1e-3}
    return result


@pytest.fixture
def atol_poly():
    """Absolute error tolerances for poly-decaying test matrices."""
    result = {torch.float64: 3e-3, torch.float32: 3e-3}
    return result


@pytest.fixture
def heights_widths_ranks_outer_inner_posteriori(request):
    """Test cases for different matrices and measurements.

    Entries are ``((h, w), r, (o, i)), q``. For a matrix of shape ``h, w`` and
    rank ``r``, the Skinny SVD will do ``o`` outer measurements and ``i``
    inner measurements. Recommended is that ``i >= 2*o``. The a-posteriori
    number of measurements is ``q``.

    For constrained budgets, the Skinny SVD will naturally yield higher error
    with smaller shapes, hence we test a few medium shapes.
    """
    result = [
        ((1_000, 1_000), 10, (100, 300), 30),
        ((1_000, 1_000), 50, (200, 600), 30),
        # ((1_000, 2_000), 10, (100, 300), 30),  # commentd to save time
        ((1_000, 2_000), 50, (200, 600), 30),
        # ((2_000, 2_000), 20, (100, 300), 30),  # commented to save time
        ((2_000, 2_000), 100, (200, 600), 30),
    ]
    if request.config.getoption("--quick"):
        result = result[1:2]
    return result


@pytest.fixture
def dims_ranks_outer_inner_posteriori(request):
    """Test cases for different (square) matrices and measurements.

    Like ``heights_widths_ranks_outer_inner_posteriori`` but for square shapes
    to test on symmetric matrices.
    """
    result = [
        (1_000, 10, (100, 300), 30),
        (1_000, 25, (200, 600), 30),
        # (2_000, 10, (100, 300), 30),  # commented to save time
        (2_000, 50, (200, 600), 30),
        (2_000, 100, (200, 600), 30),
    ]
    if request.config.getoption("--quick"):
        result = result[1:2]
    return result


@pytest.fixture
def recons_frob_tol():
    """Ratio tolerances for Frobenius residual errors of recovered matrices.

    Threshold for the Frobenius of residual divided by energy of original.
    If 0.05, it means that the residual must be 5% or less of the original.
    """
    result = {torch.float64: 0.05, torch.float32: 0.05}
    return result


@pytest.fixture
def svectors_frob_tol():
    """Ratio tolerances for Frobenius residual errors of recovered eigenbases.

    Threshold for the Frobenius of residual divided by energy of original.
    If 0.05, it means that the residual must be 5% or less of the original.
    """
    result = {torch.float64: 0.05, torch.float32: 0.05}
    return result


@pytest.fixture
def svectors_check_n():
    """How many singular vectors to check."""
    return 5


@pytest.fixture
def high_snr(request):
    """SNR values for Lowrank+noise matrix. The larger, the more noise.

    These values are low-noise, allowing for better recoveries.
    """
    if request.config.getoption("--quick"):
        return [1e-2]
    else:
        return [1e-3, 1e-2]


@pytest.fixture
def steep_exp_decay(request):
    """Steep exponential decay values, which allow for better recoveries.

    The larger, the faster decay.
    """
    if request.config.getoption("--quick"):
        return [0.1]
    else:
        return [0.5, 0.1]


@pytest.fixture
def steep_poly_decay(request):
    """Steep polynomial decay values, which allow for better recoveries.

    The larger, the faster decay.
    """
    if request.config.getoption("--quick"):
        return [2]
    else:
        return [4, 2]


# ##############################################################################
# # SKETCHED SVD
# ##############################################################################
def test_dssvd_asymmetric_exp(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    atol_exp,
    heights_widths_ranks_outer_inner_posteriori,
    steep_exp_decay,
    recons_frob_tol,
    svectors_frob_tol,
    svectors_check_n,
):
    """Test distributed SSVD on asymmetric exponential matrices.

    Sample matrices, and compute their ``torch.svd`` as well as the sketched
    SVD. Test that:
    * Recovered singular values are correct
    * Recovered matrix is correct
    * The first bulk singular vectors are correct
    * A-posteriori error estimation is low
    """
    for seed in rng_seeds:
        for dtype, atol in atol_exp.items():
            for (
                (h, w),
                r,
                (outer, inner),
                post,
            ) in heights_widths_ranks_outer_inner_posteriori:
                for dec in steep_exp_decay:
                    # we create matrix and full SVD on cpu, it is faster
                    mat = SynthMat.exp_decay(
                        shape=(h, w),
                        rank=r,
                        decay=dec,
                        symmetric=False,
                        seed=seed,
                        dtype=dtype,
                        device="cpu",
                    )
                    U, S, Vt = torch.linalg.svd(mat)
                    # then we test decomposition on both devices
                    for device in torch_devices:
                        mat = mat.to(device)
                        U = U.to(device)
                        S = S.to(device)
                        Vt = Vt.to(device)
                        #
                        tmpdir = tempfile.TemporaryDirectory()
                        (ro_Q, core_U, core_S, core_Vt, lo_Qt) = mock_dssvd(
                            tmpdir.name, mat, device, dtype, outer, inner, seed
                        )
                        # check that singular values are correct
                        assert torch.allclose(
                            S[: 2 * r], core_S[: 2 * r], atol=atol
                        ), "Bad recovery of singular values!"
                        # check that recovered matrix is the same
                        core_U, core_S, core_Vt = truncate_core(
                            2 * r, core_U, core_S, core_Vt
                        )
                        mat_recons = (
                            ro_Q @ core_U @ torch.diag(core_S) @ core_Vt @ lo_Qt
                        )
                        recons_err = torch.dist(mat, mat_recons) / mat.norm()
                        assert recons_err <= abs(
                            recons_frob_tol[dtype]
                        ), "Bad SSVD reconstruction!"
                        # check that a few singular vectors are correct: avoid
                        # the ones that have the same singular values, since
                        # they aren't unique. Also compare outer prods, since
                        # multiplying left and right by -1 yields same result.
                        left = ro_Q @ core_U[:, r:]
                        right = core_Vt[r:, :] @ lo_Qt

                        for idx in range(svectors_check_n):
                            recons_err = torch.dist(
                                torch.outer(left[:, idx], right[idx]),
                                torch.outer(U[:, idx + r], Vt[idx + r]),
                            ) / (U[:, idx + r].norm() * Vt[idx + r].norm())
                            assert recons_err <= abs(
                                svectors_frob_tol[dtype]
                            ), "Bad recovery of singular vectors!"
                        # check a-posteriori error. With 30 measurements, we
                        # know that the probability of the actual error being
                        # twice as big is less than 1%, and keeps shrinking
                        # exponentially: a_posteriori_error_bounds(30, 1)
                        # Therefore, a very small estimate almost certainly
                        # guarantees that the approximation is good, and we
                        # have enough rank in our truncated SVD.
                        for adjoint in [True, False]:
                            _, _, err_estimate = a_posteriori_error(
                                mat,
                                mat_recons,
                                post,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                adjoint=adjoint,
                            )[0]
                            assert (
                                err_estimate / mat.numel()
                            ) <= atol, "Bad a-posteriori estimate!"
                        # cleanup HDF5 tmpdir
                        tmpdir.cleanup()


def test_dssvd_symmetric_exp(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    atol_exp,
    dims_ranks_outer_inner_posteriori,
    steep_exp_decay,
    recons_frob_tol,
    svectors_frob_tol,
    svectors_check_n,
):
    """Test distributed SSVD on symmetric exponential matrices.

    Sample matrices, and compute their ``torch.svd`` as well as the sketched
    SVD. Test that:
    * Recovered singular values are correct
    * Recovered matrix is correct and symmetric
    * The first bulk singular vectors are correct
    * A-posteriori error estimation is low
    """
    for seed in rng_seeds:
        for dtype, atol in atol_exp.items():
            for d, r, (outer, inner), post in dims_ranks_outer_inner_posteriori:
                for dec in steep_exp_decay:
                    # we create matrix and full SVD on cpu, it is faster
                    mat = SynthMat.exp_decay(
                        shape=(d, d),
                        rank=r,
                        decay=dec,
                        symmetric=True,
                        seed=seed,
                        dtype=dtype,
                        device="cpu",
                    )
                    U, S, Vt = torch.linalg.svd(mat)

                    # then we test decomposition on both devices
                    for device in torch_devices:
                        mat = mat.to(device)
                        U = U.to(device)
                        S = S.to(device)
                        Vt = Vt.to(device)
                        #
                        tmpdir = tempfile.TemporaryDirectory()
                        (ro_Q, core_U, core_S, core_Vt, lo_Qt) = mock_dssvd(
                            tmpdir.name, mat, device, dtype, outer, inner, seed
                        )
                        # check that singular values are correct
                        assert torch.allclose(
                            S[: 2 * r], core_S[: 2 * r], atol=atol
                        ), "Bad recovery of singular values!"
                        # check that recovered matrix is the same
                        core_U, core_S, core_Vt = truncate_core(
                            2 * r, core_U, core_S, core_Vt
                        )
                        mat_recons = (
                            ro_Q @ core_U @ torch.diag(core_S) @ core_Vt @ lo_Qt
                        )
                        recons_err = torch.dist(mat, mat_recons) / mat.norm()
                        assert recons_err <= abs(
                            recons_frob_tol[dtype]
                        ), "Bad SSVD reconstruction!"
                        # check that a few singular vectors are correct: avoid
                        # the ones that have the same singular values, since
                        # they aren't unique. Also compare outer prods, since
                        # multiplying left and right by -1 yields same result.
                        left = ro_Q @ core_U[:, r:]
                        right = core_Vt[r:, :] @ lo_Qt

                        for idx in range(svectors_check_n):
                            recons_err = torch.dist(
                                torch.outer(left[:, idx], right[idx]),
                                torch.outer(U[:, idx + r], Vt[idx + r]),
                            ) / (U[:, idx + r].norm() * Vt[idx + r].norm())
                            assert recons_err <= abs(
                                svectors_frob_tol[dtype]
                            ), "Bad recovery of singular vectors!"
                        # check a-posteriori error. With 30 measurements, we
                        # know that the probability of the actual error being
                        # twice as big is less than 1%, and keeps shrinking
                        # exponentially: a_posteriori_error_bounds(30, 1)
                        # Therefore, a very small estimate almost certainly
                        # guarantees that the approximation is good, and we
                        # have enough rank in our truncated SVD.
                        for adjoint in [True, False]:
                            _, _, err_estimate = a_posteriori_error(
                                mat,
                                mat_recons,
                                post,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                adjoint=adjoint,
                            )[0]
                            assert (
                                err_estimate / mat.numel()
                            ) <= atol, "Bad a-posteriori estimate!"
                        # cleanup HDF5 tmpdir
                        tmpdir.cleanup()


def test_dssvd_asymmetric_poly(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    atol_poly,
    heights_widths_ranks_outer_inner_posteriori,
    steep_poly_decay,
    recons_frob_tol,
    svectors_frob_tol,
    svectors_check_n,
):
    """Test distributed SSVD on asymmetric polynomial matrices.

    Sample matrices, and compute their ``torch.svd`` as well as the sketched
    SVD. Test that:
    * Recovered singular values are correct
    * Recovered matrix is correct
    * The first bulk singular vectors are correct
    * A-posteriori error estimation is low
    """
    for seed in rng_seeds:
        for dtype, atol in atol_poly.items():
            for (
                (h, w),
                r,
                (outer, inner),
                post,
            ) in heights_widths_ranks_outer_inner_posteriori:
                for dec in steep_poly_decay:
                    # we create matrix and full SVD on cpu, it is faster
                    mat = SynthMat.poly_decay(
                        shape=(h, w),
                        rank=r,
                        decay=dec,
                        symmetric=False,
                        seed=seed,
                        dtype=dtype,
                        device="cpu",
                    )
                    U, S, Vt = torch.linalg.svd(mat)
                    # then we test decomposition on both devices
                    for device in torch_devices:
                        mat = mat.to(device)
                        U = U.to(device)
                        S = S.to(device)
                        Vt = Vt.to(device)
                        #
                        tmpdir = tempfile.TemporaryDirectory()
                        (ro_Q, core_U, core_S, core_Vt, lo_Qt) = mock_dssvd(
                            tmpdir.name, mat, device, dtype, outer, inner, seed
                        )
                        # check that singular values are correct
                        assert torch.allclose(
                            S[: 2 * r], core_S[: 2 * r], atol=atol
                        ), "Bad recovery of singular values!"
                        # check that recovered matrix is the same
                        core_U, core_S, core_Vt = truncate_core(
                            2 * r, core_U, core_S, core_Vt
                        )
                        mat_recons = (
                            ro_Q @ core_U @ torch.diag(core_S) @ core_Vt @ lo_Qt
                        )
                        recons_err = torch.dist(mat, mat_recons) / mat.norm()
                        assert recons_err <= abs(
                            recons_frob_tol[dtype]
                        ), "Bad SSVD reconstruction!"
                        # check that a few singular vectors are correct: avoid
                        # the ones that have the same singular values, since
                        # they aren't unique. Also compare outer prods, since
                        # multiplying left and right by -1 yields same result.
                        left = ro_Q @ core_U[:, r:]
                        right = core_Vt[r:, :] @ lo_Qt

                        for idx in range(svectors_check_n):
                            recons_err = torch.dist(
                                torch.outer(left[:, idx], right[idx]),
                                torch.outer(U[:, idx + r], Vt[idx + r]),
                            ) / (U[:, idx + r].norm() * Vt[idx + r].norm())
                            assert recons_err <= abs(
                                svectors_frob_tol[dtype]
                            ), "Bad recovery of singular vectors!"
                        # check a-posteriori error. With 30 measurements, we
                        # know that the probability of the actual error being
                        # twice as big is less than 1%, and keeps shrinking
                        # exponentially: a_posteriori_error_bounds(30, 1)
                        # Therefore, a very small estimate almost certainly
                        # guarantees that the approximation is good, and we
                        # have enough rank in our truncated SVD.
                        for adjoint in [True, False]:
                            _, _, err_estimate = a_posteriori_error(
                                mat,
                                mat_recons,
                                post,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                adjoint=adjoint,
                            )[0]
                            assert (
                                err_estimate / mat.numel()
                            ) <= atol, "Bad a-posteriori estimate!"
                        # cleanup HDF5 tmpdir
                        tmpdir.cleanup()


def test_dssvd_symmetric_poly(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    atol_poly,
    dims_ranks_outer_inner_posteriori,
    steep_poly_decay,
    recons_frob_tol,
    svectors_frob_tol,
    svectors_check_n,
):
    """Test distributed SSVD on symmetric polynomial matrices.

    Sample matrices, and compute their ``torch.svd`` as well as the sketched
    SVD. Test that:
    * Recovered singular values are correct
    * Recovered matrix is correct and symmetric
    * The first bulk singular vectors are correct
    * A-posteriori error estimation is low
    """
    for seed in rng_seeds:
        for dtype, atol in atol_poly.items():
            for d, r, (outer, inner), post in dims_ranks_outer_inner_posteriori:
                for dec in steep_poly_decay:
                    # we create matrix and full SVD on cpu, it is faster
                    mat = SynthMat.poly_decay(
                        shape=(d, d),
                        rank=r,
                        decay=dec,
                        symmetric=True,
                        seed=seed,
                        dtype=dtype,
                        device="cpu",
                    )
                    U, S, Vt = torch.linalg.svd(mat)

                    # then we test decomposition on both devices
                    for device in torch_devices:
                        mat = mat.to(device)
                        U = U.to(device)
                        S = S.to(device)
                        Vt = Vt.to(device)
                        #
                        tmpdir = tempfile.TemporaryDirectory()
                        (ro_Q, core_U, core_S, core_Vt, lo_Qt) = mock_dssvd(
                            tmpdir.name, mat, device, dtype, outer, inner, seed
                        )
                        # check that singular values are correct
                        assert torch.allclose(
                            S[: 2 * r], core_S[: 2 * r], atol=atol
                        ), "Bad recovery of singular values!"
                        # check that recovered matrix is the same
                        core_U, core_S, core_Vt = truncate_core(
                            2 * r, core_U, core_S, core_Vt
                        )
                        mat_recons = (
                            ro_Q @ core_U @ torch.diag(core_S) @ core_Vt @ lo_Qt
                        )
                        recons_err = torch.dist(mat, mat_recons) / mat.norm()
                        assert recons_err <= abs(
                            recons_frob_tol[dtype]
                        ), "Bad SSVD reconstruction!"
                        # check that a few singular vectors are correct: avoid
                        # the ones that have the same singular values, since
                        # they aren't unique. Also compare outer prods, since
                        # multiplying left and right by -1 yields same result.
                        left = ro_Q @ core_U[:, r:]
                        right = core_Vt[r:, :] @ lo_Qt

                        for idx in range(svectors_check_n):
                            recons_err = torch.dist(
                                torch.outer(left[:, idx], right[idx]),
                                torch.outer(U[:, idx + r], Vt[idx + r]),
                            ) / (U[:, idx + r].norm() * Vt[idx + r].norm())
                            assert recons_err <= abs(
                                svectors_frob_tol[dtype]
                            ), "Bad recovery of singular vectors!"
                        # check a-posteriori error. With 30 measurements, we
                        # know that the probability of the actual error being
                        # twice as big is less than 1%, and keeps shrinking
                        # exponentially: a_posteriori_error_bounds(30, 1)
                        # Therefore, a very small estimate almost certainly
                        # guarantees that the approximation is good, and we
                        # have enough rank in our truncated SVD.
                        for adjoint in [True, False]:
                            _, _, err_estimate = a_posteriori_error(
                                mat,
                                mat_recons,
                                post,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                adjoint=adjoint,
                            )[0]
                            assert (
                                err_estimate / mat.numel()
                            ) <= atol, "Bad a-posteriori estimate!"
                        # cleanup HDF5 tmpdir
                        tmpdir.cleanup()


# ##############################################################################
# # SKETCHED EIGH
# ##############################################################################
def test_dseigh_exp(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    atol_exp,
    dims_ranks_outer_inner_posteriori,
    steep_exp_decay,
    recons_frob_tol,
    svectors_frob_tol,
    svectors_check_n,
):
    """Test distributed SEIGH on (PSD and non-PSD) exponential matrices.

    Sample matrices, and compute their ``torch.eigh`` as well as the sketched
    EIGH. Test that:
    * Recovered eigenvalues are correct
    * Recovered matrix is correct and symmetric
    * The first bulk eigenvectors are correct
    * A-posteriori error estimation is low
    """
    for seed in rng_seeds:
        for dtype, atol in atol_exp.items():
            for d, r, (outer, inner), post in dims_ranks_outer_inner_posteriori:
                for dec in steep_exp_decay:
                    for psd in (True, False):
                        # we create matrix and full eigh on cpu, it is faster
                        mat = SynthMat.exp_decay(
                            shape=(d, d),
                            rank=r,
                            decay=dec,
                            symmetric=True,
                            seed=seed,
                            dtype=dtype,
                            device="cpu",
                            psd=psd,
                        )
                        S, U = torch.linalg.eigh(mat)  # mat = U @ diag(S) @ U.T
                        # sort eigdec in descending eigval magnitude:
                        _, perm = abs(S).sort(descending=True)
                        S, U = S[perm], U[:, perm]
                        # make sure we actually have negative eigvals
                        if not psd:
                            assert (S < -abs(atol)).any(), (
                                "non-PSD should have " + "negative eigvals!"
                            )
                        # then we test decomposition on both devices
                        for device in torch_devices:
                            mat = mat.to(device)
                            S = S.to(device)
                            U = U.to(device)
                            #
                            tmpdir = tempfile.TemporaryDirectory()
                            (Q, core_U, core_S) = mock_dseigh(
                                tmpdir.name,
                                mat,
                                device,
                                dtype,
                                outer,
                                inner,
                                seed,
                            )
                            # check that singular values are correct
                            assert torch.allclose(
                                S[: 2 * r], core_S[: 2 * r], atol=atol
                            ), "Bad recovery of singular values!"

                            # check that recovered matrix is the same
                            core_U, core_S = truncate_core(
                                2 * r, core_U, core_S
                            )
                            mat_recons = (
                                Q @ core_U @ torch.diag(core_S) @ core_U.T @ Q.T
                            )
                            # check that reconstruction is symmetric
                            assert torch.allclose(
                                mat_recons, mat_recons.T, atol=atol
                            ), "Recovered sseigh is not symmetric!"
                            recons_err = (
                                torch.dist(mat, mat_recons) / mat.norm()
                            )
                            assert recons_err <= abs(
                                recons_frob_tol[dtype]
                            ), f"Bad SSVD reconstruction! {recons_err}"

                            # check a few singular vectors are correct: avoid
                            # the ones that have the same singular values, since
                            # they aren't unique. And compare outer prods, since
                            # multiplying both sides by -1 yields same result.
                            recons_evs = Q @ core_U[:, r:]
                            for idx in range(svectors_check_n):
                                recons_err = torch.dist(
                                    recons_evs[:, idx], U[:, r + idx]
                                ).item()
                                recons_err_flip = torch.dist(
                                    recons_evs[:, idx], -U[:, r + idx]
                                ).item()
                                recons_err = min(recons_err, recons_err_flip)
                                assert recons_err <= abs(
                                    svectors_frob_tol[dtype]
                                ), "Bad recovery of singular vectors!"
                            # check a-posteriori error. With 30 measurements, we
                            # know that the prob of the actual error being
                            # twice as big is less than 1%, and keeps shrinking
                            # exponentially: a_posteriori_error_bounds(30, 1)
                            # Therefore, a very small estimate almost certainly
                            # guarantees that the approximation is good, and we
                            # have enough rank in our truncated SVD.
                            for adjoint in [True, False]:
                                _, _, err_estimate = a_posteriori_error(
                                    mat,
                                    mat_recons,
                                    post,
                                    seed=seed,
                                    dtype=dtype,
                                    device=device,
                                    adjoint=adjoint,
                                )[0]
                                assert (
                                    err_estimate / mat.numel()
                                ) <= atol, "Bad a-posteriori estimate!"
                            # cleanup HDF5 tmpdir
                            tmpdir.cleanup()


def test_dseigh_poly(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    atol_poly,
    dims_ranks_outer_inner_posteriori,
    steep_poly_decay,
    recons_frob_tol,
    svectors_frob_tol,
    svectors_check_n,
):
    """Test distributed SEIGH on (PSD and non-PSD) polynomial matrices.

    Sample matrices, and compute their ``torch.eigh`` as well as the sketched
    EIGH. Test that:
    * Recovered eigenvalues are correct
    * Recovered matrix is correct and symmetric
    * The first bulk eigenvectors are correct
    * A-posteriori error estimation is low
    """
    for seed in rng_seeds:
        for dtype, atol in atol_poly.items():
            for d, r, (outer, inner), post in dims_ranks_outer_inner_posteriori:
                for dec in steep_poly_decay:
                    for psd in (True, False):
                        # we create matrix and full eigh on cpu, it is faster
                        mat = SynthMat.poly_decay(
                            shape=(d, d),
                            rank=r,
                            decay=dec,
                            symmetric=True,
                            seed=seed,
                            dtype=dtype,
                            device="cpu",
                            psd=psd,
                        )
                        S, U = torch.linalg.eigh(mat)  # mat = U @ diag(S) @ U.T
                        # sort eigdec in descending eigval magnitude:
                        _, perm = abs(S).sort(descending=True)
                        S, U = S[perm], U[:, perm]
                        # make sure we actually have negative eigvals
                        if not psd:
                            assert (S < -abs(atol)).any(), (
                                "non-PSD should have " + "negative eigvals!"
                            )
                        # then we test decomposition on both devices
                        for device in torch_devices:
                            mat = mat.to(device)
                            S = S.to(device)
                            U = U.to(device)
                            #
                            tmpdir = tempfile.TemporaryDirectory()
                            (Q, core_U, core_S) = mock_dseigh(
                                tmpdir.name,
                                mat,
                                device,
                                dtype,
                                outer,
                                inner,
                                seed,
                            )
                            # check that singular values are correct
                            assert torch.allclose(
                                S[: 2 * r], core_S[: 2 * r], atol=atol
                            ), "Bad recovery of singular values!"

                            # check that recovered matrix is the same
                            core_U, core_S = truncate_core(
                                2 * r, core_U, core_S
                            )
                            mat_recons = (
                                Q @ core_U @ torch.diag(core_S) @ core_U.T @ Q.T
                            )
                            # check that reconstruction is symmetric
                            assert torch.allclose(
                                mat_recons, mat_recons.T, atol=atol
                            ), "Recovered sseigh is not symmetric!"
                            recons_err = (
                                torch.dist(mat, mat_recons) / mat.norm()
                            )
                            assert recons_err <= abs(
                                recons_frob_tol[dtype]
                            ), f"Bad SSVD reconstruction! {recons_err}"

                            # check a few singular vectors are correct: avoid
                            # the ones that have the same singular values, since
                            # they aren't unique. And compare outer prods, since
                            # multiplying both sides by -1 yields same result.
                            recons_evs = Q @ core_U[:, r:]
                            for idx in range(svectors_check_n):
                                recons_err = torch.dist(
                                    recons_evs[:, idx], U[:, r + idx]
                                ).item()
                                recons_err_flip = torch.dist(
                                    recons_evs[:, idx], -U[:, r + idx]
                                ).item()
                                recons_err = min(recons_err, recons_err_flip)
                                assert recons_err <= abs(
                                    svectors_frob_tol[dtype]
                                ), "Bad recovery of singular vectors!"
                            # check a-posteriori error. With 30 measurements, we
                            # know that the prob of the actual error being
                            # twice as big is less than 1%, and keeps shrinking
                            # exponentially: a_posteriori_error_bounds(30, 1)
                            # Therefore, a very small estimate almost certainly
                            # guarantees that the approximation is good, and we
                            # have enough rank in our truncated SVD.
                            for adjoint in [True, False]:
                                _, _, err_estimate = a_posteriori_error(
                                    mat,
                                    mat_recons,
                                    post,
                                    seed=seed,
                                    dtype=dtype,
                                    device=device,
                                    adjoint=adjoint,
                                )[0]
                                assert (
                                    err_estimate / mat.numel()
                                ) <= atol, "Bad a-posteriori estimate!"
                            # cleanup HDF5 tmpdir
                            tmpdir.cleanup()
