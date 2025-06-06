#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for in-core sketched (sub-)diagonal estimation."""


import pytest
import torch

from skerch.subdiagonals import subdiag_hadamard_pattern, subdiagpp, xdiag
from skerch.synthmat import SynthMat
from skerch.utils import gaussian_noise

from . import rng_seeds, torch_devices  # noqa: F401


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dim_rank_decay_sym_diags_meas_defl_rtol(request):
    """Test cases for main diag estimation on different matrices via Diag++.

    Entries are in the form
    ``(dim, rank, decay, sym, [diag_idxs], num_meas, defl, rtol)``, where:

    * ``dim`` is the number of dimensions of a square test matrix of
      exp-decaying singular values
    * ``rank`` is the number of unit singular values in the test matrix
    * ``decay`` is the speed of the exp-decay of singular values after ``rank``.
      The larger, the faster decay.
    * ``sym`` is a boolean specifying whether the matrix is symmetric
    * ``diag_idxs`` is a collection of tuples of integers specifying the
      diagonal indices to be tested: 0 is the main diagonal, 1 the diagonal
      above, -1 the diagonal below, and so on.
    * ``num_meas`` specifies the number of measurements for the Hutchinson
      diagonal estimator
    * ``defl`` specifies the rank of the orthogonal projector for the
      Hutchinson++ deflation.

    .. note::

      The given diagonal indices should not exceed the furthest away
      diagonal, which depends on the shape.

    This is a bit of a complicated fixture, the reason being that the tolerance
    and number of measurements needed depend heavily on the type of matrix
    encountered: symmetric random matrices have strong diagonals, which lead
    to much faster Hutchinson convergence. Low-rank matrices can be effectively
    characterized via deflation, which also helps. And so on.
    """
    dims, rank = 1000, 100
    if request.config.getoption("--quick"):
        dims, rank = 500, 50
    result = [
        # fast-decay: just Hutch does poorly, but better if symmetric
        (dims, rank, 0.5, True, [0], round(dims * 0.995), 0, 0.02),
        (dims, rank, 0.5, False, [0], round(dims * 0.995), 0, 0.1),
        # slow-decay: just Hutch behaves a bit worse as with fast decay
        (dims, rank, 0.01, True, [0], round(dims * 0.995), 0, 0.02),
        (dims, rank, 0.01, False, [0], round(dims * 0.995), 0, 0.1),
        # fast-decay: just deflating is great (also for asymmetric)
        (dims, rank, 0.5, True, [0], 0, rank + 10, 1e-5),
        (dims, rank, 0.5, False, [0], 0, rank + 10, 1e-4),
        # slow-decay: deflating is less good and affected by asym
        (dims, rank, 0.01, True, [0], 0, rank * 3, 0.05),
        (dims, rank, 0.01, False, [0], 0, rank * 4, 0.06),
        # slow-decay: A lot of Hutch are needed, but deflation tends to help
        # for asym
        (dims, rank, 0.01, True, [0], round(dims * 0.7), rank * 3, 0.02),
        (dims, rank, 0.01, False, [0], round(dims * 0.7), rank * 4, 0.05),
    ]
    return result


@pytest.fixture
def dim_rank_decay_sym_subdiags_meas_defl_rtol(request):
    """Test cases for subdiag estimation on different matrices via Diag++.

    Note that very extremal subdiagonals have few entries, and we are better
    off just directly measuring them. For this reason, this test focuses on
    subdiagonals in the mid-section. It also focuses on asymmetric matrices
    only, since symmetry mostly impacts the main diagonal.
    """
    dims, rank = 100, 10
    if request.config.getoption("--quick"):
        subdiag_idxs = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]
    else:
        subdiag_idxs = [-30, -10, 10, 30]
    #
    result = [
        # fast-decay: low-rank recovery works nicely
        (dims, rank, 0.5, False, subdiag_idxs, 0, rank * 2, 5e-5),
        # mid-decay: low-rank recovery works less nicely for subdiags
        (dims, rank, 0.1, False, subdiag_idxs, 0, rank * 3, 0.03),
        # fast-decay: Hutch is bad for central subdiags (worse than direct)...
        (dims, rank, 0.5, False, subdiag_idxs, 99, 0, 0.2),
        # but as with main diag, Hutch doesn't depend much on the decay
        (dims, rank, 0.1, False, subdiag_idxs, 99, 0, 0.2),
    ]
    return result


@pytest.fixture
def subdiag_hadamard_idxs():
    """Subdiagonal indices to test with the Hadamard pattern."""
    result = [
        [0, 1, 2, 3, 4, 5],
        [0, -1, -2, -3, -4, -5],
        [-1, 5, -9, 9],
    ]
    return result


@pytest.fixture
def hadamard_atol():
    """Absolute tolerances to test Hadamard pattern."""
    result = {torch.float64: 1e-13, torch.float32: 1e-5}
    return result


@pytest.fixture
def dim_rank_decay_sym_meas_rtol_xdiag(request):
    """Test cases for main diag estimation on different matrices via XDiag.

    Similarly to the corresponding Diag++ fixture, entries are in the form
    ``(dim, rank, decay, sym, [diag_idxs], num_meas, rtol)``. The main
    difference here is that there is no distinction between deflation and
    Hutchinson measurements, and we always target the main diagonal.

    Another difference is the generally lower tolerance and smaller amount
    of measurements, due to the superior performance of XDiag.
    """
    dims, rank = 1000, 100
    if request.config.getoption("--quick"):
        dims, rank = 500, 50
    result = [
        # fast-decay: performance is good, but needs 2x more measurements than
        # diagpp since projector has half the rank. Also worse for asymmetric
        (dims, rank, 0.5, True, rank * 2 + 20, 5e-5),
        (dims, rank, 0.5, False, rank * 2 + 20, 5e-4),
        # slow-decay: problem here is that a lot of Hutch are needed, but
        # in xdiag those are tied to the projector rank, so we can't
        # regulate the memory-runtime tradeoff.
        # rank-limited performance is thus less good. Also affected by asym.
        (dims, rank, 0.01, True, rank * 5, 0.07),
        (dims, rank, 0.01, False, rank * 6, 0.17),
    ]
    return result


# ##############################################################################
# # HADAMARD
# ##############################################################################
def test_subdiag_hadamard_pattern(
    rng_seeds,  # noqa: F811
    hadamard_atol,
    torch_devices,  # noqa: F811
    subdiag_hadamard_idxs,
):
    """Test case for generic Hadamard pattern generator.

    Given a vector and a set of indices, the Hadamard pattern is a
    non-circular convolution using indices as deltas in the impulse response.
    This pattern is useful to estimate sub-diagonals using Hutchinson.

    This test case generates random vectors and checks that the responses
    are right for different indices. It also checks that the FFTconv
    implementation and the plain one yield same results.
    """
    for seed in rng_seeds:
        for dtype, atol in hadamard_atol.items():
            for device in torch_devices:
                # Simple tests with delta responses:
                v = gaussian_noise(100, seed=seed, dtype=dtype, device=device)
                had = subdiag_hadamard_pattern(v, [], use_fft=False)
                assert had.norm() == 0, "Empty idxs should yield zeros!"
                had = subdiag_hadamard_pattern(v, [0], use_fft=False)
                assert torch.allclose(
                    v, had, atol=atol
                ), "Delta(0) should yield same result!"
                had = subdiag_hadamard_pattern(v, [1], use_fft=False)
                assert (had[0] == 0) and (
                    torch.allclose(had[1:], v[:-1], atol=atol)
                ), "Delta(1) should shift vector one position to the right!"
                had = subdiag_hadamard_pattern(v, [-1], use_fft=False)
                assert (had[-1] == 0) and (
                    torch.allclose(had[:-1], v[1:], atol=atol)
                ), "Delta(-1) should shift vector one position to the left!"
                # random test with more complex responses
                for idxs in subdiag_hadamard_idxs:
                    v = gaussian_noise(
                        100,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    had = subdiag_hadamard_pattern(v, idxs, use_fft=False)
                    had_fft = subdiag_hadamard_pattern(v, idxs, use_fft=True)
                    assert torch.allclose(
                        had, had_fft, atol=atol
                    ), "Inconsistent FFT implementation of Hadamard pattern!"
                    if not idxs:
                        assert had.norm() == 0, "Hadamard should be empty!"


# ##############################################################################
# # DIAGPP
# ##############################################################################
def test_main_diags_diagpp(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_diags_meas_defl_rtol,
):
    """Test case for main diagonal estimation with Diag++.

    This test creates an ``exp_decay`` random (square) test matrix, and checks
    that its main diagonal is estimated within ``rtol``.

    Depending on the passed fixture, the matrix will have different order,
    symmetry and spectral decay.
    """
    for seed in rng_seeds:
        for dtype in (torch.float64, torch.float32):
            for device in torch_devices:
                for (
                    dim,
                    rank,
                    decay,
                    sym,
                    diags,
                    meas,
                    defl,
                    rtol,
                ) in dim_rank_decay_sym_diags_meas_defl_rtol:
                    mat = SynthMat.exp_decay(
                        shape=(dim, dim),
                        rank=rank,
                        decay=decay,
                        symmetric=sym,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                        psd=False,
                    )
                    for diag_idx in diags:
                        # retrieve the true diag
                        diag = torch.diag(mat, diagonal=diag_idx)
                        # matrix-free estimation of the diag
                        diag_est, _, norms = subdiagpp(
                            mat,
                            meas,
                            dtype,
                            device,
                            seed + 1,
                            defl,
                            diag_idx,
                        )
                        # then assert
                        dist = torch.dist(diag, diag_est)
                        rel_err = (dist / torch.norm(diag)).item()
                        assert rel_err <= rtol, "Incorrect diagonal recovery!"


def test_subdiags_diagpp(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_subdiags_meas_defl_rtol,
):
    """Test case for subdiagonal estimation with Diag++.

    This test creates an ``exp_decay`` random test matrix, and checks that
    several of its subdiagonals are estimated within ``rtol``.

    """
    for seed in rng_seeds:
        for dtype in (torch.float64, torch.float32):
            for device in torch_devices:
                for (
                    dim,
                    rank,
                    decay,
                    sym,
                    diags,
                    meas,
                    defl,
                    rtol,
                ) in dim_rank_decay_sym_subdiags_meas_defl_rtol:
                    mat = SynthMat.exp_decay(
                        shape=(dim, dim),
                        rank=rank,
                        decay=decay,
                        symmetric=sym,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                        psd=False,
                    )
                    for diag_idx in diags:
                        # retrieve the true diag
                        diag = torch.diag(mat, diagonal=diag_idx)
                        # matrix-free estimation of the diag
                        diag_est, _, norms = subdiagpp(
                            mat,
                            meas,
                            dtype,
                            device,
                            seed + 1,
                            defl,
                            diag_idx,
                        )
                        # then assert
                        dist = torch.dist(diag, diag_est)
                        rel_err = (dist / torch.norm(diag)).item()
                        assert rel_err <= rtol, "Incorrect subdiag recovery!"


# ##############################################################################
# # XDIAG
# ##############################################################################
def test_deflation_projector_xdiag(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_meas_rtol_xdiag,
):
    """Test case for rank-1 deflation in XDiag.

    This test creates an ``exp_decay`` random (square) test matrix, and checks
    that the rank-1 deflation via the ``s`` vectors is indeed equivalent to
    removing one measurement.
    """
    for seed in rng_seeds:
        for dtype, atol in ((torch.float64, 1e-12), (torch.float32, 1e-4)):
            for device in torch_devices:
                dim, rank, meas = 50, 5, 20
                mat = SynthMat.exp_decay(
                    shape=(dim, dim),
                    rank=rank,
                    decay=0.5,
                    symmetric=False,
                    seed=seed,
                    dtype=dtype,
                    device=device,
                    psd=False,
                )
                _, (Q, _, S, rand_lop), _ = xdiag(
                    mat, meas, dtype, device, seed + 1, with_variance=False
                )
                for i in range(meas // 2):
                    range_skip = list(range(meas // 2))
                    range_skip.pop(i)
                    meas_skip = torch.zeros(
                        (dim, meas // 2 - 1), dtype=dtype, device=device
                    )
                    for j, jskip in enumerate(range_skip):
                        meas_skip[:, j] = mat @ rand_lop.get_row(
                            jskip, dtype, device
                        )
                    Q_i = torch.linalg.qr(meas_skip)[0]
                    #
                    Proj = Q @ Q.T
                    Proj_i = Q_i @ Q_i.T
                    QSi = Q @ S[:, i]
                    dist1 = torch.dist(Proj_i, Proj)
                    dist2 = torch.dist(Proj_i, Proj - torch.outer(QSi, QSi))
                    #
                    assert torch.isclose(
                        dist1, torch.ones_like(dist1), atol=atol
                    ), "Projectors not having a distance of 1"
                    assert torch.isclose(
                        dist2, torch.zeros_like(dist1), atol=atol
                    ), "Projectors should have near-zero distance!"


def test_main_diags_xdiag(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_meas_rtol_xdiag,
):
    """Test case for main diagonal estimation with XDiag.

    This test creates an ``exp_decay`` random (square) test matrix, and checks
    that its main diagonal is estimated within ``rtol``.
    """
    for seed in rng_seeds:
        for dtype in (torch.float64, torch.float32):
            for device in torch_devices:
                for (
                    dim,
                    rank,
                    decay,
                    sym,
                    meas,
                    rtol,
                ) in dim_rank_decay_sym_meas_rtol_xdiag:
                    mat = SynthMat.exp_decay(
                        shape=(dim, dim),
                        rank=rank,
                        decay=decay,
                        symmetric=sym,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                        psd=False,
                    )
                    # retrieve the true diag
                    diag = torch.diag(mat, diagonal=0)
                    # matrix-free estimation of the diag
                    diag_est, _, _ = xdiag(
                        mat,
                        meas,  # rank of Q is actually half this
                        dtype,
                        device,
                        seed + 1,
                        with_variance=False,
                    )
                    # then assert
                    dist = torch.dist(diag, diag_est)
                    rel_err = (dist / torch.norm(diag)).item()
                    assert rel_err <= rtol, "Incorrect XDiag recovery!"
