#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for in-core sketched (sub-)diagonal estimation."""


import pytest
import torch

from skerch.subdiagonals import subdiagpp
from skerch.subdiagonals import subdiag_hadamard_pattern
from skerch.subdiagonals import serrated_hadamard_pattern
from skerch.synthmat import SynthMat
from skerch.utils import gaussian_noise
from . import rng_seeds, torch_devices  # noqa: F401


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dim_rank_decay_sym_diags_meas_defl_rtol(request):
    """Test cases for main diagonal estimation on different matrices.

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
        (dims, rank, 0.5, False, [0], round(dims * 0.995), 0, 0.2),
        # slow-decay: just Hutch behaves the same as with fast decay
        (dims, rank, 0.01, True, [0], round(dims * 0.995), 0, 0.02),
        (dims, rank, 0.01, False, [0], round(dims * 0.995), 0, 0.2),
        # fast-decay: just deflating is great (also for asymmetric)
        (dims, rank, 0.5, True, [0], 0, rank + 10, 1e-4),
        (dims, rank, 0.5, False, [0], 0, rank + 10, 1e-4),
        # slow-decay: deflating is less good and affected by asym
        (dims, rank, 0.01, True, [0], 0, rank * 3, 0.05),
        (dims, rank, 0.01, False, [0], 0, rank * 4, 0.1),
        # slow-decay: A lot of Hutch are needed, but deflation tends to help
        # fo asym
        (dims, rank, 0.01, True, [0], round(dims * 0.7), rank * 3, 0.02),
        (dims, rank, 0.01, False, [0], round(dims * 0.7), rank * 4, 0.1),
    ]
    return result


@pytest.fixture
def dim_rank_decay_sym_subdiags_meas_defl_rtol(request):
    """Test cases for subdiagonal estimation on different matrices.

    Note that very extremal subdiagonals have few entries, and we are better
    off just directly measuring them. For this reason, this test focuses on
    subdiagonals in the mid-section. It also focuses on asymmetric matrices
    only, since symmetry mostly impacts the diagonal.
    """
    dims, rank = 100, 10
    if request.config.getoption("--quick"):
        subdiag_idxs = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]
    else:
        subdiag_idxs = [-50, -30, -10, 10, 30, 50]
    #
    result = [
        # fast-decay: low-rank recovery works nicely
        (dims, rank, 0.5, False, subdiag_idxs, 0, rank * 2, 0.001),
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
    """ """
    result = [
        [0, 1, 2, 3, 4, 5],
        [0, -1, -2, -3, -4, -5],
        [-1, 5, -9, 9],
    ]
    return result


@pytest.fixture
def serrated_hadamard_dims_chunks():
    """ """
    result = [(100, 2), (100, 3), (100, 10), (100, 100)]
    return result


@pytest.fixture
def hadamard_atol():
    """ """
    result = {torch.float64: 1e-15, torch.float32: 1e-5}
    return result


# ##############################################################################
# #
# ##############################################################################
def test_main_diags(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_diags_meas_defl_rtol,
):
    """Test diagonal estimation on exponential matrices."""
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
                            meas,
                            mat,
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


def test_subdiags(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    dim_rank_decay_sym_subdiags_meas_defl_rtol,
):
    """Test diagonal estimation on exponential matrices."""
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
                            meas,
                            mat,
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


def test_subdiag_hadamard_pattern(
    rng_seeds,  # noqa: F811
    hadamard_atol,
    torch_devices,  # noqa: F811
    subdiag_hadamard_idxs,
):
    """ """
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


def test_serrated_hadamard_pattern(
    rng_seeds,  # noqa: F811
    hadamard_atol,
    torch_devices,  # noqa: F811
    serrated_hadamard_dims_chunks,
):
    """ """
    for seed in rng_seeds:
        for dtype, atol in hadamard_atol.items():
            for device in torch_devices:
                # Simple tests with unit vector:
                v = torch.ones(10, dtype=dtype, device=device)
                # chunk size must be 1 or greater
                with pytest.raises(ValueError):
                    serrated_hadamard_pattern(v, 0)
                #
                had = serrated_hadamard_pattern(v, 1, use_fft=False)
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(v, 2, use_fft=False)
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                #
                had = serrated_hadamard_pattern(v, 4, use_fft=False)
                assert torch.allclose(
                    had,
                    torch.tensor(
                        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
                        dtype=dtype,
                        device=device,
                    ),
                    atol=atol,
                ), "Incorrect serrated result!"
                # random test with more complex responses
                for dims, chunk in serrated_hadamard_dims_chunks:
                    v = gaussian_noise(
                        dims,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    had = serrated_hadamard_pattern(v, chunk, use_fft=False)
                    had_fft = serrated_hadamard_pattern(v, chunk, use_fft=True)
                    assert torch.allclose(
                        had, had_fft, atol=atol
                    ), "Inconsistent FFT implementation of serrated Hadamard!"


# def test_combined_diags(
#     rng_seeds,  # noqa: F811
#     torch_devices,  # noqa: F811
#     dim_rank_decay_sym_subdiags_meas_defl_rtol,
# ):
#     """Test subdiagonal estimation on exponential matrices

#     Sample random asymmetric matrices, and retrieve arbitrary (sub-)diagonals
#     from them. Test that:
#     * Sketched diagonals are close to actual diagonals
#     * Sketches of linear combinations of diagonals are close to actual ones
#     * Lowtri evaluations?

#     """
#     for seed in rng_seeds:
#         for dtype in (torch.float64, torch.float32):
#             for device in torch_devices:
#                 for (
#                     dim,
#                     rank,
#                     decay,
#                     sym,
#                     diags,
#                     meas,
#                     defl,
#                     rtol,
#                 ) in dim_rank_decay_sym_subdiags_meas_defl_rtol:
#                     mat = SynthMat.exp_decay(
#                         shape=(dim, dim),
#                         rank=rank,
#                         decay=decay,
#                         symmetric=sym,
#                         seed=seed,
#                         dtype=dtype,
#                         device=device,
#                         psd=False,
#                     )
#                     for diag_idxs in diags:
#                         # retrieve the true diag
#                         diags = {
#                             idx: torch.diag(mat, diagonal=idx)
#                             for idx in diag_idxs
#                         }
#                         # matrix-free estimation of the diag
#                         diag_est, _, norms = subdiagpp(
#                             meas,
#                             mat,
#                             dtype,
#                             device,
#                             seed + 1,
#                             defl,
#                             diag_idxs,
#                         )
#                         breakpoint()
#                         # then assert
#                         dist = torch.dist(diag, diag_est)
#                         rel_err = (dist / torch.norm(diag)).item()
#                         assert rel_err <= rtol, "Incorrect diagonal recovery!"
