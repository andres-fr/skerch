#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for in-core sketched (sub-)diagonal estimation."""


import pytest
import torch

from skerch.subdiagonals import subdiagpp
from skerch.synthmat import SynthMat

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

    Like the main diagonal test, but all matrices are smaller and asymmetric,
    since subdiag recovery behaves here similarly to asymmetric main diagonals,
    and we care more about discarding off-by-one errors and such things.
    """
    dims, rank = 100, 10
    if request.config.getoption("--quick"):
        subdiag_idxs = torch.arange(-99, 100, 22).tolist()
    else:
        subdiag_idxs = torch.arange(-99, 100).tolist()
    result = [
        # fast-decay: low-rank recovery works nicely
        (dims, rank, 0.5, False, subdiag_idxs, 0, 20, 1e-4),
    ]
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

                        try:
                            assert (
                                rel_err <= rtol
                            ), "Incorrect subdiagonal recovery!"
                        except Exception as e:
                            print("!!!", e)
                            breakpoint()


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
