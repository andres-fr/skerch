#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for in-core sketched (sub-)diagonal estimation."""


import pytest
import torch

from skerch.subdiagonals import subdiagpp
from skerch.synthmat import SynthMat
from skerch.utils import gaussian_noise

from . import rng_seeds, torch_devices  # noqa: F401


import matplotlib.pyplot as plt


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def diag_dist_rtol():
    """Relative Frobenius error tolerance for the diagonal estimates."""
    result = 0.05
    return result


@pytest.fixture
def dim_rank_diags_meas_defl(request):
    """Test cases for matrix sizes, ranks and subdiagpp measurements

    Entries are in the form
    ``(dims, rank, (diag_idxs), num_meas, defl)``, where dims is the number of
    dimensions of a square test matrix of exp-decaying singular values with
    given minimum ``rank``. The ``num_meas`` parameter specifies the number of
    measurements for the hutchinson diagonal estimator, and ``defl`` the rank
    of the deflation orthogonal projector.

    .. note::

      The given diagonal indices should not exceed the furthest away
      diagonal, which depends on the shape.
    """
    result = [
        (1000, 0, [0], 750, 250),
        # (1000, 200, torch.arange(-9, 9).tolist(), 0, 200),
        # (10, 5, torch.arange(-9, 9).tolist(), 10),
        # (100, 20, torch.arange(-99, 100, 99).tolist(), 20),
    ]
    # if request.config.getoption("--quick"):
    #     result = result[:4]
    return result


@pytest.fixture
def exp_decays():
    """Decay values for the synthetic matrix. The larger, the faster decay."""
    return [0.01]


# ##############################################################################
# #
# ##############################################################################
def test_subdiags(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    diag_dist_rtol,
    dim_rank_diags_meas_defl,
    exp_decays,
):
    """Test SSVD on asymmetric exponential matrices.

    Sample random asymmetric matrices, and retrieve arbitrary (sub-)diagonals
    from them. Test that:
    * Sketched diagonals are close to actual diagonals
    * Sketches of linear combinations of diagonals are close to actual ones
    * Lowtri evaluations?
    """
    for seed in rng_seeds:
        for dim, rank, diags, meas, defl in dim_rank_diags_meas_defl:
            for dtype in (torch.float64, torch.float32):
                for device in torch_devices:
                    for decay in exp_decays:
                        for sym in (True, False):
                            print(
                                "\n\n\n\n>>>>",
                                dim,
                                rank,
                                diags,
                                dtype,
                                device,
                                decay,
                                "SYM" if sym else "ASYM",
                            )
                            mat = SynthMat.exp_decay(
                                shape=(dim, dim),
                                rank=rank,
                                decay=decay,
                                symmetric=False,
                                seed=seed,
                                dtype=dtype,
                                device=device,
                                psd=False,
                            )

                            for diag_idx in diags:
                                diag_idx = 0
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
                                print("\n", rel_err)
                                breakpoint()
                                try:
                                    assert (
                                        rel_err <= diag_dist_rtol
                                    ), "Incorrect diagonal recovery!"
                                except:
                                    # plt.plot(diag, color="grey"); plt.plot(diag_est); plt.show()
                                    # plt.plot(diag, color="grey"); plt.plot(diag - diag_est); plt.show()

                                    # plt.hist(diag, bins=100, color="grey"); plt.hist(diag - diag_est, alpha=0.5, bins=100); plt.show()

                                    breakpoint()
