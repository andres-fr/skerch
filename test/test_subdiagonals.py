#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for in-core sketched (sub-)diagonal estimation."""


import pytest
import torch

from skerch.utils import gaussian_noise
from skerch.subdiagonals import do_stuff

from . import rng_seeds, torch_devices  # noqa: F401


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def entry_atol():
    """Absolute, entry-wise error tolerance for ``diff(Q.T @ Q, I)``."""
    result = 1e-5
    return result


@pytest.fixture
def heights_widths_diags_meas(request):
    """Test cases for different matrix shapes and (sub-)diagonal measurements

    Entries are in the form ``((h, w), (diag_idxs))``. Note that given
    diagonal indices should not exceed the furthest away diagonal, which
    depends on the shape.
    """
    result = [
        ((1000, 1000), torch.arange(-9, 9).tolist(), 500),
        ((10, 100), torch.arange(-9, 9).tolist(), 10),
        ((100, 10), torch.arange(-9, 9).tolist(), 10),
        ((100, 100), torch.arange(-99, 100, 99).tolist(), 20),
        ((1000, 1000), torch.arange(-999, 1000, 54).tolist(), 50),
    ]
    if request.config.getoption("--quick"):
        result = result[:4]
    return result


# ##############################################################################
# #
# ##############################################################################
def test_subdiags(
    rng_seeds,  # noqa: F811
    torch_devices,  # noqa: F811
    entry_atol,
    heights_widths_diags_meas,
):
    """Test SSVD on asymmetric exponential matrices.

    Sample random asymmetric matrices, and retrieve arbitrary (sub-)diagonals
    from them. Test that:
    * Sketched diagonals are close to actual diagonals
    * Sketches of linear combinations of diagonals are close to actual ones
    * Lowtri evaluations?
    """
    for seed in rng_seeds:
        for (h, w), diags, meas in heights_widths_diags_meas:
            for dtype in (torch.float64, torch.float32):
                for device in torch_devices:
                    # Just sample a random gaussian matrix
                    mat = gaussian_noise(
                        (h, w),
                        mean=0,
                        std=1.0,
                        seed=seed,
                        dtype=dtype,
                        device=device,
                    )
                    for diag_idx in diags:
                        # retrieve the true diag
                        diag = torch.diag(mat, diagonal=diag_idx)
                        for adjoint in (False, True):
                            # matrix-free estimation of the diag
                            do_stuff(
                                meas, mat, dtype, device, seed + 1, adjoint
                            )
                            # here estimate things
                            # then assert
                            breakpoint()

                    # assert torch.allclose(
                    #     S[: 2 * r], core_S[: 2 * r], atol=atol
                    # ), "Bad recovery of singular values!"
