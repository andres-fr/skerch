#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`algorithms`."""


import pytest
import torch
import numpy as np

from skerch.synthmat import RandomLordMatrix
from skerch.algorithms import ssvd
from . import rng_seeds, torch_devices


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
        ((50, 50), 5, 25, 35),
        ((40, 50), 5, 25, 35),
        ((50, 40), 5, 25, 35),
        ((1000, 800), 20, 50, 100),
    ]
    if request.config.getoption("--quick"):
        result = result[:3]
    return result


@pytest.fixture
def noise_types():
    """ """
    result = ["rademacher", "gaussian", "phase", "ssrft"]
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


# ##############################################################################
# # HELPERS
# ##############################################################################
def test_ssvd_correctness(
    rng_seeds, torch_devices, dtypes_tols, ssvd_recovery_shapes, noise_types
):
    """ """
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
                    lop = BasicMatrixLinOp(mat)
                    #
                    for noise_type in noise_types:
                        # singlepass recovery
                        U, S, Vh = ssvd(
                            lop,
                            device,
                            dtype,
                            outermeas,
                            seed + 1,
                            noise_type,
                            "singlepass",
                        )
                        breakpoint()
