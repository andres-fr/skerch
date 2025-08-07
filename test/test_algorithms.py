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


# ##############################################################################
# # HELPERS
# ##############################################################################
def test_ssvd_correctness(
    rng_seeds, torch_devices, dtypes_tols, ssvd_recovery_shapes
):
    """ """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in dtypes_tols.items():
                for hw, rank, outermeas, innermeas in ssvd_recovery_shapes:
                    breakpoint()
