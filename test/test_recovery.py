#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for :mod:`recovery`."""


import pytest
import torch
from skerch.linops import (
    linop_to_matrix,
    TransposedLinOp,
)

from skerch.measurements import (
    perform_measurements,
    RademacherNoiseLinOp,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
    SSRFT,
    SsrftNoiseLinOp,
)

from skerch.utils import gaussian_noise, BadShapeError, BadSeedError

from . import rng_seeds, torch_devices, autocorrelation_test_helper


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def dtypes_tols():
    """Error tolerances for each dtype."""
    result = {
        torch.float32: 3e-5,
        torch.complex64: 1e-5,
        torch.float64: 1e-10,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def parallel_modes():
    result = (None, "mp")
    return result


@pytest.fixture
def complex_dtypes_tols():
    """Error tolerances for each complex dtype."""
    result = {
        torch.complex64: 1e-5,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def iid_noise_linop_types():
    """Class names for all noise linops to be tested.

    :returns: Collection of pairs ``(lop_type, is_complex_only)``
    """
    result = {
        (GaussianNoiseLinOp, False),
        (RademacherNoiseLinOp, False),
        (PhaseNoiseLinOp, True),
    }
    return result


@pytest.fixture
def iid_hw_and_autocorr_tolerances():
    """Error tolerances for each complex dtype."""
    hw = (20, 20)
    delta_at_least = 0.7
    nondelta_at_most = 0.5
    return hw, delta_at_least, nondelta_at_most


@pytest.fixture
def ssrft_hw_and_autocorr_tolerances():
    """Error tolerances for each complex dtype."""
    hw = (30, 10)
    delta_at_least = 0.7
    nondelta_at_most = 0.5
    return hw, delta_at_least, nondelta_at_most


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


def get_meas_vec(idx, meas_lop, device=None, dtype=None, conj=False):
    """ """
    if isinstance(meas_lop, torch.Tensor):
        result = meas_lop[:, idx]
    elif isinstance(meas_lop, SsrftNoiseLinOp):
        if device is None or dtype is None:
            raise ValueError("SsrftNoiseLinop requires device and dtype!")
        result = meas_lop.get_vector(idx, device, dtype, by_row=False)
    else:
        result = meas_lop.get_vector(idx, device)
    #
    return result.conj() if conj else result


# ##############################################################################
# # PERFORM MEASUREMENTS
# ##############################################################################
