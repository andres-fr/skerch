#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for ``TorchLinOpWrapper``."""


import numpy as np
import pytest
import torch

from skerch.linops import TorchLinOpWrapper, linop_to_matrix
from skerch.utils import gaussian_noise

from . import rng_seeds, torch_devices


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


# ##############################################################################
# # HELPERS
# ##############################################################################
class NumpyScalarLinOp:
    """A scalar matrix-free linop that works with numpy arrays only."""

    def __init__(self, scalar=1.0):
        """Initializer. See class docstring."""
        self.scalar = scalar

    def __matmul__(self, x):
        """Forward (right) matrix-vector multiplication ``self @ x``."""
        if not isinstance(x, np.ndarray):
            raise ValueError("Input not a numpy array!")
        return x * self.scalar

    def __rmatmul__(self, x):
        """Adjoint (left) matrix-vector multiplication ``x @ self``."""
        return self.__matmul__(x)

    def __repr__(self):
        """Test string"""
        return f"NumpyScalarLinOp({self.scalar})"


class TorchScalarLinOp(TorchLinOpWrapper, NumpyScalarLinOp):
    """A scalar matrix-free linop wrapped to work with torch tensors."""

    pass


# ##############################################################################
# # DIAGONAL TESTS
# ##############################################################################
def test_torchwrapper_formal(torch_devices, dtypes_tols):
    """Formal test case for diagonal linops.

    Tests that:
    * feeding tensor to lop raises error
    * feeding tensor of all types to wrapped lop now works and delivers correct
      results on correct device
    * repr
    """
    lop = NumpyScalarLinOp(2)
    torchlop = TorchScalarLinOp(2)
    for device in torch_devices:
        for dtype in dtypes_tols.keys():
            tnsr = torch.ones(5, dtype=dtype, device=device)
            arr = tnsr.cpu().numpy()
            # tensor to lop raises error
            with pytest.raises(ValueError):
                _ = lop @ tnsr
            with pytest.raises(ValueError):
                _ = tnsr @ lop
            # wrapped linop returns right tensor in right device and dtype
            v = torchlop @ tnsr
            assert (v == 2 * tnsr).all(), "Incorrect fwd wrapper?"
            assert v.dtype == tnsr.dtype, "Incorrect dtype in fwd wrapper?"
            assert v.device == tnsr.device, "Incorrect dtype in fwd wrapper?"
            #
            v = tnsr @ torchlop
            assert (v == 2 * tnsr).all(), "Incorrect adj wrapper?"
            assert v.dtype == tnsr.dtype, "Incorrect dtype in adj wrapper?"
            assert v.device == tnsr.device, "Incorrect dtype in adj wrapper?"
            # repr
            s1, s2 = str(lop), str(torchlop)
            assert s2 == f"TorchLinOpWrapper<{s1}>", "Wrong torchwrapper repr!"
