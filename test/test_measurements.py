#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Pytest for all noisy measurements

SSRFT

rademacher

gaussian



TODO:
* design a measurement linop that supports shape and @, but can also be
  parallelized if needed (e.g. via get_row or sth). Ideally this is all
  done through byvectorlinop, including ssrft

* once we have the measurement linops in place, test
  - seed consistency
  - (quasi-) orthogonality of randmats
  - formal corner cases




LATER TODO:

* Implement perform_measurement as per below.
  - test correctness and formal (valerr etc)
  - test that parallel versions are equal to inline
* Implement 3 recovery methods
  - test correctness and formal

* Implement sketched algorithms, at least svd and lord


TODO:

* add utest of phase_noise to utils
* create test for phase noise, also conj, check it looks OK
* same for SSRFT
* same for the measurement function, and we are done with meas


"""


import pytest
import torch

from skerch.linops import (
    linop_to_matrix,
    TransposedLinOp,
)

from skerch.measurements import (
    perform_measurement,
    RademacherNoiseLinOp,
    GaussianNoiseLinOp,
    PhaseNoiseLinOp,
)

from skerch.utils import BadShapeError, BadSeedError

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


@pytest.fixture
def complex_dtypes_tols():
    """Error tolerances for each complex dtype."""
    result = {
        torch.complex64: 1e-5,
        torch.complex128: 1e-10,
    }
    return result


@pytest.fixture
def noise_linop_types():
    """Class names for all noise linops to be tested.

    :returns: Collection of pairs ``(lop_type, is_complex_only)``
    """
    result = {
        (GaussianNoiseLinOp, False),
        (RademacherNoiseLinOp, False),
        (PhaseNoiseLinOp, True),
    }
    return result


# ##############################################################################
# # TESTS
# ##############################################################################
def test_measurements_formal(
    rng_seeds, torch_devices, dtypes_tols, noise_linop_types
):
    """Formal test case for measurement linops.

    For every noise linop tests:
    * Repr creates correct strings
    * Register triggers error if overlapping seeds are used if active
    * Get_vector triggers error for invalid index, and returns right dtype and
      device otherwise
    * Invalid index to ``get_vector`` triggers error
    * Deterministic behaviour (fwd and adjoint): running twice is same
    * Seed consistency
    * Output is in requested datatype and device

    For complex_only linops, it also tests:
    * Providing a noncomplex dtype raises a ``ValueError``
    """
    # correct string conversion
    hw = (3, 3)
    lop = RademacherNoiseLinOp(
        hw, 0, torch.float32, by_row=False, register=False
    )
    s = "<RademacherNoiseLinOp(3x3, seed=0, dtype=torch.float32, by_row=False)>"
    assert str(lop) == s, "Unexpected repr for Rademacher noise linop!"
    lop = GaussianNoiseLinOp(
        hw, 0, torch.float32, by_row=False, register=False
    )
    s = (
        "<GaussianNoiseLinOp(3x3, mean=0.0, std=1.0, seed=0, "
        + "dtype=torch.float32, by_row=False)>"
    )
    assert str(lop) == s, "Unexpected repr for Gaussian noise linop!"
    lop = PhaseNoiseLinOp(hw, 0, torch.complex64, by_row=False, register=False)
    s = (
        "<PhaseNoiseLinOp(3x3, conj=False, seed=0, "
        + "dtype=torch.complex64, by_row=False)>"
    )
    assert str(lop) == s, "Unexpected repr for Phase noise linop!"
    #
    for lop_type, complex_only in noise_linop_types:
        if complex_only:
            dtype1, dtype2 = torch.complex64, torch.complex128
        else:
            dtype1, dtype2 = torch.float32, torch.float64
        # dtype1 = torch.complex64 if lop_type is PhaseNoiseLinOp torch.float32
        lop = lop_type((5, 5), seed=0, dtype=dtype1, by_row=False)
        # register triggers for overlapping seeds regardless of other factors
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=0, dtype=dtype1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, dtype=dtype1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((20, 5), seed=1, dtype=dtype1, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, dtype=dtype2, by_row=False)
        with pytest.raises(BadSeedError):
            _ = lop_type((5, 5), seed=1, dtype=dtype1, by_row=True)
        # invalid index triggers error
        with pytest.raises(ValueError):
            lop.get_vector(idx=-1, device="cpu")
        with pytest.raises(ValueError):
            lop.get_vector(idx=5, device="cpu")
        #
        for seed in rng_seeds:
            for device in torch_devices:
                for dtype, tol in dtypes_tols.items():
                    #
                    if complex_only and dtype not in {
                        torch.complex32,
                        torch.complex64,
                        torch.complex128,
                    }:
                        with pytest.raises(ValueError):
                            _ = lop_type(hw, 0, dtype, register=False)
                        continue
                    #
                    hw = (100, 2)
                    lop1 = lop_type(
                        hw, seed, dtype, by_row=False, register=False
                    )
                    lop2 = lop_type(
                        hw, seed, dtype, by_row=False, register=False
                    )
                    lop3 = lop_type(
                        hw, seed + 5, dtype, by_row=False, register=False
                    )
                    # deterministic behaviour and seed consistency
                    mat1a = linop_to_matrix(
                        lop1, lop1.dtype, device, adjoint=False
                    )
                    mat1b = linop_to_matrix(
                        lop1, lop1.dtype, device, adjoint=False
                    )
                    mat1c = linop_to_matrix(
                        lop1, lop1.dtype, device, adjoint=True
                    )
                    mat2 = linop_to_matrix(
                        lop2, lop1.dtype, device, adjoint=False
                    )
                    mat3 = linop_to_matrix(
                        lop3, lop1.dtype, device, adjoint=False
                    )
                    assert (
                        mat1a == mat1b
                    ).all(), f"Nondeterministic linop? {lop1}"
                    assert (
                        mat1a == mat1c
                    ).all(), f"Different fwd and adjoint? {lop1}"
                    assert (
                        mat1a == mat2
                    ).all(), f"Same seed, differentl linop? {lop1}"
                    #
                    for col in mat1a.H:
                        cosim = abs(col @ mat3) / (col.norm() ** 2)
                        assert (
                            cosim < 0.5
                        ).all(), "Different seeds, similar vectors? {lop1}"
                    # dtype and device check
                    assert mat1a.dtype == dtype, "Mismatching dtype!"
                    assert mat1a.device.type == device, "Mismatching device!"


def test_phasenoise_conj(rng_seeds, torch_devices, complex_dtypes_tols):
    """Test case for conjugation of ``PhaseNoise`` linop.

    For all seeds, devices and complex dtypes, tests that:
    * Conj of linop produces elementwise conjugated entries
    """
    for seed in rng_seeds:
        for device in torch_devices:
            for dtype, tol in complex_dtypes_tols.items():
                lop = PhaseNoiseLinOp(
                    (5, 5), seed, dtype, conj=False, register=False
                )
                lop_conj = PhaseNoiseLinOp(
                    (5, 5), seed, dtype, conj=True, register=False
                )
                mat = linop_to_matrix(lop, lop.dtype, device, adjoint=False)
                mat_conj = linop_to_matrix(
                    lop_conj, lop.dtype, device, adjoint=False
                )
                assert (mat.conj() == mat_conj).all(), "Wrong conj linop?"
