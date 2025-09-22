#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""General ``pytest`` utilities, like fixtures.

.. note::

  See `docs <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_.
"""


import pytest
import torch
import numpy as np


# ##############################################################################
# # FIXTURES
# ##############################################################################
@pytest.fixture
def torch_devices():
    """Available PyTorch devices for the unit tests."""
    result = ["cpu"]
    if torch.cuda.is_available():
        result.append("cuda")
    return result


@pytest.fixture
def rng_seeds(request):
    """Random seeds for the unit tests.

    .. note::
     This implicitly uses code from :mod:`.conftest`.
    """
    result = request.config.getoption("--seeds")
    return result


# ##############################################################################
# # HELPERS
# ##############################################################################
def autocorrelation_test_helper(vec, delta_at_least=0.8, nondelta_at_most=0.1):
    """If vec is iid noise, its autocorrelation resembles a delta."""
    # normalize and compute unit-norm autocorrelation
    vec_mean = vec.mean()
    vec_norm = (vec - vec_mean).norm()
    vec = (vec - vec_mean) / vec_norm
    autocorr = torch.fft.fft(vec, norm="ortho")
    autocorr = (autocorr * autocorr.conj()) ** 0.5
    autocorr = abs(torch.fft.ifft(autocorr, norm="ortho"))
    # check that autocorrelation is close to standard unit delta
    assert abs(autocorr.norm() - 1) < 1e-5, "Autocorr should have unit norm"
    assert (
        autocorr[0] >= delta_at_least
    ), "Noise autocorr does not have a strong delta"
    assert (
        autocorr[1:] <= nondelta_at_most
    ).all(), "Noise autocorr has strong non-delta!"


def svd_test_helper(mat, I, U, S, Vh, atol):
    """Helper to test SVD output.

    Given the produced SVD of ``mat``, tests that:

    * The SVD is actually close to the matrix
    * ``U, V`` have orthonormal columns
    * Recovered ``S`` is a vector
    * The recovered ``S`` are nonnegative and in descending order
    * The devices and dtypes all match
    """
    allclose = torch.allclose if isinstance(S, torch.Tensor) else np.allclose
    diff = torch.diff if isinstance(S, torch.Tensor) else np.diff
    # correctness of result
    assert allclose(mat, (U * S) @ Vh, atol=atol), "Incorrect recovery!"
    # orthogonality of recovered U, V
    assert allclose(I, U.conj().T @ U, atol=atol), "U not orthogonal?"
    assert allclose(I, Vh @ Vh.conj().T, atol=atol), "V not orthogonal?"
    # svals given as vector
    assert S.shape == (len(I),), f"Svals not a vector? {S.shape}"
    # svals nonnegative and by descending magnitude
    assert (S >= 0).all(), "Negative svals!"
    assert (diff(S) <= 0).all(), "Ascending svals?"
    # matching device and type
    assert U.device == mat.device, "Incorrect U device!"
    assert S.device == mat.device, "Incorrect S device!"
    assert Vh.device == mat.device, "Incorrect V device!"
    assert U.dtype == mat.dtype, "Incorrect U dtype!"
    assert S.dtype == mat.real.dtype, "Incorrect S dtype!"
    assert Vh.dtype == mat.dtype, "Incorrect V dtype!"


def eigh_test_helper(mat, I, ews_rec, evs_rec, atol, by_mag=True):
    """Helper to test Hermitian eigendecomposition output.

    Given the produced EIGH of ``mat``, tests that:

    * The EIGH is actually close to the matrix
    * Eigenvectors are orthonormal columns
    * Recovered eigvals given as a vector
    * The recovered eigvals are by descending magnitude/value
    * The devices and dtypes all match
    """
    allclose = torch.allclose if isinstance(I, torch.Tensor) else np.allclose
    diff = torch.diff if isinstance(I, torch.Tensor) else np.diff
    V, Lbd, Vh = evs_rec, ews_rec, evs_rec.conj().T
    # correctness of result
    assert allclose(mat, (V * Lbd) @ Vh, atol=atol), "Incorrect recovery!"
    # orthogonality of recovered V
    assert allclose(I, Vh @ V, atol=atol), "Eigvecs not orthogonal?"
    # eigvals given as vector
    assert Lbd.shape == (len(Vh),), f"Eigvals not a vector? {Lbd.shape}"
    # Eigvals by descending magnitude
    Lbd_sorted = abs(Lbd) if by_mag else Lbd
    assert (
        diff(Lbd_sorted) <= 0
    ).all(), f"Eigvals in wrong order? (by_mag={by_mag})"
    # matching device and type
    assert V.device == mat.device, "Incorrect eigvecs device!"
    assert Lbd.device == mat.device, "Incorrect eigvals device!"
    assert V.dtype == mat.dtype, "Incorrect eigvecs dtype!"
    assert Lbd.dtype == mat.real.dtype, "Incorrect eigvals dtype!"
