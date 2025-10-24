#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""General ``pytest`` utilities, like fixtures.

.. note::

  See `docs <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_.
"""


import numpy as np
import pytest
import torch


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
# # TEST LINOPS
# ##############################################################################
class BasicMatrixLinOp:
    """Intentionally simple linop, only supporting ``shape`` and @."""

    def __init__(self, matrix):
        """Creates linop."""
        self.matrix = matrix
        self.shape = matrix.shape

    def __matmul__(self, x):
        """Implements self @ x."""
        return self.matrix @ x

    def __rmatmul__(self, x):
        """Implements x @ self."""
        return x @ self.matrix


# ##############################################################################
# # TEST METRICS
# ##############################################################################
def relerr(ori, rec, squared=True):
    """Relative error in the form ``(frob(ori - rec) / frob(ori))**2``."""
    result = (ori - rec).norm() / ori.norm()
    if squared:
        result = result**2
    return result


def relsumerr(ori_sum, rec_sum, ori_vec, squared=True):
    """Relative error of a sum of estimators.

    The error for adding N estimators increases with ``sqrt(N)`` times the
    norm of said estimators, because:
    ``(1^T ori) - (1^T rec) = 1^T (ori - rec)``, and the norm of this, by
    Cauchy-Schwarz, is bounded as:
    ``norm(1^T (ori - rec)) <= norm(1)*norm(ori-rec) = sqrt(N)*norm(ori-rec)``.

    So, for the sum of entries, we apply ``relerr``, but divided by ``sqrt(N)``
    to account for this factor:

    ``| ori_sum - rec_sum |`` / (sqrt(N) * norm(ori_vec))``.

    This is consistent in the sense that, if rec_vec is close to ori_vec by
    0.001, this metric will also output at most 0.001 for the estimated sum.
    """
    result = abs(ori_sum - rec_sum) / (len(ori_vec) ** 0.5 * ori_vec.norm())
    if squared:
        result = result**2
    return result


# ##############################################################################
# # TEST HELPERS
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


def svd_test_helper(mat, idty, U, S, Vh, atol):
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
    assert allclose(idty, U.conj().T @ U, atol=atol), "U not orthogonal?"
    assert allclose(idty, Vh @ Vh.conj().T, atol=atol), "V not orthogonal?"
    # svals given as vector
    assert S.shape == (len(idty),), f"Svals not a vector? {S.shape}"
    # svals nonnegative and by descending magnitude
    assert (S >= 0).all(), "Negative svals!"
    assert (diff(S) <= 0).all(), "Ascending svals?"
    # matching device and type
    assert U.dtype == mat.dtype, "Incorrect U dtype!"
    assert S.dtype == mat.real.dtype, "Incorrect S dtype!"
    assert Vh.dtype == mat.dtype, "Incorrect V dtype!"
    if isinstance(mat, torch.Tensor):
        assert U.device == mat.device, "Incorrect U device!"
        assert S.device == mat.device, "Incorrect S device!"
        assert Vh.device == mat.device, "Incorrect V device!"


def eigh_test_helper(mat, idty, ews_rec, evs_rec, atol, by_mag=True):
    """Helper to test Hermitian eigendecomposition output.

    Given the produced EIGH of ``mat``, tests that:

    * The EIGH is actually close to the matrix
    * Eigenvectors are orthonormal columns
    * Recovered eigvals given as a vector
    * The recovered eigvals are by descending magnitude/value
    * The devices and dtypes all match
    """
    allclose = torch.allclose if isinstance(idty, torch.Tensor) else np.allclose
    diff = torch.diff if isinstance(idty, torch.Tensor) else np.diff
    V, Lbd, Vh = evs_rec, ews_rec, evs_rec.conj().T
    # correctness of result
    assert allclose(mat, (V * Lbd) @ Vh, atol=atol), "Incorrect recovery!"
    # orthogonality of recovered V
    assert allclose(idty, Vh @ V, atol=atol), "Eigvecs not orthogonal?"
    # eigvals given as vector
    assert Lbd.shape == (len(Vh),), f"Eigvals not a vector? {Lbd.shape}"
    # Eigvals by descending magnitude
    Lbd_sorted = abs(Lbd) if by_mag else Lbd
    assert (
        diff(Lbd_sorted) <= 0
    ).all(), f"Eigvals in wrong order? (by_mag={by_mag})"
    # matching device and type
    assert V.dtype == mat.dtype, "Incorrect eigvecs dtype!"
    assert Lbd.dtype == mat.real.dtype, "Incorrect eigvals dtype!"
    if isinstance(mat, torch.Tensor):
        assert V.device == mat.device, "Incorrect eigvecs device!"
        assert Lbd.device == mat.device, "Incorrect eigvals device!"


def diag_trace_test_helper(
    diag, tr, idty, results, tr_tol, diag_tol, q_tol, errcode=""
):
    """Helper to test correctness of diag/trace estimators.

    * ``results["tr"]`` via relsumerr
    * If present, ``results["diag"]`` via relerr
    * If present, orthogonality of ``results["Q"]``
    """
    # tr
    assert (
        relsumerr(tr, results["tr"], diag) < tr_tol
    ), f"[{errcode}]: Bad trace?"
    # diag
    if "diag" in results:
        err = relerr(diag, results["diag"])
        assert err < diag_tol, f"[{errcode}]: Bad diag? {err}"
    # orth Q
    if "Q" in results:
        Q = results["Q"]
        assert torch.allclose(
            Q.H @ Q, idty, atol=q_tol
        ), f"[{errcode}]: Q not orthogonal?"
